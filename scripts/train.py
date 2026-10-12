"""Train the binary "same scene" PairClassifierHead on frozen DINOv3 descriptors.

1. Cache DINOv3 descriptors of every train image in its 4 rot90 versions
   (train_embeddings_dinov3_4rotations.h5). Datasets in --datasets-excluded-from-training
   (default ETs, stairs: they are also in data/test) are never loaded.
2. Each epoch samples pairs inside a dataset, balanced across datasets:
     50%  same scene (neither image an outlier), both rotated by the same random r   -> label 1
     25%  same scene, b rotated by r + d, d in {1, 2, 3}                             -> label 0
     25%  different scenes or an outlier image, both rotated by r                     -> label 0
   The rotated positives teach the head that only the right relative rotation matches, which
   is what step 1's argmax over 4 rotations relies on.
3. Validation on held-out train datasets: ROC-AUC of the step-1 max score, rotation accuracy
   on synthetically rotated positives, and outlier precision/recall of the step-2 rule.
4. Saves the best (mean val AUC) head to pair_classifier_head_dinov3.pt.

    python scripts/train.py --competition-data-dir /kaggle/input/image-matching-challenge-2025 \
        --dinov3-model-dir facebook/dinov3-vitb16-pretrain-lvd1689m --output-dir /kaggle/working/pair_classifier
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

from sc4000.config import ROTATION_COUNT, build_argument_parser, config_from_arguments  # noqa: E402
from sc4000.dataset_io import dataset_image_dir, labels_csv_path  # noqa: E402
from sc4000.models.dinov3_global_descriptor import Dinov3GlobalDescriptor  # noqa: E402
from sc4000.models.pair_classifier_head import PairClassifierHead, save_pair_classifier_head  # noqa: E402
from sc4000.step1_pair_similarity_dinov3 import score_all_pairs_with_four_rotations  # noqa: E402
from sc4000.step2_outlier_removal_and_pair_selection import find_outlier_image_mask  # noqa: E402

OUTLIER_SCENE_NAME = "outliers"


# ---------------------------------------------------------------------------------------- data


class TrainDataset:
    """One competition dataset: image names, scene labels and descriptors [N, 4, D]."""

    def __init__(self, name: str, image_names: "list[str]", scene_names: "list[str]", descriptors: np.ndarray):
        self.name = name
        self.image_names = image_names
        self.scene_names = np.array(scene_names)
        self.descriptors_in_four_rotations = descriptors
        upper_i, upper_j = np.triu_indices(len(image_names), k=1)
        is_outlier = self.scene_names == OUTLIER_SCENE_NAME
        same_scene = (self.scene_names[upper_i] == self.scene_names[upper_j]) & ~is_outlier[upper_i] & ~is_outlier[upper_j]
        self.positive_pairs = np.stack([upper_i[same_scene], upper_j[same_scene]], axis=1)
        self.negative_pairs = np.stack([upper_i[~same_scene], upper_j[~same_scene]], axis=1)

    def same_scene_label_matrix(self) -> np.ndarray:
        is_outlier = self.scene_names == OUTLIER_SCENE_NAME
        same_scene = self.scene_names[:, None] == self.scene_names[None, :]
        return same_scene & ~is_outlier[:, None] & ~is_outlier[None, :]


def cache_train_descriptors(config, cache_path: Path, dataset_names: "list[str]", labels: pd.DataFrame) -> None:
    """Compute missing DINOv3 descriptors into cache_path[dataset][image] -> [4, D]."""
    with h5py.File(cache_path, "a") as cache_file:
        missing = [
            (row.dataset, row.image) for row in labels.itertuples(index=False)
            if row.dataset in dataset_names and row.image not in cache_file.get(row.dataset, {})
        ]
        if not missing:
            return
        descriptor_model = Dinov3GlobalDescriptor(config.dinov3_model_dir, config.device, config.global_descriptor_resize_long_side)
        for dataset_name, image_name in tqdm(missing, desc="caching DINOv3 x4 rotations", dynamic_ncols=True):
            descriptors = descriptor_model.describe_image_in_four_rotations(
                dataset_image_dir(config, "train", dataset_name) / image_name
            )
            cache_file.require_group(dataset_name).create_dataset(image_name, data=descriptors)
        del descriptor_model
        torch.cuda.empty_cache()


def load_train_datasets(cache_path: Path, dataset_names: "list[str]", labels: pd.DataFrame) -> "list[TrainDataset]":
    train_datasets = []
    with h5py.File(cache_path, "r") as cache_file:
        for dataset_name in dataset_names:
            dataset_labels = labels[labels.dataset == dataset_name]
            descriptors = np.stack([cache_file[dataset_name][image][()] for image in dataset_labels.image])
            train_datasets.append(TrainDataset(dataset_name, list(dataset_labels.image), list(dataset_labels.scene), descriptors))
    return train_datasets


# ---------------------------------------------------------------------------------------- pair sampling


def sample_epoch_pairs(train_datasets: "list[TrainDataset]", pair_count: int, random_generator: np.random.Generator):
    """Returns global index arrays (a, b), rotations (rotation_a, rotation_b) and float labels."""
    dataset_offsets = np.cumsum([0] + [len(d.image_names) for d in train_datasets])
    kinds = random_generator.choice(3, size=pair_count, p=[0.5, 0.25, 0.25])  # 0 positive, 1 rotated positive, 2 negative
    index_a, index_b = np.zeros(pair_count, np.int64), np.zeros(pair_count, np.int64)
    valid = np.zeros(pair_count, bool)
    dataset_choice = random_generator.integers(len(train_datasets), size=pair_count)
    for dataset_index, train_dataset in enumerate(train_datasets):
        for kind in (0, 1, 2):
            selected = np.flatnonzero((dataset_choice == dataset_index) & (kinds == kind))
            pool = train_dataset.negative_pairs if kind == 2 else train_dataset.positive_pairs
            if len(selected) == 0 or len(pool) == 0:
                continue  # e.g. single-scene datasets have no negatives; those samples are dropped
            picked = pool[random_generator.integers(len(pool), size=len(selected))]
            swap = random_generator.random(len(selected)) < 0.5
            picked[swap] = picked[swap][:, ::-1]
            index_a[selected] = picked[:, 0] + dataset_offsets[dataset_index]
            index_b[selected] = picked[:, 1] + dataset_offsets[dataset_index]
            valid[selected] = True
    shared_rotation = random_generator.integers(ROTATION_COUNT, size=pair_count)
    rotation_offset = np.where(kinds == 1, random_generator.integers(1, ROTATION_COUNT, size=pair_count), 0)
    labels = (kinds == 0).astype(np.float32)
    return (
        index_a[valid], index_b[valid],
        shared_rotation[valid], ((shared_rotation + rotation_offset) % ROTATION_COUNT)[valid],
        labels[valid],
    )


# ---------------------------------------------------------------------------------------- validation


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive_count, negative_count = labels.sum(), (~labels).sum()
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    return float((ranks[labels].sum() - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


@torch.no_grad()
def evaluate_head(head, val_datasets: "list[TrainDataset]", config, device) -> dict:
    head.eval()
    metrics = {}
    for val_dataset in val_datasets:
        pair_max_score, _ = score_all_pairs_with_four_rotations(val_dataset.descriptors_in_four_rotations, head, device)
        upper_i, upper_j = np.triu_indices(len(val_dataset.image_names), k=1)
        label_matrix = val_dataset.same_scene_label_matrix()
        auc = roc_auc(pair_max_score[upper_i, upper_j], label_matrix[upper_i, upper_j])

        # rotation accuracy: rotate image b of each positive pair by d; the argmax must undo it
        descriptors = torch.from_numpy(val_dataset.descriptors_in_four_rotations).float().to(device)
        positive_pairs = val_dataset.positive_pairs[:20000]
        correct, total = 0, 0
        for rotation_d in range(ROTATION_COUNT):
            upright_a = descriptors[positive_pairs[:, 0], 0][:, None].expand(-1, ROTATION_COUNT, -1)
            rotated_b = descriptors[positive_pairs[:, 1]][:, [(rotation_d + k) % ROTATION_COUNT for k in range(ROTATION_COUNT)]]
            predicted_k = head(upright_a, rotated_b).argmax(dim=1).cpu().numpy()
            correct += int((predicted_k == (ROTATION_COUNT - rotation_d) % ROTATION_COUNT).sum())
            total += len(positive_pairs)

        predicted_outlier = find_outlier_image_mask(pair_max_score, config.pair_similarity_threshold, config.min_neighbor_count_for_inlier)
        true_outlier = val_dataset.scene_names == OUTLIER_SCENE_NAME
        true_positive = int((predicted_outlier & true_outlier).sum())
        metrics[val_dataset.name] = {
            "auc": auc,
            "rotation_accuracy": correct / max(total, 1),
            "outlier_precision": true_positive / max(int(predicted_outlier.sum()), 1),
            "outlier_recall": true_positive / max(int(true_outlier.sum()), 1) if true_outlier.any() else float("nan"),
            "predicted_outlier_count": int(predicted_outlier.sum()),
            "true_outlier_count": int(true_outlier.sum()),
        }
    return metrics


# ---------------------------------------------------------------------------------------- main


def main() -> None:
    parser = build_argument_parser(__doc__.splitlines()[0], needs_dataset=False)
    parser.add_argument("--val-datasets", nargs="+", default=["imc2023_haiper", "imc2024_lizard_pond"])
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/pair_classifier"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pairs-per-epoch", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    config = config_from_arguments(arguments)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    random_generator = np.random.default_rng(arguments.seed)
    torch.manual_seed(arguments.seed)

    labels = pd.read_csv(labels_csv_path(config, "train"))
    excluded = set(config.datasets_excluded_from_training)
    all_names = [name for name in labels.dataset.unique() if name not in excluded]
    val_names = [name for name in arguments.val_datasets if name in all_names]
    train_names = [name for name in all_names if name not in val_names]
    assert not excluded & (set(train_names) | set(val_names)), "excluded datasets leaked into training"
    print(f"train datasets: {train_names}\nval datasets:   {val_names}\nexcluded:       {sorted(excluded)}")

    cache_path = arguments.output_dir / "train_embeddings_dinov3_4rotations.h5"
    cache_train_descriptors(config, cache_path, train_names + val_names, labels)
    train_datasets = load_train_datasets(cache_path, train_names, labels)
    val_datasets = load_train_datasets(cache_path, val_names, labels)

    all_train_descriptors = torch.from_numpy(
        np.concatenate([d.descriptors_in_four_rotations for d in train_datasets])
    ).float().to(device)
    head = PairClassifierHead(input_dimension=all_train_descriptors.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arguments.epochs)

    best_mean_auc = -1.0
    checkpoint_path = arguments.output_dir / "pair_classifier_head_dinov3.pt"
    for epoch in range(1, arguments.epochs + 1):
        epoch_start_time = time.time()
        head.train()
        index_a, index_b, rotation_a, rotation_b, pair_labels = sample_epoch_pairs(
            train_datasets, arguments.pairs_per_epoch, random_generator
        )
        permutation = random_generator.permutation(len(pair_labels))
        epoch_loss = 0.0
        for batch_start in range(0, len(permutation), arguments.batch_size):
            batch = permutation[batch_start:batch_start + arguments.batch_size]
            descriptor_a = all_train_descriptors[index_a[batch], rotation_a[batch]]
            descriptor_b = all_train_descriptors[index_b[batch], rotation_b[batch]]
            target = torch.from_numpy(pair_labels[batch]).to(device)
            loss = F.binary_cross_entropy_with_logits(head(descriptor_a, descriptor_b), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        scheduler.step()

        val_metrics = evaluate_head(head, val_datasets, config, device)
        mean_auc = float(np.nanmean([m["auc"] for m in val_metrics.values()]))
        mean_rotation_accuracy = float(np.mean([m["rotation_accuracy"] for m in val_metrics.values()]))
        print(f"epoch {epoch:3d} loss {epoch_loss / len(permutation):.4f} val_auc {mean_auc:.4f} "
              f"val_rotation_acc {mean_rotation_accuracy:.4f} ({time.time() - epoch_start_time:.1f}s)")
        for name, metric_values in val_metrics.items():
            print(f"    {name}: " + " ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                                         for key, value in metric_values.items()))
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            save_pair_classifier_head(head, checkpoint_path, {
                "backbone": "dinov3",
                "dinov3_model_dir": config.dinov3_model_dir,
                "global_descriptor_resize_long_side": config.global_descriptor_resize_long_side,
                "train_datasets": train_names,
                "val_datasets": val_names,
                "epoch": epoch,
                "val_metrics": val_metrics,
            })
            print(f"    saved best head -> {checkpoint_path}")
    print(f"best mean val AUC {best_mean_auc:.4f}")


if __name__ == "__main__":
    main()
