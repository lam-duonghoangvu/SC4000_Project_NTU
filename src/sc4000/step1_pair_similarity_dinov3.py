"""STEP 1 - rotation-aware pair similarity with DINOv3.

Every pair (i, j) is compared 4 times: image i as-is against image j rotated by k * 90 degrees
counter-clockwise, k = 0..3. We keep the highest score and the k that produced it:

    pair_max_score[i, j]           = max_k  score(i, rot90(j, k))
    pair_relative_rotation_k[i, j] = argmax_k score(i, rot90(j, k))   ("rotate j by this k to look like i")

The matrices are mirrored, so pair_relative_rotation_k[j, i] = (4 - pair_relative_rotation_k[i, j]) % 4.

Output: <working_dir>/<dataset>/step1_pair_scores_dinov3.npz
"""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sc4000.config import (
    ROTATION_COUNT,
    STEP1_PAIR_SCORES_FILENAME,
    PipelineConfig,
    build_argument_parser,
    config_from_arguments,
)
from sc4000.dataset_io import dataset_image_dir, list_dataset_image_names
from sc4000.models.dinov3_global_descriptor import Dinov3GlobalDescriptor
from sc4000.models.pair_classifier_head import PairClassifierHead, load_pair_classifier_head


class RotationAwarePairScorer:
    """Bundles the frozen DINOv3 backbone and the (optional) trained pair classifier head."""

    def __init__(self, config: PipelineConfig):
        self.descriptor_model = Dinov3GlobalDescriptor(
            config.dinov3_model_dir, config.device, config.global_descriptor_resize_long_side
        )
        self.device = self.descriptor_model.device
        self.pair_classifier_head: "PairClassifierHead | None" = None
        checkpoint_path = Path(config.pair_classifier_checkpoint_path)
        if config.use_cosine_similarity_without_classifier:
            print("[step1] using cosine similarity (classifier disabled by config)")
        elif checkpoint_path.exists():
            self.pair_classifier_head = load_pair_classifier_head(checkpoint_path, str(self.device))
            print(f"[step1] loaded pair classifier head from {checkpoint_path}")
        else:
            print(f"[step1] WARNING: no classifier at {checkpoint_path}, falling back to cosine similarity. "
                  "Tune --pair-similarity-threshold for cosine scores.")

    def describe_images(self, image_paths: "list[Path]") -> np.ndarray:
        """float32 [N, ROTATION_COUNT, D]."""
        descriptors = [
            self.descriptor_model.describe_image_in_four_rotations(path)
            for path in tqdm(image_paths, desc="[step1] DINOv3 x4 rotations", dynamic_ncols=True)
        ]
        return np.stack(descriptors)


@torch.no_grad()
def score_all_pairs_with_four_rotations(
    descriptors_in_four_rotations: np.ndarray,
    pair_classifier_head: "PairClassifierHead | None",
    device: "torch.device | str" = "cpu",
) -> "tuple[np.ndarray, np.ndarray]":
    """descriptors [N, 4, D] -> (pair_max_score [N, N], pair_relative_rotation_k [N, N]); diagonal = -inf / 0."""
    descriptors = torch.from_numpy(descriptors_in_four_rotations).float().to(device)
    image_count, rotation_count, descriptor_dimension = descriptors.shape
    all_rotated_descriptors = descriptors.reshape(image_count * rotation_count, descriptor_dimension)

    # score_per_rotation[i, j, k] = score(i at rotation 0, j at rotation k)
    score_per_rotation = torch.empty(image_count, image_count, rotation_count, device=device)
    for i in range(image_count):
        upright_descriptor_i = descriptors[i, 0].expand_as(all_rotated_descriptors)
        if pair_classifier_head is None:
            row_scores = (upright_descriptor_i * all_rotated_descriptors).sum(dim=-1)
        else:
            row_scores = torch.sigmoid(pair_classifier_head(upright_descriptor_i, all_rotated_descriptors))
        score_per_rotation[i] = row_scores.view(image_count, rotation_count)

    pair_max_score_full, pair_relative_rotation_k_full = score_per_rotation.max(dim=-1)
    pair_max_score_full = pair_max_score_full.cpu().numpy()
    pair_relative_rotation_k_full = pair_relative_rotation_k_full.cpu().numpy()

    # Keep the upper triangle and mirror it, so (i, j) and (j, i) always agree.
    pair_max_score = np.full((image_count, image_count), -np.inf, dtype=np.float32)
    pair_relative_rotation_k = np.zeros((image_count, image_count), dtype=np.int64)
    upper_i, upper_j = np.triu_indices(image_count, k=1)
    pair_max_score[upper_i, upper_j] = pair_max_score_full[upper_i, upper_j]
    pair_max_score[upper_j, upper_i] = pair_max_score_full[upper_i, upper_j]
    pair_relative_rotation_k[upper_i, upper_j] = pair_relative_rotation_k_full[upper_i, upper_j]
    pair_relative_rotation_k[upper_j, upper_i] = (ROTATION_COUNT - pair_relative_rotation_k_full[upper_i, upper_j]) % ROTATION_COUNT
    return pair_max_score, pair_relative_rotation_k


def run_step1_pair_similarity(
    config: PipelineConfig, split: str, dataset_name: str, pair_scorer: "RotationAwarePairScorer | None" = None
) -> Path:
    pair_scorer = pair_scorer or RotationAwarePairScorer(config)
    image_names = list_dataset_image_names(config, split, dataset_name)
    image_dir = dataset_image_dir(config, split, dataset_name)
    descriptors_in_four_rotations = pair_scorer.describe_images([image_dir / name for name in image_names])
    pair_max_score, pair_relative_rotation_k = score_all_pairs_with_four_rotations(
        descriptors_in_four_rotations, pair_scorer.pair_classifier_head, pair_scorer.device
    )
    output_path = config.dataset_working_dir(dataset_name) / STEP1_PAIR_SCORES_FILENAME
    np.savez(
        output_path,
        image_names=np.array(image_names),
        pair_max_score=pair_max_score,
        pair_relative_rotation_k=pair_relative_rotation_k,
    )
    print(f"[step1] {dataset_name}: scored {len(image_names)} images -> {output_path}")
    return output_path


def load_step1_pair_scores(config: PipelineConfig, dataset_name: str) -> "tuple[list[str], np.ndarray, np.ndarray]":
    with np.load(config.dataset_working_dir(dataset_name) / STEP1_PAIR_SCORES_FILENAME) as step1_output:
        return (
            [str(name) for name in step1_output["image_names"]],
            step1_output["pair_max_score"],
            step1_output["pair_relative_rotation_k"],
        )


def main() -> None:
    arguments = build_argument_parser(__doc__.splitlines()[1]).parse_args()
    run_step1_pair_similarity(config_from_arguments(arguments), arguments.split, arguments.dataset)


if __name__ == "__main__":
    main()
