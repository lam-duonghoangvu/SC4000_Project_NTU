"""STEP 6 - write submission.csv from the step-5 poses and, on the train split, score it.

Images without a pose (step-2 outliers or not registered by COLMAP) are written as scene
"outliers" with NaN poses. On the train split the score uses imc25-utils `metric.score` against
train_labels.csv restricted to the processed datasets, since the metric requires every
ground-truth row to be present.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sc4000.config import PipelineConfig, build_argument_parser, config_from_arguments
from sc4000.dataset_io import ImagePrediction, labels_csv_path, load_image_predictions_by_dataset
from sc4000.step5_reconstruction_colmap import load_step5_registered_poses


def _array_to_string(values) -> str:
    return ";".join(f"{value:.09f}" for value in values)


def _nan_string(count: int) -> str:
    return ";".join(["nan"] * count)


def fill_predictions_from_step5(config: PipelineConfig, predictions: "list[ImagePrediction]", dataset_name: str) -> None:
    registered_poses = load_step5_registered_poses(config, dataset_name)
    for prediction in predictions:
        pose = registered_poses.get(prediction.filename)
        if pose is None:
            continue
        prediction.cluster_index = pose["cluster_index"]
        prediction.rotation_matrix = np.asarray(pose["rotation_matrix"])
        prediction.translation_vector = np.asarray(pose["translation_vector"])


def write_submission_csv(predictions_by_dataset: "dict[str, list[ImagePrediction]]", split: str, submission_csv_path: Path) -> None:
    rows = []
    for predictions in predictions_by_dataset.values():
        for prediction in predictions:
            row = {
                "dataset": prediction.dataset,
                "scene": "outliers" if prediction.cluster_index is None else f"cluster{prediction.cluster_index}",
                "image": prediction.filename,
                "rotation_matrix": _nan_string(9) if prediction.rotation_matrix is None else _array_to_string(prediction.rotation_matrix),
                "translation_vector": _nan_string(3) if prediction.translation_vector is None else _array_to_string(prediction.translation_vector),
            }
            if split == "test":
                row = {"image_id": prediction.image_id, **row}
            rows.append(row)
    submission_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(submission_csv_path, index=False)
    print(f"[step6] wrote {len(rows)} rows -> {submission_csv_path}")


def score_train_submission(config: PipelineConfig, submission_csv_path: Path, dataset_names: "list[str]") -> float:
    if config.imc25_utils_dir not in sys.path:
        sys.path.append(config.imc25_utils_dir)
    import metric

    ground_truth = pd.read_csv(labels_csv_path(config, "train"))
    ground_truth_subset_path = Path(config.working_dir) / "step6_train_labels_processed_datasets.csv"
    ground_truth[ground_truth.dataset.isin(dataset_names)].to_csv(ground_truth_subset_path, index=False)
    (final_score, _, _), (score_by_dataset, _, _) = metric.score(
        gt_csv=str(ground_truth_subset_path),
        user_csv=str(submission_csv_path),
        thresholds_csv=str(Path(config.competition_data_dir) / "train_thresholds.csv"),
        mask_csv=None,
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )
    for dataset_name, dataset_score in score_by_dataset.items():
        print(f"[step6] {dataset_name}: {dataset_score:.2f}")
    print(f"[step6] final score over {len(dataset_names)} datasets: {final_score:.2f}")
    return final_score


def run_step6_submission(config: PipelineConfig, split: str, dataset_names: "list[str]") -> None:
    predictions_by_dataset = load_image_predictions_by_dataset(config, split)
    if split == "train":  # the train submission only covers what was processed, matching the filtered GT
        predictions_by_dataset = {name: predictions_by_dataset[name] for name in dataset_names}
    for dataset_name in dataset_names:
        fill_predictions_from_step5(config, predictions_by_dataset[dataset_name], dataset_name)
    submission_csv_path = Path(config.submission_csv_path)
    write_submission_csv(predictions_by_dataset, split, submission_csv_path)
    if split == "train":
        score_train_submission(config, submission_csv_path, dataset_names)


def main() -> None:
    parser = build_argument_parser(__doc__.splitlines()[1], needs_dataset=False)
    parser.add_argument("--datasets", nargs="+", required=True)
    arguments = parser.parse_args()
    run_step6_submission(config_from_arguments(arguments), arguments.split, arguments.datasets)


if __name__ == "__main__":
    main()
