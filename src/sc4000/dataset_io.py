"""Reading the competition CSVs, image paths and the rot90 keypoint geometry shared by all steps."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sc4000.config import PipelineConfig


@dataclasses.dataclass
class ImagePrediction:
    image_id: "str | None"  # only used on the hidden test set
    dataset: str
    filename: str
    ground_truth_scene: "str | None" = None  # only known on the train split
    cluster_index: "int | None" = None
    rotation_matrix: "np.ndarray | None" = None
    translation_vector: "np.ndarray | None" = None


def labels_csv_path(config: PipelineConfig, split: str) -> Path:
    filename = "train_labels.csv" if split == "train" else "sample_submission.csv"
    return Path(config.competition_data_dir) / filename


def load_image_predictions_by_dataset(config: PipelineConfig, split: str) -> "dict[str, list[ImagePrediction]]":
    """{dataset: [ImagePrediction]} in CSV order. For the test split the scene column is meaningless."""
    labels = pd.read_csv(labels_csv_path(config, split))
    predictions_by_dataset: "dict[str, list[ImagePrediction]]" = {}
    for row in labels.itertuples(index=False):
        predictions_by_dataset.setdefault(row.dataset, []).append(ImagePrediction(
            image_id=None if split == "train" else row.image_id,
            dataset=row.dataset,
            filename=row.image,
            ground_truth_scene=row.scene if split == "train" else None,
        ))
    return predictions_by_dataset


def dataset_image_dir(config: PipelineConfig, split: str, dataset_name: str) -> Path:
    return Path(config.competition_data_dir) / split / dataset_name


def list_dataset_image_names(config: PipelineConfig, split: str, dataset_name: str) -> "list[str]":
    """Image file names of a dataset, taken from the CSV so train and test behave the same."""
    predictions_by_dataset = load_image_predictions_by_dataset(config, split)
    image_dir = dataset_image_dir(config, split, dataset_name)
    return [p.filename for p in predictions_by_dataset[dataset_name] if (image_dir / p.filename).exists()]


# ---------------------------------------------------------------------------------------- rot90 geometry


def rotated_image_shape(original_height: int, original_width: int, rotation_k: int) -> "tuple[int, int]":
    """(height, width) of np.rot90(image, rotation_k)."""
    return (original_width, original_height) if rotation_k % 2 == 1 else (original_height, original_width)


def map_keypoints_from_rotated_to_original(
    keypoints_xy_in_rotated_image: np.ndarray,
    original_height: int,
    original_width: int,
    rotation_k: int,
) -> np.ndarray:
    """Map (x, y) pixel-index coordinates in np.rot90(original, rotation_k) back into the original image.

    One counter-clockwise rot90 maps original pixel (x, y) of an image with width W to
    (y, W - 1 - x); its inverse is (x', y') -> (W - 1 - y', x'). Applying the inverse
    rotation_k times walks back through the intermediate rotated images.
    """
    keypoints_xy = np.asarray(keypoints_xy_in_rotated_image, dtype=np.float64).copy()
    rotation_k %= 4
    for step_index in range(rotation_k, 0, -1):
        # width of the image *before* this rotation step, i.e. rot90(original, step_index - 1)
        _, previous_width = rotated_image_shape(original_height, original_width, step_index - 1)
        x_rotated, y_rotated = keypoints_xy[:, 0].copy(), keypoints_xy[:, 1].copy()
        keypoints_xy[:, 0] = previous_width - 1 - y_rotated
        keypoints_xy[:, 1] = x_rotated
    return keypoints_xy


# ---------------------------------------------------------------------------------------- small json helpers


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1))


def read_json(path: Path):
    return json.loads(path.read_text())
