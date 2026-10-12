"""Single source of truth for paths and tunable knobs of the IMC2025 pipeline.

Every field of `PipelineConfig` is exposed as a `--field-name` command line flag
by `build_argument_parser`, so each step script can override any knob.
Kaggle runs Python 3.10, so keep the syntax 3.10-compatible.
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

ROTATION_COUNT = 4  # rotations by 0, 90, 180, 270 degrees counter-clockwise (np.rot90 / torch.rot90 convention)


@dataclass
class PipelineConfig:
    # ---------------------------------------------------------------- paths (Kaggle defaults)
    competition_data_dir: str = "/kaggle/input/image-matching-challenge-2025"
    working_dir: str = "/kaggle/working/result"
    submission_csv_path: str = "/kaggle/working/submission.csv"
    imc25_utils_dir: str = "/kaggle/input/imc25-utils"
    # outputs of the kaggle/training notebook, mounted in the inference notebook via kernel_sources
    dinov3_model_dir: str = "/kaggle/input/sc4000-training/dinov3-vitb16-pretrain-lvd1689m"
    pair_classifier_checkpoint_path: str = "/kaggle/input/sc4000-training/pair_classifier/pair_classifier_head_dinov3.pt"

    # ---------------------------------------------------------------- step 1: rotation-aware pair similarity (DINOv3)
    global_descriptor_resize_long_side: int = 448
    use_cosine_similarity_without_classifier: bool = False

    # ---------------------------------------------------------------- step 2: outlier removal and pair selection
    pair_similarity_threshold: float = 0.5
    min_neighbor_count_for_inlier: int = 10
    lightglue_pair_top_k: int = 10

    # ---------------------------------------------------------------- step 3: absolute rotation from pairwise rotations
    rotation_propagation_mode: str = "max_spanning_tree"  # "none" | "max_spanning_tree"

    # ---------------------------------------------------------------- step 4: ALIKED + LightGlue local matching
    local_feature_resize_long_side: int = 1024
    aliked_max_num_keypoints: int = 4096
    aliked_detection_threshold: float = 0.01
    lightglue_depth_confidence: float = -1.0
    lightglue_width_confidence: float = 0.99
    lightglue_min_match_score: float = 0.2
    min_matches_per_pair: int = 15

    # ---------------------------------------------------------------- step 5: COLMAP reconstruction
    colmap_camera_model: str = "simple-radial"  # simple-pinhole | pinhole | simple-radial (imc25-utils h5_to_db)
    colmap_min_model_size: int = 3
    colmap_max_num_models: int = 5

    # ---------------------------------------------------------------- training
    datasets_excluded_from_training: list = field(default_factory=lambda: ["ETs", "stairs"])

    # ---------------------------------------------------------------- runtime
    device: str = "cuda"

    def dataset_working_dir(self, dataset_name: str) -> Path:
        path = Path(self.working_dir) / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        return path


# Intermediate file names, kept in one place so every step agrees on them.
STEP1_PAIR_SCORES_FILENAME = "step1_pair_scores_dinov3.npz"
STEP2_OUTLIER_IMAGES_FILENAME = "step2_outlier_images.json"
STEP2_SELECTED_PAIRS_FILENAME = "step2_selected_pairs.txt"
STEP3_ABSOLUTE_ROTATION_FILENAME = "step3_absolute_rotation_k.json"
STEP4_KEYPOINTS_FILENAME = "step4_keypoints_aliked_1024.h5"
STEP4_MATCHES_FILENAME = "step4_matches_aliked_lightglue_1024.h5"
STEP5_COLMAP_DIRNAME = "step5_colmap"
STEP5_REGISTERED_POSES_FILENAME = "step5_registered_poses.json"


def _parse_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


def build_argument_parser(description: str, needs_dataset: bool = True) -> argparse.ArgumentParser:
    """Argument parser with --split, --dataset and one flag per PipelineConfig field."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    if needs_dataset:
        parser.add_argument("--dataset", required=True, help="dataset name, e.g. ETs")
    default_config = PipelineConfig()
    for config_field in dataclasses.fields(PipelineConfig):
        flag = "--" + config_field.name.replace("_", "-")
        default_value = getattr(default_config, config_field.name)
        if isinstance(default_value, bool):
            parser.add_argument(flag, type=_parse_bool, default=default_value)
        elif isinstance(default_value, list):
            parser.add_argument(flag, nargs="*", default=default_value)
        else:
            parser.add_argument(flag, type=type(default_value), default=default_value)
    return parser


def config_from_arguments(arguments: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(**{
        config_field.name: getattr(arguments, config_field.name)
        for config_field in dataclasses.fields(PipelineConfig)
    })
