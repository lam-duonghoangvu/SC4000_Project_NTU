"""STEP 4 - ALIKED + LightGlue local matching at a single resize (long side 1024).

Each inlier image is turned upright with np.rot90(image, absolute_rotation_k) from step 3, then
ALIKED runs on the rotated image. LightGlue matches the step-2 pairs in that rotated frame,
because its positional encoding is not rotation invariant. Keypoints are stored **mapped back to
original-image pixels**, so COLMAP runs on the untouched images and its poses are directly in the
ground-truth orientation.

Outputs (IMC baseline layout, keys are image file names):
  step4_keypoints_aliked_1024.h5            keypoints[name] -> float32 [N, 2] (x, y) in the original image
  step4_matches_aliked_lightglue_1024.h5    matches[name_a][name_b] -> int64 [M, 2]

Self test: `--selftest` checks the rot90 keypoint back-mapping on a random array, and
`--selftest-image PATH` also matches PATH against rotated copies of itself, turned upright the way
step 3 would, and checks the back-mapped keypoints against the known rotation.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from sc4000.config import (
    ROTATION_COUNT,
    STEP4_KEYPOINTS_FILENAME,
    STEP4_MATCHES_FILENAME,
    PipelineConfig,
    build_argument_parser,
    config_from_arguments,
)
from sc4000.dataset_io import dataset_image_dir, map_keypoints_from_rotated_to_original
from sc4000.step2_outlier_removal_and_pair_selection import load_step2_outputs
from sc4000.step3_absolute_rotation_from_pairs import load_step3_absolute_rotation


class AlikedLightGlueMatcher:
    def __init__(self, config: PipelineConfig):
        from lightglue import ALIKED, LightGlue

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.resize_long_side = config.local_feature_resize_long_side
        self.min_match_score = config.lightglue_min_match_score
        self.extractor = ALIKED(
            max_num_keypoints=config.aliked_max_num_keypoints,
            detection_threshold=config.aliked_detection_threshold,
        ).eval().to(self.device)
        self.matcher = LightGlue(
            features="aliked",
            depth_confidence=config.lightglue_depth_confidence,
            width_confidence=config.lightglue_width_confidence,
        ).eval().to(self.device)

    @torch.no_grad()
    def extract_features_after_rotation(self, image_path: Path, rotation_k: int) -> dict:
        """Features in the rotated frame (for LightGlue) plus keypoints mapped back to the original image."""
        from lightglue.utils import load_image

        original_image = load_image(image_path)  # float [3, H, W] in [0, 1]
        _, original_height, original_width = original_image.shape
        rotated_image = torch.rot90(original_image, rotation_k, dims=(1, 2)).contiguous()
        features = self.extractor.extract(rotated_image.to(self.device), resize=self.resize_long_side)
        keypoints_in_rotated_image = features["keypoints"][0].cpu().numpy()
        return {
            "features_in_rotated_frame": {
                "keypoints": features["keypoints"].cpu(),
                "descriptors": features["descriptors"].cpu(),
                "image_size": features["image_size"].cpu(),
            },
            "keypoints_in_original_image": map_keypoints_from_rotated_to_original(
                keypoints_in_rotated_image, original_height, original_width, rotation_k
            ).astype(np.float32),
        }

    @torch.no_grad()
    def match_feature_pair(self, features_a: dict, features_b: dict) -> np.ndarray:
        """int64 [M, 2] keypoint index pairs with LightGlue score >= min_match_score."""
        to_device = lambda features: {key: value.to(self.device) for key, value in features.items()}
        output = self.matcher({"image0": to_device(features_a), "image1": to_device(features_b)})
        matches = output["matches"][0].cpu().numpy()
        scores = output["scores"][0].cpu().numpy()
        return matches[scores >= self.min_match_score].astype(np.int64)


def run_step4_local_matching(
    config: PipelineConfig, split: str, dataset_name: str, local_matcher: "AlikedLightGlueMatcher | None" = None
) -> None:
    _, selected_pairs = load_step2_outputs(config, dataset_name)
    absolute_rotation_k_by_name = load_step3_absolute_rotation(config, dataset_name)
    output_dir = config.dataset_working_dir(dataset_name)
    keypoints_path = output_dir / STEP4_KEYPOINTS_FILENAME
    matches_path = output_dir / STEP4_MATCHES_FILENAME
    keypoints_path.unlink(missing_ok=True)
    matches_path.unlink(missing_ok=True)

    images_in_pairs = sorted({name for pair in selected_pairs for name in pair})
    if not selected_pairs:
        print(f"[step4] {dataset_name}: no selected pairs, nothing to match")
        h5py.File(keypoints_path, "w").close()
        h5py.File(matches_path, "w").close()
        return

    local_matcher = local_matcher or AlikedLightGlueMatcher(config)
    image_dir = dataset_image_dir(config, split, dataset_name)
    features_by_name = {}
    with h5py.File(keypoints_path, "w") as keypoints_file:
        for image_name in tqdm(images_in_pairs, desc="[step4] ALIKED", dynamic_ncols=True):
            features = local_matcher.extract_features_after_rotation(
                image_dir / image_name, absolute_rotation_k_by_name.get(image_name, 0)
            )
            features_by_name[image_name] = features["features_in_rotated_frame"]
            keypoints_file.create_dataset(image_name, data=features["keypoints_in_original_image"])

    kept_pair_count = 0
    with h5py.File(matches_path, "w") as matches_file:
        for image_a, image_b in tqdm(selected_pairs, desc="[step4] LightGlue", dynamic_ncols=True):
            matches = local_matcher.match_feature_pair(features_by_name[image_a], features_by_name[image_b])
            if len(matches) < config.min_matches_per_pair:
                continue
            matches_file.require_group(image_a).create_dataset(image_b, data=matches)
            kept_pair_count += 1
    print(f"[step4] {dataset_name}: {len(images_in_pairs)} images, {kept_pair_count}/{len(selected_pairs)} pairs "
          f"with >= {config.min_matches_per_pair} matches")


# ---------------------------------------------------------------------------------------- self tests


def selftest_keypoint_back_mapping() -> None:
    random_generator = np.random.default_rng(0)
    original_image = random_generator.integers(0, 1_000_000, size=(7, 11))
    for rotation_k in range(ROTATION_COUNT):
        rotated_image = np.rot90(original_image, rotation_k)
        rotated_y, rotated_x = np.meshgrid(np.arange(rotated_image.shape[0]), np.arange(rotated_image.shape[1]), indexing="ij")
        keypoints_in_rotated_image = np.stack([rotated_x.ravel(), rotated_y.ravel()], axis=1)
        keypoints_in_original_image = map_keypoints_from_rotated_to_original(
            keypoints_in_rotated_image, *original_image.shape, rotation_k
        ).astype(int)
        values_rotated = rotated_image[keypoints_in_rotated_image[:, 1], keypoints_in_rotated_image[:, 0]]
        values_original = original_image[keypoints_in_original_image[:, 1], keypoints_in_original_image[:, 0]]
        assert np.array_equal(values_rotated, values_original), f"back-mapping wrong for rotation_k={rotation_k}"
    print("[step4 selftest] rot90 keypoint back-mapping: OK for k = 0..3")


def selftest_rotated_copy_matching(config: PipelineConfig, image_path: Path, scratch_dir: Path) -> None:
    """X = image, Y = rot90(X, k) saved to disk. Step 3 would give Y absolute_rotation_k = (4 - k) % 4,
    which makes both upright, so LightGlue should match them densely. Y keypoints, back-mapped to Y's
    own pixels and then mapped into X with the known rotation, must land on the matched X keypoints."""
    from PIL import Image

    local_matcher = AlikedLightGlueMatcher(config)
    original_image = np.asarray(Image.open(image_path).convert("RGB"))
    original_height, original_width = original_image.shape[:2]
    features_x = local_matcher.extract_features_after_rotation(image_path, 0)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for rotation_k in range(1, ROTATION_COUNT):
        rotated_copy_path = scratch_dir / f"selftest_rotated_copy_k{rotation_k}.png"
        Image.fromarray(np.ascontiguousarray(np.rot90(original_image, rotation_k))).save(rotated_copy_path)
        features_y = local_matcher.extract_features_after_rotation(rotated_copy_path, (ROTATION_COUNT - rotation_k) % ROTATION_COUNT)
        matches = local_matcher.match_feature_pair(features_x["features_in_rotated_frame"], features_y["features_in_rotated_frame"])
        keypoints_y_in_x = map_keypoints_from_rotated_to_original(
            features_y["keypoints_in_original_image"][matches[:, 1]], original_height, original_width, rotation_k
        )
        error = np.linalg.norm(features_x["keypoints_in_original_image"][matches[:, 0]] - keypoints_y_in_x, axis=1)
        print(f"[step4 selftest] rotated copy k={rotation_k}: {len(matches)} matches, "
              f"median error {np.median(error):.2f}px")
        assert len(matches) > 100 and np.median(error) < 2.0, "rotated-copy matching failed"
    print("[step4 selftest] rotated-copy matching: OK")


def main() -> None:
    parser = build_argument_parser(__doc__.splitlines()[1], needs_dataset=False)
    parser.add_argument("--dataset")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--selftest-image", type=Path)
    arguments = parser.parse_args()
    config = config_from_arguments(arguments)
    if arguments.selftest or arguments.selftest_image:
        selftest_keypoint_back_mapping()
        if arguments.selftest_image:
            selftest_rotated_copy_matching(config, arguments.selftest_image, Path(config.working_dir) / "step4_selftest")
        return
    run_step4_local_matching(config, arguments.split, arguments.dataset)


if __name__ == "__main__":
    main()
