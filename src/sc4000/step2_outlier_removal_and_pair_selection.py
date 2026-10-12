"""STEP 2 - outlier removal and pair selection for local matching.

An image is an outlier when fewer than `min_neighbor_count_for_inlier` other images score above
`pair_similarity_threshold` with it. Outliers are reported as "outliers" and skipped by later steps.

Pairs kept for LightGlue (inlier images only):
  * every pair with pair_max_score > pair_similarity_threshold, plus
  * each inlier's `lightglue_pair_top_k` best-scoring inlier neighbours, so no inlier is left isolated.

Outputs: step2_outlier_images.json (list of names), step2_selected_pairs.txt ("name_a name_b" per line)
"""

import numpy as np

from sc4000.config import (
    STEP2_OUTLIER_IMAGES_FILENAME,
    STEP2_SELECTED_PAIRS_FILENAME,
    PipelineConfig,
    build_argument_parser,
    config_from_arguments,
)
from sc4000.dataset_io import read_json, write_json
from sc4000.step1_pair_similarity_dinov3 import load_step1_pair_scores


def find_outlier_image_mask(
    pair_max_score: np.ndarray, pair_similarity_threshold: float, min_neighbor_count_for_inlier: int
) -> np.ndarray:
    """bool [N]; True = outlier."""
    neighbor_count_above_threshold = (pair_max_score > pair_similarity_threshold).sum(axis=1)
    return neighbor_count_above_threshold < min_neighbor_count_for_inlier


def select_pairs_for_local_matching(
    pair_max_score: np.ndarray, outlier_image_mask: np.ndarray, pair_similarity_threshold: float, lightglue_pair_top_k: int
) -> "list[tuple[int, int]]":
    """Sorted list of (i, j) with i < j, both inliers."""
    inlier_indices = np.flatnonzero(~outlier_image_mask)
    inlier_scores = pair_max_score[np.ix_(inlier_indices, inlier_indices)]
    selected_pairs: "set[tuple[int, int]]" = set()

    above_i, above_j = np.nonzero(inlier_scores > pair_similarity_threshold)
    for local_i, local_j in zip(above_i, above_j):
        if local_i < local_j:
            selected_pairs.add((int(inlier_indices[local_i]), int(inlier_indices[local_j])))

    neighbor_count = min(lightglue_pair_top_k, len(inlier_indices) - 1)
    if neighbor_count > 0:
        top_neighbors = np.argsort(-inlier_scores, axis=1)[:, :neighbor_count]  # diagonal is -inf, never picked first
        for local_i, neighbors in enumerate(top_neighbors):
            for local_j in neighbors:
                if local_i == local_j:
                    continue
                image_a, image_b = int(inlier_indices[local_i]), int(inlier_indices[local_j])
                selected_pairs.add((min(image_a, image_b), max(image_a, image_b)))
    return sorted(selected_pairs)


def run_step2_outlier_removal_and_pair_selection(config: PipelineConfig, dataset_name: str) -> None:
    image_names, pair_max_score, _ = load_step1_pair_scores(config, dataset_name)
    outlier_image_mask = find_outlier_image_mask(
        pair_max_score, config.pair_similarity_threshold, config.min_neighbor_count_for_inlier
    )
    selected_pairs = select_pairs_for_local_matching(
        pair_max_score, outlier_image_mask, config.pair_similarity_threshold, config.lightglue_pair_top_k
    )
    output_dir = config.dataset_working_dir(dataset_name)
    write_json(output_dir / STEP2_OUTLIER_IMAGES_FILENAME, [image_names[i] for i in np.flatnonzero(outlier_image_mask)])
    (output_dir / STEP2_SELECTED_PAIRS_FILENAME).write_text(
        "".join(f"{image_names[i]} {image_names[j]}\n" for i, j in selected_pairs)
    )
    inlier_count = int((~outlier_image_mask).sum())
    print(f"[step2] {dataset_name}: {inlier_count}/{len(image_names)} inliers, "
          f"{len(selected_pairs)} pairs selected for LightGlue")
    if inlier_count == 0:
        print(f"[step2] WARNING: every image of {dataset_name} is an outlier "
              f"(min_neighbor_count_for_inlier={config.min_neighbor_count_for_inlier}, {len(image_names)} images)")


def load_step2_outputs(config: PipelineConfig, dataset_name: str) -> "tuple[list[str], list[tuple[str, str]]]":
    output_dir = config.dataset_working_dir(dataset_name)
    outlier_image_names = read_json(output_dir / STEP2_OUTLIER_IMAGES_FILENAME)
    selected_pairs = [
        tuple(line.split()) for line in (output_dir / STEP2_SELECTED_PAIRS_FILENAME).read_text().splitlines() if line.strip()
    ]
    return outlier_image_names, selected_pairs


def main() -> None:
    arguments = build_argument_parser(__doc__.splitlines()[1]).parse_args()
    run_step2_outlier_removal_and_pair_selection(config_from_arguments(arguments), arguments.dataset)


if __name__ == "__main__":
    main()
