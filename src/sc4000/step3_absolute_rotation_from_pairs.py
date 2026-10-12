"""STEP 3 - absolute rot90 per image from the pairwise relative rotations of step 1.

absolute_rotation_k[x] is the number of counter-clockwise 90 degree turns that brings image x into
the shared frame of its component. If rot90(j, r) looks like i (r = pair_relative_rotation_k[i, j]),
then absolute_rotation_k[j] = (absolute_rotation_k[i] + r) % 4.

Mode "max_spanning_tree": per connected component of the selected-pair graph (edge weight = pair
score), propagate along the maximum spanning tree from its highest-degree node, then add the global
offset c that leaves the most images unrotated (most photos are upright).
Mode "none": every image gets k = 0.

Output: step3_absolute_rotation_k.json  {image_name: k}  (inlier images only)
"""

from collections import Counter

import networkx as nx
import numpy as np

from sc4000.config import (
    ROTATION_COUNT,
    STEP3_ABSOLUTE_ROTATION_FILENAME,
    PipelineConfig,
    build_argument_parser,
    config_from_arguments,
)
from sc4000.dataset_io import read_json, write_json
from sc4000.step1_pair_similarity_dinov3 import load_step1_pair_scores
from sc4000.step2_outlier_removal_and_pair_selection import load_step2_outputs


def propagate_absolute_rotation_on_max_spanning_tree(
    image_count: int,
    selected_pair_indices: "list[tuple[int, int]]",
    pair_max_score: np.ndarray,
    pair_relative_rotation_k: np.ndarray,
) -> "dict[int, int]":
    """{image_index: absolute_rotation_k} for every image that appears in a selected pair."""
    pair_graph = nx.Graph()
    for image_a, image_b in selected_pair_indices:
        pair_graph.add_edge(image_a, image_b, weight=float(pair_max_score[image_a, image_b]))

    absolute_rotation_k: "dict[int, int]" = {}
    for component_nodes in nx.connected_components(pair_graph):
        component_tree = nx.maximum_spanning_tree(pair_graph.subgraph(component_nodes), weight="weight")
        root = max(component_tree.degree, key=lambda node_and_degree: node_and_degree[1])[0]
        component_rotation_k = {root: 0}
        for parent, child in nx.bfs_edges(component_tree, root):
            component_rotation_k[child] = (component_rotation_k[parent] + int(pair_relative_rotation_k[parent, child])) % ROTATION_COUNT

        # choose the offset that leaves the largest number of images unrotated; ties keep offset 0
        rotation_histogram = Counter(component_rotation_k.values())
        best_offset = max(range(ROTATION_COUNT), key=lambda offset: (rotation_histogram[(-offset) % ROTATION_COUNT], -offset))
        for image_index, rotation_k in component_rotation_k.items():
            absolute_rotation_k[image_index] = (rotation_k + best_offset) % ROTATION_COUNT
    return absolute_rotation_k


def run_step3_absolute_rotation(config: PipelineConfig, dataset_name: str) -> None:
    image_names, pair_max_score, pair_relative_rotation_k = load_step1_pair_scores(config, dataset_name)
    outlier_image_names, selected_pairs = load_step2_outputs(config, dataset_name)
    name_to_index = {name: index for index, name in enumerate(image_names)}
    outlier_set = set(outlier_image_names)
    inlier_image_names = [name for name in image_names if name not in outlier_set]

    absolute_rotation_k_by_name = {name: 0 for name in inlier_image_names}
    if config.rotation_propagation_mode == "max_spanning_tree":
        selected_pair_indices = [(name_to_index[a], name_to_index[b]) for a, b in selected_pairs]
        propagated = propagate_absolute_rotation_on_max_spanning_tree(
            len(image_names), selected_pair_indices, pair_max_score, pair_relative_rotation_k
        )
        for image_index, rotation_k in propagated.items():
            absolute_rotation_k_by_name[image_names[image_index]] = rotation_k
    elif config.rotation_propagation_mode != "none":
        raise ValueError(f"unknown rotation_propagation_mode {config.rotation_propagation_mode!r}")

    write_json(config.dataset_working_dir(dataset_name) / STEP3_ABSOLUTE_ROTATION_FILENAME, absolute_rotation_k_by_name)
    rotated_image_names = {name: k for name, k in absolute_rotation_k_by_name.items() if k != 0}
    print(f"[step3] {dataset_name}: {len(rotated_image_names)}/{len(absolute_rotation_k_by_name)} images rotated "
          f"({config.rotation_propagation_mode}) {rotated_image_names if len(rotated_image_names) <= 10 else ''}")


def load_step3_absolute_rotation(config: PipelineConfig, dataset_name: str) -> "dict[str, int]":
    return read_json(config.dataset_working_dir(dataset_name) / STEP3_ABSOLUTE_ROTATION_FILENAME)


def main() -> None:
    arguments = build_argument_parser(__doc__.splitlines()[1]).parse_args()
    run_step3_absolute_rotation(config_from_arguments(arguments), arguments.dataset)


if __name__ == "__main__":
    main()
