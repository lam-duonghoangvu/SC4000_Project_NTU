"""STEP 5 - COLMAP reconstruction (pycolmap 0.6.1 on Kaggle).

1. Import the step-4 keypoints and matches into a fresh COLMAP database with imc25-utils
   `h5_to_db.add_keypoints` / `add_matches` (these require the files to be named keypoints.h5 and
   matches.h5, so they are symlinked into a staging dir).
2. Geometric verification of the imported pairs with `pycolmap.match_exhaustive`.
3. `pycolmap.incremental_mapping`; every model becomes one cluster.

Output: step5_registered_poses.json
  {image_name: {"cluster_index": int, "rotation_matrix": [9], "translation_vector": [3]}}
Images that are not registered are absent and later reported as outliers.
"""

import os
import shutil
import sys

import h5py
import numpy as np

from sc4000.config import (
    STEP4_KEYPOINTS_FILENAME,
    STEP4_MATCHES_FILENAME,
    STEP5_COLMAP_DIRNAME,
    STEP5_REGISTERED_POSES_FILENAME,
    PipelineConfig,
    build_argument_parser,
    config_from_arguments,
)
from sc4000.dataset_io import dataset_image_dir, read_json, write_json


def import_step4_features_into_colmap_database(config: PipelineConfig, split: str, dataset_name: str, database_path) -> int:
    """Returns the number of imported image pairs."""
    if config.imc25_utils_dir not in sys.path:
        sys.path.append(config.imc25_utils_dir)
    from database import COLMAPDatabase
    from h5_to_db import add_keypoints, add_matches

    dataset_dir = config.dataset_working_dir(dataset_name)
    staging_dir = dataset_dir / STEP5_COLMAP_DIRNAME / "h5_import_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    for staged_name, step4_filename in (("keypoints.h5", STEP4_KEYPOINTS_FILENAME), ("matches.h5", STEP4_MATCHES_FILENAME)):
        staged_path = staging_dir / staged_name
        staged_path.unlink(missing_ok=True)
        try:
            staged_path.symlink_to((dataset_dir / step4_filename).resolve())
        except OSError:
            shutil.copy(dataset_dir / step4_filename, staged_path)

    database = COLMAPDatabase.connect(str(database_path))
    database.create_tables()
    filename_to_image_id = add_keypoints(
        database, str(staging_dir), str(dataset_image_dir(config, split, dataset_name)), "",
        config.colmap_camera_model, False,
    )
    add_matches(database, str(staging_dir), filename_to_image_id)
    database.commit()
    database.close()
    with h5py.File(staging_dir / "matches.h5", "r") as matches_file:
        return sum(len(matches_file[image_a].keys()) for image_a in matches_file.keys())


def _cam_from_world(image):
    """pycolmap 0.6 exposes cam_from_world as a property, newer versions as a method."""
    return image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world


def run_step5_reconstruction(config: PipelineConfig, split: str, dataset_name: str) -> None:
    import pycolmap

    dataset_dir = config.dataset_working_dir(dataset_name)
    colmap_dir = dataset_dir / STEP5_COLMAP_DIRNAME
    shutil.rmtree(colmap_dir, ignore_errors=True)
    models_dir = colmap_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    database_path = colmap_dir / "database.db"
    poses_path = dataset_dir / STEP5_REGISTERED_POSES_FILENAME
    poses_path.unlink(missing_ok=True)  # a failed run must not leave stale poses for step 6

    imported_pair_count = import_step4_features_into_colmap_database(config, split, dataset_name, database_path)
    if imported_pair_count == 0:
        print(f"[step5] {dataset_name}: no matched pairs, skipping reconstruction")
        write_json(poses_path, {})
        return

    pycolmap.match_exhaustive(str(database_path))  # geometric verification of the imported matches
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_model_size = config.colmap_min_model_size
    mapper_options.max_num_models = config.colmap_max_num_models
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(dataset_image_dir(config, split, dataset_name)),
        output_path=str(models_dir),
        options=mapper_options,
    )

    registered_poses = {}
    for cluster_index, reconstruction in sorted(reconstructions.items()):
        for image in reconstruction.images.values():
            if image.name in registered_poses:  # keep the first (largest) model an image appears in
                continue
            cam_from_world = _cam_from_world(image)
            registered_poses[image.name] = {
                "cluster_index": int(cluster_index),
                "rotation_matrix": np.asarray(cam_from_world.rotation.matrix()).reshape(-1).tolist(),
                "translation_vector": np.asarray(cam_from_world.translation).reshape(-1).tolist(),
            }
    write_json(poses_path, registered_poses)
    image_count = len(os.listdir(dataset_image_dir(config, split, dataset_name)))
    print(f"[step5] {dataset_name}: registered {len(registered_poses)} images in {len(reconstructions)} clusters "
          f"({imported_pair_count} matched pairs, ~{image_count} files in folder)")


def load_step5_registered_poses(config: PipelineConfig, dataset_name: str) -> dict:
    poses_path = config.dataset_working_dir(dataset_name) / STEP5_REGISTERED_POSES_FILENAME
    return read_json(poses_path) if poses_path.exists() else {}


def main() -> None:
    arguments = build_argument_parser(__doc__.splitlines()[1]).parse_args()
    run_step5_reconstruction(config_from_arguments(arguments), arguments.split, arguments.dataset)


if __name__ == "__main__":
    main()
