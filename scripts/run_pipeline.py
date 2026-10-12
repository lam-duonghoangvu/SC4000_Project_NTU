"""Run steps 1-6 over every dataset of a split and write submission.csv.

Models are loaded once and reused across datasets. A failing step skips the rest of that dataset
(its images become outliers) instead of killing the whole Kaggle run.

    python scripts/run_pipeline.py --split test
    python scripts/run_pipeline.py --split train --datasets ETs stairs imc2023_haiper
    python scripts/run_pipeline.py --split train --datasets ETs --start-step 4   # reuse steps 1-3 outputs
"""

import gc
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from sc4000.config import build_argument_parser, config_from_arguments  # noqa: E402
from sc4000.dataset_io import dataset_image_dir, load_image_predictions_by_dataset  # noqa: E402
from sc4000.step1_pair_similarity_dinov3 import RotationAwarePairScorer, run_step1_pair_similarity  # noqa: E402
from sc4000.step2_outlier_removal_and_pair_selection import run_step2_outlier_removal_and_pair_selection  # noqa: E402
from sc4000.step3_absolute_rotation_from_pairs import run_step3_absolute_rotation  # noqa: E402
from sc4000.step4_local_matching_aliked_lightglue import AlikedLightGlueMatcher, run_step4_local_matching  # noqa: E402
from sc4000.step5_reconstruction_colmap import run_step5_reconstruction  # noqa: E402
from sc4000.step6_submission_and_scoring import run_step6_submission  # noqa: E402


def main() -> None:
    parser = build_argument_parser(__doc__.splitlines()[0], needs_dataset=False)
    parser.add_argument("--datasets", nargs="*", help="default: every dataset of the split that has an image folder")
    parser.add_argument("--start-step", type=int, default=1, choices=range(1, 7))
    arguments = parser.parse_args()
    config = config_from_arguments(arguments)
    split = arguments.split

    predictions_by_dataset = load_image_predictions_by_dataset(config, split)
    dataset_names = arguments.datasets or [
        name for name in predictions_by_dataset if dataset_image_dir(config, split, name).exists()
    ]
    print(f"[pipeline] split={split} datasets={dataset_names}")

    pair_scorer = RotationAwarePairScorer(config) if arguments.start_step <= 1 else None
    local_matcher = AlikedLightGlueMatcher(config) if arguments.start_step <= 4 else None

    for dataset_name in dataset_names:
        dataset_start_time = time.time()
        steps = [
            (1, lambda: run_step1_pair_similarity(config, split, dataset_name, pair_scorer)),
            (2, lambda: run_step2_outlier_removal_and_pair_selection(config, dataset_name)),
            (3, lambda: run_step3_absolute_rotation(config, dataset_name)),
            (4, lambda: run_step4_local_matching(config, split, dataset_name, local_matcher)),
            (5, lambda: run_step5_reconstruction(config, split, dataset_name)),
        ]
        for step_number, run_step in steps:
            if step_number < arguments.start_step:
                continue
            try:
                run_step()
            except Exception:
                print(f"[pipeline] {dataset_name}: step {step_number} failed, its images become outliers")
                traceback.print_exc()
                break
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[pipeline] {dataset_name}: {time.time() - dataset_start_time:.1f}s")

    run_step6_submission(config, split, dataset_names)


if __name__ == "__main__":
    main()
