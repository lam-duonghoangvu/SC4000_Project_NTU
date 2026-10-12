# sc4000 – IMC 2025 pipeline

DINOv3 rotation-aware pair scoring → outlier removal → rotation from pairs → ALIKED + LightGlue @1024 → COLMAP.

| Step | Module (`src/sc4000/`) | Output in `<working_dir>/<dataset>/` |
|---|---|---|
| 1 | `step1_pair_similarity_dinov3.py` – each pair scored 4× (image b rotated 0/90/180/270°), keep max score + its rotation | `step1_pair_scores_dinov3.npz` |
| 2 | `step2_outlier_removal_and_pair_selection.py` – outlier = fewer than `min_neighbor_count_for_inlier` neighbours above `pair_similarity_threshold` | `step2_outlier_images.json`, `step2_selected_pairs.txt` |
| 3 | `step3_absolute_rotation_from_pairs.py` – propagate pairwise rotations on the maximum spanning tree | `step3_absolute_rotation_k.json` |
| 4 | `step4_local_matching_aliked_lightglue.py` – ALIKED on the upright image, LightGlue, keypoints mapped back to original pixels | `step4_keypoints_aliked_1024.h5`, `step4_matches_aliked_lightglue_1024.h5` |
| 5 | `step5_reconstruction_colmap.py` – imc25-utils DB import, `match_exhaustive`, `incremental_mapping` | `step5_colmap/`, `step5_registered_poses.json` |
| 6 | `step6_submission_and_scoring.py` – submission CSV, metric on train | `submission.csv` |

All knobs live in `src/sc4000/config.py` (`PipelineConfig`); every field is also a `--kebab-case` flag.

## Run

```bash
python scripts/run_pipeline.py --split test                                  # all datasets, steps 1-6
python scripts/run_pipeline.py --split train --datasets ETs stairs           # + metric
python scripts/run_pipeline.py --split train --datasets ETs --start-step 4   # reuse steps 1-3
PYTHONPATH=src python -m sc4000.step2_outlier_removal_and_pair_selection --dataset ETs --min-neighbor-count-for-inlier 5
PYTHONPATH=src python -m sc4000.step4_local_matching_aliked_lightglue --selftest-image some.png
python scripts/train.py --dinov3-model-dir <dir> --output-dir <dir>         # pair classifier head
```

`ETs` and `stairs` are never used for training (they are also in `data/test`).

## Kaggle

1. Upload `src/` and `scripts/` as the dataset `sc4000-src`.
2. Add the secret `HF_TOKEN` (access to the gated `facebook/dinov3-vitb16-pretrain-lvd1689m`) and run `kaggle/training` (internet on). Its output holds the DINOv3 snapshot, transformers wheels, ALIKED/LightGlue weights and `pair_classifier/pair_classifier_head_dinov3.pt`.
3. Run `kaggle/inference` (internet off). It mounts the training output via `kernel_sources` and writes `/kaggle/working/submission.csv`.
