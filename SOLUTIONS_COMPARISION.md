# Solutions comparison – IMC 2025

Tags: **[R4]** = idea taken from the rank-4 solution, **[CROP]** = idea taken from the crop-ensemble solution, **[NEW]** = not in either sample.

## Current method (`sc4000`)

| Step | What it does | Origin |
|---|---|---|
| 1. Pair scoring | Frozen DINOv3 ViT-B/16 (CLS + mean patch) and a small trained MLP head that answers "same scene?" | Pair classifier: **[R4]** (R4 uses DINOv2 + LoFTR-style transformer). DINOv3 and the MLP head: **[NEW]** |
| 1. Rotation-aware scoring | Each pair is scored with image b at 0/90/180/270°; the best score and its rotation are kept. The head is trained with wrongly rotated same-scene pairs as negatives | **[NEW]** |
| 2. Outlier removal | An image is an outlier if fewer than 2 other images score > 0.5 with it | **[NEW]** (neither sample removes outliers explicitly) |
| 2. Pair selection | Pairs scoring > 0.5, plus each image's top-10 neighbours | Threshold: **[R4]**. Top-k safety net: like R4's NetVLAD top-k **[R4]** |
| 3. Absolute rotation | Pairwise rotations are spread along a maximum spanning tree | **[NEW]** (R4 classifies each image's orientation on its own) |
| 4. Local matching | ALIKED + LightGlue on the upright image, long side 1024, 4096 keypoints, score ≥ 0.2, ≥ 15 matches; keypoints mapped back to the original image | ALIKED + LightGlue: **[CROP]**. Single scale: simpler than both samples |
| 5. Reconstruction | imc25-utils database import, COLMAP geometric verification, incremental mapping (min model size 3, max 5 models); clusters = models | **[R4]** / **[CROP]** (shared baseline) |

Not adopted yet: multi-scale matching **[R4]**, several feature families, crop re-matching, a high RANSAC match bar **[CROP]**.

Last test run: ETs registered 19/22 images (2 clusters). stairs registered 16/51 images (3 clusters): only 159 of 671 selected pairs reached 15 matches, so local matching is the bottleneck.

## Rank 4 (`sample-solutions/imc2025-rank-4`)

- **Orientation:** `check_orientation` ResNeXt50 classifies each image's rotation, which is corrected if confidence > 0.9.
- **Pairs:** all pairs pass through a DINOv2 + transformer pair classifier (keep if ≥ 0.6), joined with NetVLAD top-k (max(20, 12% of N)) and deduplicated.
- **Features:** RDD + LightGlue at 2 scales (1024 and 1280, 8192 keypoints); the keypoints and matches of both scales are concatenated.
- **SfM:** COLMAP geometric verification, then incremental mapping (min model size 3, max 5 models).
- **Key idea:** high pair recall (learned classifier plus retrieval) and one strong detector run at two scales.

## Crop ensemble (pasted pipeline)

- **Pairs:** SuperPoint (GIM weights) on all pairs, keep pairs with ≥ 30 matches. Matching is used as the shortlist instead of global descriptors.
- **Features (1280):** SuperPoint, ALIKED + LightGlue, and SIFT + nearest neighbour on the kept pairs.
- **Crops:** crop the shared object using the merged matches, then re-match the crops with ALIKED + LightGlue and SuperPoint + LightGlue.
- **Filter:** concatenate all 5 match sets, run RANSAC, keep pairs with ≥ 150 matches.
- **SfM:** default COLMAP; clusters = the models it produces.
- **Key idea:** dense and diverse matches (3 detector families plus crop zoom), and a strict match bar that also separates the scenes. Costly, because it matches every pair, and it has no rotation handling.
