# VinDr-CXR Kaggle `train.csv` audit

Date: 2026-08-15

## Scope

Audit of `/home/yguo56/projects/MultiRaterConformalSeg/data/public/vindr_cxr/raw/train.csv` before downloading images. The CSV was reduced to image × radiologist × class presence; multiple boxes from the same radiologist for the same class were not counted as multiple votes.

## Structural integrity

- Rows: 67,914
- Images: 15,000
- Unique radiologists: 17
- Image-radiologist pairs: 45,000
- Every image has exactly 3 unique radiologists.
- 15 classes: 14 abnormalities plus `No finding`.
- Exact duplicate rows: 0
- Contradictory image-reader pairs containing both `No finding` and an abnormal class: 0
- Abnormal rows with missing bounding boxes: 0
- `No finding` rows with bounding boxes: 0
- Invalid abnormal boxes (`x_max <= x_min` or `y_max <= y_min`): 0

## Critical finding 1: normal/abnormal is unusable as a disagreement endpoint

At the image level, normal/abnormal status is unanimous for all 15,000 images:

- 10,606 images: all 3 readers recorded `No finding`
- 4,394 images: all 3 readers recorded at least one abnormal finding
- 0 images: 1-vs-2 or 2-vs-1 disagreement on normal/abnormal

Therefore the Kaggle CSV is **No-Go** for studying normal-vs-abnormal expert disagreement.

## Critical finding 2: disease-level disagreement is substantial

Among images for which at least one reader marked the finding, the fraction with non-unanimous reader labels is:

| Finding | Any-positive images | 1 vote | 2 votes | 3 votes | Non-unanimous among any-positive |
|---|---:|---:|---:|---:|---:|
| Consolidation | 353 | 232 | 90 | 31 | 0.912 |
| Atelectasis | 186 | 124 | 44 | 18 | 0.903 |
| Other lesion | 1,134 | 772 | 235 | 127 | 0.888 |
| Calcification | 452 | 275 | 120 | 57 | 0.874 |
| Lung Opacity | 1,322 | 775 | 380 | 167 | 0.874 |
| Infiltration | 613 | 368 | 154 | 91 | 0.852 |
| ILD | 386 | 234 | 81 | 71 | 0.816 |
| Pleural thickening | 1,981 | 1,099 | 517 | 365 | 0.816 |
| Nodule/Mass | 826 | 421 | 224 | 181 | 0.781 |
| Pulmonary fibrosis | 1,617 | 600 | 384 | 633 | 0.609 |
| Pleural effusion | 1,032 | 398 | 195 | 439 | 0.575 |
| Pneumothorax | 96 | 38 | 13 | 45 | 0.531 |
| Cardiomegaly | 2,300 | 483 | 520 | 1,297 | 0.436 |
| Aortic enlargement | 3,067 | 721 | 602 | 1,744 | 0.431 |

Across all 15,000 images, the largest absolute disagreement counts are:

1. Pleural thickening: 1,616 images (10.77%)
2. Aortic enlargement: 1,323 (8.82%)
3. Lung Opacity: 1,155 (7.70%)
4. Other lesion: 1,007 (6.71%)
5. Cardiomegaly: 1,003 (6.69%)
6. Pulmonary fibrosis: 984 (6.56%)
7. Nodule/Mass: 645 (4.30%)

Exactly identical 14-class multilabel sets occur for all three readers on 11,317 / 15,000 images (75.45%). The remaining 3,683 images have reader-level multilabel disagreement; 1,654 images have two distinct label sets and 2,029 have three distinct label sets.

## Critical finding 3: reader-panel assignment is highly non-uniform

The reader triplet `(R8, R9, R10)` occurs for 5,501 / 15,000 images and for 4,146 / 4,394 abnormal images. The remaining abnormal images are spread across a small number of other triplets, while `No finding` images are distributed across 411 triplet combinations.

Reader-level abnormal rates are therefore extremely heterogeneous because of assignment structure, not necessarily diagnostic behavior. For example, R8/R9/R10 each read thousands of abnormal images, whereas several other readers have zero or nearly zero abnormal images in the Kaggle representation.

Consequences:

- Do **not** treat the 17 radiologists as iid draws from one common reader population without adjustment.
- Do **not** use raw per-reader abnormal prevalence as evidence of reader bias.
- Any later model must separate case-level disagreement from panel/assignment effects.
- The safest first endpoint is class-specific 3-reader voting within each image, not cross-reader prevalence comparisons.

## Go / No-Go decision

### Go

VinDr-CXR Kaggle is suitable for a real-data experiment on **finding-level expert disagreement**, especially:

- Pleural thickening
- Lung Opacity
- Nodule/Mass
- Pulmonary fibrosis
- Pleural effusion
- Cardiomegaly / Aortic enlargement as higher-prevalence, lower-disagreement controls

It provides a strong large-n / fixed-m regime: `n = 15,000`, `m = 3`.

### No-Go

- Normal vs abnormal disagreement: zero observed disagreement.
- Reader-specific behavioral conclusions from this Kaggle file alone.
- Treating reader IDs as exchangeable without modeling the assignment design.

## Recommendation before image download

The annotation table passes the dataset-level Go gate for disease-level multi-rater theory experiments. However, before downloading all images, the next analysis should freeze 2–4 endpoints and define how the theoretical latent expert-risk model handles the fixed `m=3` panel and non-random reader assignment. Only then should image download/training begin.

Machine-readable outputs are in `experiments/pilot/results/vindr_csv_audit/`:

- `audit_summary.json`
- `class_vote_patterns.csv`
- `reader_abnormal_rates.csv`
- `readers_per_image_distribution.csv`
