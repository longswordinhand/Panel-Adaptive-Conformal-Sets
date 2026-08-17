# NIH ChestX-ray14 Expert Labels Audit

## Provenance

Source mirror: `SaimMSiddiqui/ThoracicDiseaseDetection`, directory `gcs-public-data--healthcare-nih-chest-xray-labels/`. The mirror reproduces the NIH expert-label directory structure and includes the individual-reader CSVs plus reference-label CSVs and README files.

Local raw data:

- `data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv`
- `data/public/nih_expert_labels/raw/four_findings_expert_labels/four_findings_expert_labels_individual_readers.csv`

No images were downloaded.

## All-findings individual readers: primary dataset

- Shape: 4,050 rows x 19 columns.
- Unique images: 810.
- Unique patients: 532.
- Unique readers: 5.
- Every image has exactly 5 readers: 810/810.
- All 5 readers evaluated every image: each reader has exactly 810 rows.
- Unique reader panels: 1.
- Duplicate rows: 0.
- Missing cells: 0.

This is an unusually clean repeated-reader design for the finite-rater theory. It avoids the panel-assignment confounding seen in VinDr because the same five readers occur on every image.

### Disagreement by endpoint

| Endpoint | Images with reader disagreement | Rate |
|---|---:|---:|
| Atelectasis | 511 | 63.09% |
| Consolidation | 419 | 51.73% |
| Abnormal | 295 | 36.42% |
| Pleural Thickening | 291 | 35.93% |
| Effusion | 265 | 32.72% |
| Nodule | 260 | 32.10% |
| Infiltration | 248 | 30.62% |
| Cardiomegaly | 195 | 24.07% |
| Other | 195 | 24.07% |
| Edema | 183 | 22.59% |
| Mass | 169 | 20.86% |
| Pneumothorax | 154 | 19.01% |
| Emphysema | 131 | 16.17% |
| Fibrosis | 115 | 14.20% |
| Pneumonia | 65 | 8.02% |
| Hernia | 22 | 2.72% |

The `Abnormal` field is binary 0/1. The finding fields are categorical YES/NO with no missing values in this file.

### Recommendation

**GO as the primary public real-data validation dataset.** For theory experiments it provides a clean fixed design with `m=5` repeated expert judgments per image. Candidate primary endpoints are Atelectasis, Consolidation, Pleural Thickening, Nodule, and Pneumothorax, spanning high-to-moderate disagreement regimes.

## Four-findings individual readers: secondary dataset

- Shape: 13,081 rows x 8 columns.
- Unique images: 4,376.
- Unique patients: 1,695.
- Unique readers: 22.
- Readers per image: 4,327 images have 3 readers, 47 have 2, 2 have 1.
- Unique reader panels: 66.
- Duplicate rows: 0.
- Missing cells: 0.

### Disagreement by endpoint

| Endpoint | Images with reader disagreement | Rate |
|---|---:|---:|
| Airspace opacity | 1,251 | 28.59% |
| Nodule/mass | 897 | 20.50% |
| Pneumothorax | 440 | 10.05% |
| Fracture | 274 | 6.26% |

This file is mostly an `m=3` design but uses many reader panels. It therefore has more reader-assignment heterogeneity than the all-findings file. It should be treated as a secondary/sensitivity dataset rather than the cleanest population-expert experiment.

### Recommendation

**GO as a secondary public dataset**, especially for testing robustness to variable reader panels and for categorical uncertainty states if HEDGE is used in a given endpoint. Do not pool all 22 readers as iid draws without accounting for panel structure.

## Overall conclusion

The NIH expert-label release is usable and materially strengthens the project. The all-findings file gives a clean `m=5` design with the same five radiologists on all 810 images, complementing VinDr's `m=3` data. It is currently the strongest public real-data dataset for the finite-rater latent expert-risk theory because reader-count and panel-composition confounding are minimal.

Machine-readable audit outputs are in `experiments/pilot/results/nih_csv_audit/`.
