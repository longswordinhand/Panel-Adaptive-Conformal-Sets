# QUBIQ 2021 Prostate Data Audit

Read-only audit of labeled QUBIQ 2021 prostate data (official train + validation archives).
All geometry-aware distances use the NIfTI header spacing. Every audited prostate file encodes spacing 1x1x1 and is a single-slice 2D image, so HD95 values numerically equal pixel/voxel distances; they must not be interpreted as verified acquisition-space millimetres without external metadata.

## Scope

- Unique labeled cases: **55**
- Case-task records: **110**
- Expert masks audited: **656**
- Pairwise expert comparisons: **1630**
- Official test images are excluded because the uploaded test archive contains no ground-truth masks.

## Integrity findings

- Empty masks: **7**
- Non-binary masks: **0**
- Image/mask geometry issues: **0** recorded rows
- Exact duplicate expert-mask pairs: **11**

## Task 01

- Cases: **55**; rater counts: `{'5': 1, '6': 54}`
- Image shapes: `{'640x640x1': 40, '960x640x1': 15}`
- Orientations: `{'RAS': 55}`
- Mask volume (cc), median [Q1, Q3]: **39.808 [29.207, 47.719]**
- Pairwise Dice, median [Q1, Q3]: **0.951 [0.935, 0.963]**
- Pairwise HD95 (mm), median [Q1, Q3]: **0.000 [0.000, 2.000]**
- Case-level minimum Dice, median: **0.916**
- Case-level maximum HD95 (mm), median: **5.000**

### Highest-disagreement cases

| Case | Split | n raters | Mean Dice | Min Dice | Mean HD95 mm | Max HD95 mm | Volume CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| case50 | valid | 6 | 0.872 | 0.748 | 7.80 | 21.00 | 0.207 |
| case06 | train | 6 | 0.897 | 0.794 | 6.14 | 17.00 | 0.169 |
| case54 | valid | 6 | 0.915 | 0.814 | 6.18 | 23.67 | 0.117 |
| case15 | train | 6 | 0.930 | 0.863 | 2.32 | 8.49 | 0.096 |
| case46 | train | 6 | 0.933 | 0.868 | 3.48 | 12.04 | 0.104 |
| case19 | train | 6 | 0.932 | 0.870 | 6.22 | 19.92 | 0.068 |
| case42 | train | 6 | 0.924 | 0.872 | 3.44 | 9.49 | 0.106 |
| case14 | train | 6 | 0.942 | 0.874 | 4.97 | 17.12 | 0.086 |
| case48 | train | 6 | 0.932 | 0.878 | 2.00 | 7.00 | 0.086 |
| case51 | valid | 6 | 0.939 | 0.881 | 2.58 | 9.00 | 0.090 |

## Task 02

- Cases: **55**; rater counts: `{'5': 3, '6': 52}`
- Image shapes: `{'640x640x1': 40, '960x640x1': 15}`
- Orientations: `{'RAS': 55}`
- Mask volume (cc), median [Q1, Q3]: **22.758 [14.245, 32.204]**
- Pairwise Dice, median [Q1, Q3]: **0.924 [0.890, 0.948]**
- Pairwise HD95 (mm), median [Q1, Q3]: **2.828 [1.000, 5.831]**
- Case-level minimum Dice, median: **0.870**
- Case-level maximum HD95 (mm), median: **10.000**

### Highest-disagreement cases

| Case | Split | n raters | Mean Dice | Min Dice | Mean HD95 mm | Max HD95 mm | Volume CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| case07 | train | 6 | 0.225 | 0.000 | inf | inf | 0.931 |
| case50 | valid | 6 | 0.667 | 0.000 | inf | inf | 2.449 |
| case24 | train | 6 | 0.798 | 0.611 | 16.03 | 43.01 | 0.358 |
| case04 | train | 6 | 0.869 | 0.741 | 10.65 | 29.53 | 0.194 |
| case31 | train | 5 | 0.857 | 0.743 | 9.73 | 23.02 | 0.154 |
| case53 | valid | 6 | 0.847 | 0.750 | 8.72 | 20.59 | 0.181 |
| case34 | train | 6 | 0.855 | 0.754 | 9.36 | 24.54 | 0.140 |
| case41 | train | 6 | 0.871 | 0.769 | 6.49 | 15.00 | 0.181 |
| case38 | train | 6 | 0.877 | 0.773 | 7.40 | 18.03 | 0.167 |
| case18 | train | 6 | 0.870 | 0.792 | 8.91 | 22.80 | 0.155 |

## Files

- `data/processed/audit_prostate/case_task_audit.csv`
- `data/processed/audit_prostate/mask_audit.csv`
- `data/processed/audit_prostate/pairwise_rater_metrics.csv`
- `data/processed/audit_prostate/issues.csv`
- `data/processed/audit_prostate/summary.json`
