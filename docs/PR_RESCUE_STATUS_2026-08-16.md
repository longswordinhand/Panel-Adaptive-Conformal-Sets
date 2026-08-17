# Pattern Recognition Rescue Status — 2026-08-16

## Locked paper question
Can conformal prediction under ambiguous expert ground truth control **case-level expert diagnostic mass** rather than only marginal/random-rater coverage, while avoiding pathological tail inefficiency (very large prediction sets)?

## Current main method
**PACS-EffCV** (Panel-Adaptive Conformal Sets with efficiency-selected training-only CV).

- For each calibration case, define the minimum top-k burden required to capture a prespecified expert plausibility mass q.
- Learn case-adaptive required-k from model-output features.
- Choose the conditional-quantile model using only an inner split of the outer training set, minimizing prediction-set size after inner conformalization.
- Apply the final split-conformal correction only on an untouched outer calibration set.
- The training-only efficiency selection does not consume the outer calibration split.
- Novelty is not claimed for split-conformal validity itself; contribution is the multi-expert target, adaptive set construction, and empirical tail-efficiency behavior.

## Dermatology benchmark (Stutz / DeepMind uncertain-ground-truth benchmark)
Dataset: 1,947 cases, 419 diagnostic classes, 1–10 expert differential diagnoses per case, 4 released model prediction matrices.
Protocol: 5 random train/calibration/test splits × 4 released models = 20 model×split runs. q=0.9, nominal case-level success target=0.9.

| Method | Case success | Mean set size | p90 set size |
|---|---:|---:|---:|
| Top-1 CP | 0.5593 | 9.94 | 16.93 |
| MCCP-10 | 0.7000 | 16.50 | 28.88 |
| Global panel | 0.9053 | 53.42 | 98.52 |
| **PACS-EffCV** | **0.9015** | **52.45** | **74.19** |

Relative to global panel at essentially matched nominal target:
- mean set size reduction: ~1.81%
- p90 set size reduction: ~24.7%
- MCCP-10 misses the case-level q=0.9 target substantially (0.7000 success).

Interpretation: the strongest practical advantage is tail inefficiency reduction, not a large reduction in average set size.

## NIH ChestX-ray14 external validation
Reader labels: fixed panel of 5 radiologists per case. 810 labeled cases; 803 corresponding 224×224 images successfully retrieved, 7 unavailable through the current Hugging Face extraction route.
Image representation: frozen ImageNet ResNet-50, locally cached weights, 2048-dimensional features; no new weights downloaded. Feature extraction used CUDA.

### Binary per-finding experiment
At q=0.9 with only 5 reader votes, the binary target is extremely coarse: a singleton prediction can capture >=0.9 expert mass only under 5/5 unanimity. PACS-TopK consequently collapses to full sets. This is retained as a finite-panel granularity limitation, not a positive result.

### 16-class diagnostic-mass experiment
For each reader, mass is distributed uniformly across that reader's selected findings; if none are positive, mass is assigned to No Finding. Averaging the 5 readers yields a per-case 16-class expert diagnostic-mass distribution. Image model: soft-label MLP head on frozen ResNet-50 features. Patient-level train/calibration/test splits.

q=0.8, 5 patient-level random splits:
| Method | Case success | Mean set size | p90 set size |
|---|---:|---:|---:|
| Top-1 CP | 0.7949 | 8.60 | 12.90 |
| MCCP-10 | 0.8659 | 9.91 | 14.20 |
| Global panel | 0.9110 | 11.35 | 15.20 |
| PACS-EffCV | 0.9344 | 12.31 | 14.00 |

q=0.9, 5 patient-level random splits:
| Method | Case success | Mean set size | p90 set size |
|---|---:|---:|---:|
| Top-1 CP | 0.6516 | 8.60 | 12.90 |
| MCCP-10 | 0.7580 | 9.91 | 14.20 |
| Global panel | 0.9096 | 12.70 | 15.80 |
| PACS-EffCV | 0.9443 | 13.70 | 15.80 |

Interpretation:
- The core failure mode of marginal methods reproduces: MCCP does not achieve the desired case-level expert-mass success.
- PACS achieves/highly exceeds the nominal case-level target.
- Unlike dermatology, NIH PACS currently pays a mean-set-size cost because it overcovers; tail improvement is modest at q=0.8 and absent at q=0.9.
- Therefore NIH is supportive external evidence for the target/coverage problem, but not yet evidence of universal efficiency dominance.

## Honest current paper-strength assessment
Positive:
1. Strong, directly relevant benchmark against the closest prior work (Stutz MCCP) on its own released data/predictions.
2. A reproducible, large case-level coverage gap between MCCP and the proposed target.
3. PACS-EffCV reaches the q=0.9 case-level target on dermatology and reduces p90 set size by ~24.7% versus a strong global panel baseline.
4. Independent radiology cohort reproduces the case-level coverage problem under patient-level splits.
5. Method tuning uses training-only inner splits; final calibration remains untouched.

Remaining risks:
1. Mean-size improvement on dermatology is modest (~1.8%).
2. NIH does not show mean-size dominance; PACS overcovers.
3. The theorem is a standard split-conformal consequence on a panel-derived target and must not be sold as fundamentally new conformal theory.
4. The 16-class NIH diagnostic-mass construction is an external robustness view and must be described transparently; it is not identical to the dermatology differential-diagnosis annotation process.
5. Seven NIH images are unavailable through the current extraction route; analyses use 803 cases with image features.

## Current decision
**Continue.** The project has crossed the previous stop gate because the main benchmark now shows a stable target-coverage advantage and a material tail-efficiency gain at matched nominal coverage. The manuscript should be positioned as a Pattern Recognition algorithm/application contribution centered on case-level expert-mass calibration and tail inefficiency, not as a new general conformal theorem.
