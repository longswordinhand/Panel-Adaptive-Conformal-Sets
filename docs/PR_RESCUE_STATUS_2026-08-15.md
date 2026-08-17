# Pattern Recognition Rescue Status — 2026-08-15

## Locked research target
Case-level expert plausibility-mass coverage under ambiguous ground truth. The practical target at the main operating point is q=0.9 expert plausibility mass for at least 90% of cases. Ordinary MCCP is retained as a direct predecessor/baseline because marginal expert coverage does not imply this case-level target.

## Main candidate algorithms
- PACS-CV: adaptive top-k prediction sets. The conditional quantile hyperparameter is selected using only the outer training split; the outer calibration split remains untouched until final conformal calibration.
- Safe-PACS: training-only method selection between PACS and a global panel-quantile fallback.
- PTCP / PanelCert: rigorous finite-panel safety references; retained as theory/safety baselines because they are too conservative for the practical default operating point.
- Tail-PACS: exploratory CVaR-oriented ablation; not promoted to main method after quick screening showed overly aggressive inner selection.

## Dermatology benchmark
Source: Google DeepMind uncertain_ground_truth benchmark.
- 1,947 cases
- 419 diagnostic classes
- 1–10 expert rankings/case (median 5)
- four official model-prediction arrays

### q=0.9, 5 random splits × 4 official models = 20 model/split runs
| Method | Case success | Success SD | Mean set size | Mean-size SD | p90 set size | Mean expert mass | Minority capture |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global panel | 0.9053 | 0.0183 | 53.4156 | 7.0037 | 98.52 | 0.9702 | 0.9522 |
| MCCP-10 | 0.7000 | 0.0334 | 16.4972 | 2.0281 | 28.88 | 0.8965 | 0.8419 |
| PACS-CV | 0.9035 | 0.0195 | 52.9721 | 5.9836 | 71.47 | 0.9695 | 0.9525 |
| Safe-PACS | 0.9035 | 0.0183 | 52.6925 | 7.2981 | 78.13 | 0.9698 | 0.9523 |
| Top-1 CP | 0.5593 | 0.0446 | 9.9351 | 1.6800 | 16.93 | 0.8360 | 0.7511 |

### Current interpretation
- The research problem is empirically real: MCCP-10 reaches only 70.0% case-level q=0.9 mass success while the desired target is 90%.
- PACS-CV reaches 90.35%, essentially matching the global panel baseline at 90.53%.
- PACS-CV mean set-size gain over global is modest (~0.83%), but p90 set size drops from 98.52 to 71.47 (~27.5%).
- Safe-PACS improves mean set size by ~1.35% vs global and reduces p90 by ~20.7%; it selected the global fallback in 4/20 model/split runs.
- The strongest current algorithmic signal is therefore tail-efficiency / avoidance of extremely large prediction sets at matched case-level panel coverage, not a dramatic improvement in average set size.

## NIH status
- reader-level 5-radiologist expert labels are ready for 810 cases.
- 800/810 corresponding 224×224 images have been obtained through the mapped Hugging Face subset route.
- no model training result is claimed yet.

## Integrity / implementation status
- 17 relevant unit tests pass.
- PACS/PACS-v2/PACS-v3/PACS-CV/Safe-PACS/Tail-PACS modules compile successfully.
- no relevant background experiment/download process was left running at the end of this status freeze.

## Hard decision rule
Do not claim a new conformal theorem. The publishable contribution must be algorithmic and empirical:
1. case-level expert-mass target distinct from marginal MCCP coverage;
2. training-only adaptive set construction without test tuning;
3. stable reduction of tail inefficiency at matched target coverage;
4. independent confirmation on NIH or another genuine repeated-expert dataset.

If full repeated experiments and the independent dataset fail to preserve the tail-efficiency advantage, stop the Pattern Recognition rescue rather than inventing another formulation.
