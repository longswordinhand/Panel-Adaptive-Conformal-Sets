# Pattern Recognition Rescue — Final Claim Ledger (2026-08-16)

## Status

The old TPAMI-style latent-tail / binomial-mixture manuscript is not the submission manuscript. Its mathematical results may be archived or cited as motivation, but they are not the core novelty of the Pattern Recognition rescue paper.

The active paper is an algorithmic multi-expert uncertainty paper centered on **Panel-Adaptive Conformal Sets (PACS)**.

## Core target

For case x with an expert-derived plausibility distribution lambda_x over K classes and a prediction set C(x), define expert-mass capture

M_x(C) = sum_y lambda_x(y) 1{y in C(x)}.

For a prespecified expert-mass requirement q, the case is successful when

M_x(C) >= q.

The paper evaluates both the success probability P(M_X(C)>=q) and set efficiency. This is deliberately different from ordinary marginal label coverage and from average plausibility mass alone.

## Main method

PACS ranks labels by the base model's conformity/probability scores. For each training case it computes the minimum model-ranked top-k set required to capture q of the expert plausibility mass. A quantile regressor predicts this required k from test-time model-output features. Cases are partitioned into pre-defined difficulty strata using the predicted required k; an untouched calibration split supplies a separate one-sided conformal correction in each stratum. The main configuration fixes the regression quantile at 1-alpha (0.9 when alpha=0.1), avoiding test-set hyperparameter tuning. Nested efficiency selection is implemented only as a secondary sensitivity analysis.

## Claims that are supported

1. **Problem/metric claim.** Ordinary conformal coverage and MCCP-style average plausibility coverage can hide case-level failures to capture a prespecified fraction q of expert diagnostic mass.
2. **Method claim.** PACS adapts top-k set size to the estimated case-specific amount of expert mass that must be retained, then applies difficulty-stratified split-conformal correction.
3. **Dermatology high-q result.** On the public Stutz et al. dermatology benchmark (1,947 cases, 419 conditions, four released model-prediction matrices), at q=0.9 over 48 model x repeated-split runs, global panel calibration achieved case-level success 0.9048 with P90 set size 98.82; fixed PACS achieved 0.9040 with P90 set size 85.07. This is about a 13.9% reduction in P90 size with essentially unchanged success. Mean size increased from 53.54 to 54.64, so do NOT claim mean-size dominance.
4. **NIH q=0.7 result.** On the complete 810-image NIH five-radiologist panel with ResNet50 ImageNet frozen features and a soft-label Ridge head, over 30 patient-level splits, global panel calibration achieved success 0.9104, mean size 12.54, P90 size 15.20; fixed PACS achieved success 0.9323, mean size 12.57, P90 size 13.46. PACS therefore improved case-level success by 2.2 percentage points while reducing P90 size by about 11.5%, at ~0.2% mean-size cost.
5. **NIH q=0.8 result.** Global panel: success 0.9134, mean size 13.84, P90 16.00. Fixed PACS: success 0.9431, mean size 13.88, P90 15.03. Success improves by ~3.0 percentage points and P90 falls by ~6.0%, with ~0.3% mean-size cost.
6. **Saturation boundary.** At NIH q=0.9, PACS success rises to 0.9626 but both methods approach the 16-class ceiling (P90 15.93 vs 16.00); efficiency gains necessarily saturate. On dermatology q=0.7/0.8, global calibration is already efficient and PACS can enlarge sets. This boundary must be reported, not hidden.
7. **Data integrity.** The NIH image subset is now complete: 810/810 images, 810 unique hashes, no missing images. ResNet50 features were re-extracted for all 810 cases, shape 810 x 2048, finite values, fixed seed 20260815.

## Claims that are forbidden

- Do NOT claim that PACS invents split conformal, Mondrian/group-conditional conformal prediction, conditional-quantile regression, adaptive prediction sets, or ambiguity-aware conformal prediction.
- Do NOT claim that case-level expert-mass success has a new universal conformal theorem unless a new proof beyond standard derived-target/Mondrian split conformal is established.
- Do NOT claim that the latent expert distribution is identified from five readers.
- Do NOT call expert empirical plausibilities the true posterior distribution.
- Do NOT claim PACS uniformly dominates global calibration or MCCP in mean set size.
- Do NOT hide the q=0.7/0.8 dermatology cases where PACS is less efficient, or the q=0.9 NIH saturation regime.
- Do NOT reuse the old TPAMI claim that finite-rater non-identifiability or sharp moment bounds are novel.

## Closest prior art that must be cited and distinguished

- Stutz et al., TMLR 2023, *Conformal prediction under ambiguous ground truth*: Monte Carlo CP using aggregated expert plausibilities; establishes ambiguity-aware marginal coverage.
- Caprio et al., TMLR 2025, *Conformalized Credal Regions for Classification with Ambiguous Ground Truth*: conformal regions over plausible label distributions and reductions to prediction sets.
- Jang & Lee, AAAI 2026, *Quantifying and Improving Adaptivity in Conformal Prediction Through Input Transformations*: difficulty grouping plus group-conditional conformal prediction; therefore difficulty/Mondrian stratification is prior art, not our novelty.
- Hagos et al., WACV 2026 / Scientific Reports 2026 multi-expert validation work: standard conformal set size often aligns weakly with human disagreement; supports problem importance, not algorithm novelty.

## Novelty position that remains defensible

The novelty is the **specific multi-expert prediction objective and algorithmic construction**: directly learning the model-ranked set size needed to capture a prespecified fraction of expert plausibility mass at the individual-case level, then calibrating that adaptive requirement with an untouched panel-labeled calibration split. The empirical contribution is a head-to-head evaluation on the original Stutz ambiguous-ground-truth benchmark plus a complete fixed-five-radiologist NIH panel, emphasizing tail set inflation rather than average size alone.

This is an algorithmic/application novelty claim suitable for Pattern Recognition only if the final manuscript remains conservative about general conformal-theory novelty.
