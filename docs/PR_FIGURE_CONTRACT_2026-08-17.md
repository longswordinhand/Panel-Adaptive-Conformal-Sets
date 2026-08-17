# Pattern Recognition figure contract — PACS manuscript

## Core conclusion
PACS preserves the intended case-level expert-mass reliability while reducing upper-tail set inflation specifically in regimes where global panel calibration creates heterogeneous set burdens; it is not universally more efficient.

## Figure archetype
Quantitative grid for the main evidence, plus a compact schematic-led method figure.

## Target journal/output
Pattern Recognition / Elsevier. Python/matplotlib only. White background, editable SVG/PDF, 600-dpi TIFF. Main quantitative figures sized for two-column width (~178 mm); schematic sized for single/two-column flexibility.

## Fig. 1 — PACS method schematic
- a: repeated expert labels -> empirical plausibility mass lambda_x.
- b: model ranking + oracle top-k requirement K_q on proper-training cases.
- c: predict case-specific set demand from model-output features.
- d: train-only difficulty strata + untouched calibration residual correction -> top-k prediction set.
- Reviewer risk addressed: explicitly state standard conformal correction and no claim of new generic CP theorem.

## Fig. 2 — primary paired evidence
- a (hero): Dermatology q=0.9 split-level paired P90 set size, Global vs PACS, n=12 split units after averaging the four released model matrices within each split.
- b: paired success difference for the same split units with bootstrap 95% CI.
- c: paired effect summary across Dermatology q=0.7/0.8/0.9 for success, mean size, and P90 size.
- Hero evidence: P90 delta -13.75 classes, bootstrap 95% CI [-17.23, -9.96], lower in PACS for 12/12 split units; success delta -0.08 percentage points, CI spanning zero.

## Fig. 3 — operating-regime validation and boundary
- a: NIH paired success deltas across q=0.7/0.8/0.9.
- b: NIH paired P90 deltas across q=0.7/0.8/0.9.
- c: compact regime map including Dermatology q values, NIH q values, and CIFAR-10H q=0.7, plotting reliability change versus P90-size change.
- Validation evidence: NIH q=0.7 and q=0.8 show higher success with lower P90; q=0.9 saturates.
- Boundary evidence: CIFAR-10H q=0.7 increases success but inflates P90 strongly.

## Statistics
- Independent resampling unit for displayed paired intervals is the repeated split.
- Dermatology: average the four base-model matrices within each split before paired inference.
- NIH and CIFAR-10H: split is the unit.
- Error bars: percentile bootstrap 95% CI for mean paired split-level difference, 20,000 resamples, fixed seed.
- Paired t/Wilcoxon p-values remain sensitivity summaries and are not the visual centerpiece because repeated splits reuse observations.

## Source-data integrity
All panels are generated from the frozen raw result CSVs and the derived `experiments/pr_rescue/statistics/paired_split_units.csv` / `paired_effects.csv`. No observations are excluded.

## Reviewer risk
The strongest risk is overclaiming repeated-split inference. Figures therefore label split-unit n explicitly and emphasize paired effect estimates plus intervals rather than independent-sample hypothesis testing. The CIFAR-10H negative control is retained to prevent a universal-dominance interpretation.
