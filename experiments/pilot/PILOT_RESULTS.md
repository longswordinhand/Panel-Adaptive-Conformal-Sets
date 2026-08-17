# QUBIQ Prostate Go/No-Go Pilot Results

## Decision

**NO-GO for the pre-specified hypothesis that higher inter-rater disagreement produces a larger all-rater conformal efficiency collapse relative to random-rater calibration.**

This decision is based on the morphology V2 prediction-set family, which passed the set-family adequacy gate. The original probability-threshold V1 family is retained as a documented negative methodological result because its calibrated bands covered roughly 98-100% of the image and therefore had uninterpretable efficiency.

## Training

- 5 fixed patient-level folds.
- 2 tasks per fold (Task01 low-disagreement control; Task02 high-disagreement stress test).
- 10 deterministic 2D U-Net models total.
- 60 epochs/model, fixed seed, fixed architecture and optimizer schedule.
- All models completed successfully and produced 22 calibration/test probability maps per fold-task.

Final training Dice varied by fold; Task02 was generally harder than Task01, consistent with the pre-model annotation audit. Training Dice is not used for the Go/No-Go decision.

## V2 adequacy

Morphological bands reduced ambiguity fractions from the V1 near-universal 0.98-1.00 range to practical values mostly around 0.10-0.31. Thus V2 efficiency comparisons are interpretable.

## Main pooled results

### alpha = 0.20

Task01:
- Random-Rater: random-rater coverage 0.760; all-rater coverage 0.570; ambiguity fraction 0.129.
- All-Rater: random-rater coverage 0.942; all-rater coverage 0.873; ambiguity fraction 0.172.
- All-Rater minus Random-Rater ambiguity cost: **+0.043** of image area.
- All-rater coverage gain: **+0.303**.

Task02:
- Random-Rater: random-rater coverage 0.858; all-rater coverage 0.777; ambiguity fraction 0.144.
- All-Rater: random-rater coverage 0.924; all-rater coverage 0.891; ambiguity fraction 0.162.
- All-Rater minus Random-Rater ambiguity cost: **+0.018** of image area.
- All-rater coverage gain: **+0.114**.

The hypothesized ordering is reversed: the extra set-size price of all-rater calibration is larger in Task01 than Task02.

### Across alpha values

All-Rater minus Random-Rater ambiguity fraction:

| alpha | Task01 | Task02 |
|---:|---:|---:|
| 0.10 | +0.119 | +0.017 |
| 0.20 | +0.043 | +0.018 |
| 0.30 | +0.026 | +0.016 |

This pattern is inconsistent with the pre-specified claim that the higher-disagreement task should show the larger efficiency collapse.

## Fold stability

Task02 all-rater minus random-rater ambiguity costs were small in every fold (roughly 0.006-0.030 at alpha=0.10; 0.008-0.030 at alpha=0.20; 0.007-0.033 at alpha=0.30). There is no hidden fold with the large Task02-specific efficiency explosion required by the Go criterion.

Task01 showed much larger between-fold variability and several folds with substantially larger all-rater costs.

## Disagreement associations

For Task02, the audited disagreement measure `1 - minimum pairwise Dice` was positively associated with the per-case radius required to cover all raters:

- Pearson r = 0.348, p = 0.009.
- Spearman rho = 0.293, p = 0.030.

After excluding the two pre-specified extreme presence/absence cases (`case07`, `case50`), this association weakened and was not conventionally significant:

- Pearson r = 0.218, p = 0.117.
- Spearman rho = 0.235, p = 0.090.

Relative ambiguity (`ambiguity area / consensus foreground area`) remained strongly associated with disagreement after excluding the two extremes, but this metric is strongly influenced by target size and is not sufficient to rescue the original efficiency-collapse hypothesis.

## Sensitivity to case07/case50

Removing case07/case50 from Task02 evaluation summaries does not create the missing Go pattern. Task02 morphology ambiguity fractions and calibration-target gaps remain modest.

## Interpretation

The pilot supports several descriptive facts:

1. Task02 has greater expert disagreement and is harder for a consensus-trained deterministic predictor.
2. Stronger all-rater coverage targets require wider prediction sets than random-rater targets.
3. Some case-level required radius is related to inter-rater disagreement.

However, the central pre-specified claim is not supported:

> Higher inter-rater disagreement did **not** cause a larger all-rater-vs-random-rater conformal efficiency collapse in Task02 than in Task01.

The result therefore does not justify developing a new TPAMI method around that hypothesis.

## Recommendation

Stop this specific methodological branch. Retain the code, splits, audits, trained models and both V1/V2 results as a reproducible negative pilot. Do not retrofit a new claim to these results.

Any next TPAMI direction should start from a new independently motivated scientific question, not from post-hoc re-labeling of this pilot.

## Result files

V1 (failed set-family adequacy):
- `experiments/pilot/results/main/`

V2 (decision basis):
- `experiments/pilot/results/morph_v2/`

Protocol:
- `experiments/pilot/PILOT_PROTOCOL.md`
- `experiments/pilot/PILOT_PROTOCOL_AMENDMENT_V2.md`
