# Disagreement-Gated PACS Gate — 2026-08-16

## Status

**ALIVE / READY FOR REAL-BENCHMARK EXECUTION.**

This note records the next Pattern Recognition rescue step after fixed Mondrian PACS over-expanded prediction sets on strong-consensus CIFAR-10H cases.

## Failure being repaired

Fixed PACS adapts set size to predicted case difficulty, but a single adaptive mechanism can remain too conservative when most cases are nearly unanimous. On CIFAR-10H at q=0.7, the earlier fixed PACS achieved high case-level expert-mass success but enlarged sets far beyond the global baseline. That behavior is not acceptable as a main method.

## Method: Disagreement-Gated PACS

For case i with expert plausibility distribution lambda_i and required captured mass q, define the train-only gate target

    H_i(q) = 1{ max_y lambda_i(y) < q }.

Interpretation: H_i=1 exactly when no singleton label can capture q of the observed expert plausibility mass.

A classifier g(x) is trained from model-output features to predict H_i. The gate classifier is fitted on one part of the outer-training sample; a disjoint internal gate-calibration split sets a high-recall threshold using only genuinely hard cases. The untouched outer conformal calibration split is never used to select the gate.

At prediction time:

- low predicted disagreement: use a stratum-specific global panel threshold;
- high predicted disagreement: use PACS-TopK with Mondrian residual calibration;
- if either stratum is too small: fall back to the full-calibration global panel threshold.

The gate therefore has an explicit semantic target: whether a singleton can possibly satisfy the requested expert-mass level, rather than an arbitrary entropy cutoff.

## Novelty boundary

Do **not** claim novelty for:

- difficulty grouping;
- Mondrian/group-conditional conformal prediction;
- split conformal calibration;
- soft-label learning;
- generic adaptive prediction-set size.

AAAI 2026 work already groups examples by estimated difficulty and performs group-conditional conformal calibration. Stutz et al. already address conformal prediction under ambiguous ground truth and Monte Carlo calibration from expert plausibilities.

The candidate contribution is narrower:

1. the expert-mass target itself;
2. singleton-feasibility gate H_i(q) derived from the expert distribution;
3. hybrid behavior that deliberately collapses to a simple panel-calibrated predictor on strong-consensus cases while reserving adaptive enlargement for disagreement-heavy cases;
4. tail-efficiency evaluation under multi-annotator ground truth.

No direct isomorphic paper was found in the 2026-08-16 exact search for expert-plausibility-mass / fraction-of-annotators / multi-annotator conformal top-k gating. This is not proof of novelty and remains subject to reviewer-level literature audit.

## Executable validation completed in isolated container

A truth-known synthetic mixture was constructed with approximately 90% strong-consensus cases and 10% ambiguous cases. Fifty independent repetitions were run at q=0.7.

Mean across repetitions:

| method | case-level success | mean set size | mean p90 size |
|---|---:|---:|---:|
| global panel | 0.8983 | 0.9330* | 1.000 |
| fixed Mondrian PACS | 0.9736 | 1.4734 | 1.562 |
| disagreement-gated PACS | 0.9059 | 1.0094 | 1.402 |

*The simple global threshold implementation can emit empty sets in this synthetic setup, hence mean size below one; this is a baseline artifact, not a desired property.

The gate activated on ~9.9% of cases, closely matching the simulated ambiguous fraction. This verifies the intended mechanism: the adaptive branch is mostly closed on strong-consensus data while case-level success remains near the 0.9 target.

## Code added

- `src/pacs_gated.py`
- `tests/test_pacs_gated.py`
- `scripts/run_cifar10h_pacs_gated.py`
- `scripts/run_dermatology_pacs_gated.py`
- `scripts/run_nih_multiclass_pacs_gated.py`

Isolated executable unit validation: 3/3 tests passed, including the all-consensus constant-gate fallback.

## Real benchmark decision rule

The gated method survives only if it satisfies both conditions:

1. **CIFAR-10H:** materially reduces the over-expansion seen in fixed PACS, moving mean/P90 size toward the global baseline while preserving approximately 0.9 case-level q-mass success.
2. **Dermatology:** preserves the high-q tail-efficiency result (global p90 approximately 98.8 versus fixed PACS approximately 85.1 at q=0.9) without a meaningful loss in case-level success.

NIH remains a medical sensitivity analysis because its native task is multilabel and the current 16-class distributional representation is a derived stress-test endpoint.

If condition 1 or 2 fails, gated PACS is not the final Pattern Recognition method and must not be promoted in the manuscript.
