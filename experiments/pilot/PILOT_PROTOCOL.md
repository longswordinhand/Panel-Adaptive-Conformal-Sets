# QUBIQ Prostate Multi-Rater Conformal Pilot Protocol

Status: **pre-model protocol**. This document freezes the pilot design before model training or calibration results are inspected.

## 1. Scientific purpose

This pilot is a Go/No-Go test for one empirical hypothesis:

> Increasing inter-rater disagreement may create a measurable efficiency gap between prediction sets calibrated for a typical/random rater and prediction sets calibrated to cover all observed raters for a patient.

The pilot is not intended to establish methodological novelty. Its purpose is to determine whether a nontrivial phenomenon exists before investing in new theory or complex generative models.

## 2. Data and experimental unit

Dataset: QUBIQ 2021 prostate, labeled official training + validation cases only.

- 55 unique labeled cases.
- Task01: lower-disagreement control.
- Task02: higher-disagreement stress test.
- Independent unit for splitting and conformal calibration: **case/patient**.
- Multiple expert masks within a case are repeated annotations and are never treated as independent train/test split units.
- Official QUBIQ test cases are excluded from coverage evaluation because uploaded test masks are unavailable.

Task01 and Task02 use identical case-level splits.

## 3. Fixed outer design

Five outer folds are stored in:

`experiments/pilot/splits/prostate_5fold_patient_splits.csv`

For every fold:

- train = 33 cases
- calibration = 11 cases
- test = 11 cases

Across five folds every labeled case is test exactly once. Splits are deterministically generated with seed `20260815` and balanced over Task02 disagreement rank (`1 - minimum pairwise expert Dice`).

Primary reporting pools the 55 out-of-fold test predictions. Fold-level estimates are retained to expose instability.

## 4. Predictor

The first pilot uses one simple deterministic 2D segmentation predictor. No diffusion, VAE, clustering, topology module, or multi-head uncertainty model is permitted in the first pass.

Training target: per-case majority-vote consensus mask for the selected task.

The same trained probability map is reused by all conformal calibration strategies in a fold. Thus calibration strategy is the only manipulated factor.

## 5. Nested prediction-set family

Let `p_v(x)` be the predicted foreground probability at pixel `v`, and let `q in [0, 0.5]`.

Define a lower and upper mask:

- `L_q(x) = {v : p_v(x) > 0.5 + q}`
- `U_q(x) = {v : p_v(x) >= 0.5 - q}`

The structured prediction set is

`C_q(x) = { y : L_q(x) subseteq y subseteq U_q(x) }`.

As `q` increases, the family is nested. At `q=0.5`, the set is universal (`L` empty and `U` all pixels), so empty/non-empty annotation disagreements remain representable.

For a binary mask `y`, its minimal inclusion score is

`S(x,y) = max(0, max_{v:y_v=0}(p_v-0.5), max_{v:y_v=1}(0.5-p_v))`,

with absent positive/negative classes omitted from the corresponding maximum.

This is deliberately simple and is not claimed as a new conformal construction.

## 6. Calibration strategies

All strategies use the same score and prediction-set family.

### A. Consensus CP

For each calibration case, form a majority-vote expert consensus mask and calibrate on one score per case.

Purpose: quantify what is lost by collapsing rater variability before calibration.

### B. Naive annotation-level CP

Pool every `(case, expert mask)` score and apply ordinary split-conformal quantile calibration as if all annotations were independent.

Purpose: intentionally naive baseline. No finite-sample validity claim is made because repeated annotations within a case are dependent.

### C. Random-rater patient CP

For each calibration case, sample one available expert mask uniformly and calibrate on one score per case.

Because the selected rater introduces Monte Carlo variation, each trained fold will be evaluated over multiple fixed-seed rater-resampling replicates; predictions are not retrained.

### D. All-rater patient CP

For calibration case `i`, use

`S_i^max = max_h S(x_i, y_ih)`.

Calibrate on one maximum score per case.

Purpose: conservative baseline targeting simultaneous coverage of all observed expert masks for a test case.

## 7. Conformal quantiles

Primary nominal miscoverage levels:

- alpha = 0.10
- alpha = 0.20
- alpha = 0.30

With 11 patient-level calibration cases, alpha=0.10 is already near the resolution limit. Lower alpha values are not a primary target in this pilot.

Use the standard finite-sample split-conformal order statistic `ceil((n_cal+1)*(1-alpha))`. If this index exceeds the number of calibration scores, use `q=0.5`, the universal set in this bounded family.

## 8. Test metrics

For each method/task/alpha:

1. **Patient-weighted random-rater coverage**
   `mean_i mean_h 1[y_ih in C(x_i)]`.
2. **Mean expert miss rate** = 1 - patient-weighted random-rater coverage.
3. **All-rater patient coverage**
   `mean_i 1[all h: y_ih in C(x_i)]`.
4. **Consensus-mask coverage**.
5. **Ambiguity area** = `|U_q \ L_q|` pixels.
6. **Ambiguity fraction** = ambiguity area / image area.
7. **Relative ambiguity** = ambiguity area / max(consensus foreground area, 1).
8. Case-level association between required set width and audited inter-rater disagreement.

Coverage is evaluated on out-of-fold test cases only.

## 9. Primary comparison

The central paired comparison is Task01 versus Task02 under identical cases/splits and the same four calibration strategies.

Expected Go signal (not assumed):

- Task01: relatively small efficiency differences among calibration targets.
- Task02: all-rater sets become substantially wider as inter-rater disagreement increases, while random-rater/consensus calibration misses a meaningful subset of observed expert alternatives.

## 10. Sensitivity analyses

Primary analysis keeps all cases, including presence/absence disagreement.

Pre-specified sensitivity analysis excludes the two extreme Task02 cases identified before model training:

- `case07`
- `case50`

This asks whether any efficiency gap persists when empty-vs-nonempty expert disagreement is removed.

No other case may be excluded based on downstream model or conformal results without being labeled exploratory.

## 11. Go / No-Go rules

**No-Go:** Task01 and Task02 show little practical difference in coverage-efficiency behavior, or standard patient-level calibration resolves the observed difference with little efficiency cost.

**Application-only outcome:** naive annotation-level calibration differs, but patient-level random-rater/all-rater baselines adequately explain the phenomenon without a substantial efficiency gap.

**Go for method development:** a stable, reproducible gap appears between random-rater and all-rater targets, the gap increases with independently audited inter-rater disagreement, remains material after removing `case07/case50`, and cannot be explained by a small number of anomalous folds.

No TPAMI novelty claim follows automatically from a Go result; a separate literature/theory gate is required.
