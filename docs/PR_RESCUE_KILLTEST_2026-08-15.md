# Pattern Recognition Rescue — Final Novelty Kill Test (2026-08-15)

## Decision

The rescue route **cannot rely on a new conformal coverage theorem**.

Three candidate theoretical claims are already covered by prior work:

1. **Ambiguous / multi-expert ground truth in conformal prediction** — Stutz et al. (TMLR 2023) develop Monte Carlo conformal prediction using a non-degenerate expert-label distribution rather than majority-vote labels.
2. **Expected panel loss / false-negative fraction control** — Conformal Risk Control (Angelopoulos et al., 2023) controls arbitrary monotone expected losses, including false-negative-type losses.
3. **Tail / quantile control of per-case loss** — Quantile Risk Control (Snell et al., 2022/2023) directly controls quantiles / probabilities of high-loss predictions. Therefore a target such as

   P{ panel recall(C(X)) >= q } >= 1-alpha

   is a special case by defining L=1-panel_recall and controlling P{L>1-q}.

Hence the previously proposed **Case-level Panel Coverage (CPC) theorem is mathematically valid but not a standalone novelty contribution**.

## What remains potentially publishable in Pattern Recognition

The only credible rescue is an **algorithmic multi-rater pattern-recognition paper**, with conformal prediction used as a calibration layer rather than claimed as a new theory.

### Proposed core problem

Learn a predictor from multi-expert labels that preserves clinically meaningful disagreement instead of collapsing labels by majority vote, and then construct compact prediction sets calibrated against the expert panel distribution.

For case x with expert votes Y_1,...,Y_m, define empirical vote distribution

p_hat_human(y|x) = (1/m) sum_h 1{Y_h=y}.

The model should predict a distribution p_theta(y|x) aligned with p_hat_human, not only a hard consensus label.

### Required algorithmic novelty

A publishable method must contain at least one multi-rater-specific component that is not a direct application of Stutz / CRC / QRC. Candidate components to test next:

1. **Panel-aware training objective** combining diagnostic discrimination with disagreement-distribution fidelity.
2. **Panel-aware set construction** that optimizes set size subject to a calibrated panel-recall/tail-risk constraint, rather than simply applying a generic conformal quantile to a fixed score.
3. **Rater-structure correction** for non-iid or fixed panels (important for VinDr reader-triplet confounding and NIH fixed 5-reader panel).
4. **Minority-opinion preservation objective**: explicitly prevent clinically plausible minority labels from being erased by consensus training.
5. **Adaptive set efficiency under disagreement**: produce singleton sets for high-consensus cases and larger sets only where expert disagreement supports them, while maintaining a valid panel-risk guarantee.

At least one of these must survive a dedicated prior-art search before implementation.

## Existing work that constrains claims

- Stutz et al., TMLR 2023: ambiguous ground truth + Monte Carlo conformal prediction + dermatology expert disagreement.
- Angelopoulos et al., Conformal Risk Control: arbitrary monotone loss expectation control, including false-negative-type losses.
- Snell et al., Quantile Risk Control: direct control of high-loss probabilities / loss quantiles.
- Verma et al., 2022: learning to defer to multiple experts, calibration, conformal expert ensembles.
- 2026 multi-rater calibration work (e.g. TwinTrack / Multi-Rater Calibrated Segmentation Models) further shows that explicit alignment to annotator distributions is now an active area.

## Dataset assets that remain valuable

### NIH all-findings expert labels
- 810 images
- exactly 5 readers per image
- fixed complete reader panel
- strong disagreement for several findings

This is the cleanest primary benchmark for panel-aware diagnosis.

### VinDr-CXR
- 15,000 images
- exactly 3 readers per image
- strong disease-level disagreement
- reader-triplet assignment is confounded, so it should be treated as a robustness/generalization dataset, not an iid-rater population.

### QUBIQ segmentation pilot
Keep only as a negative/archival result. Do not use as the main paper.

## Go/No-Go criterion before any new model training

Do **not** train a new model until one algorithmic component above survives prior-art audit.

GO only if we can state a contribution in the form:

> Existing ambiguous-ground-truth conformal methods calibrate to an expert-label distribution but do not solve X. We introduce Y, which specifically addresses X, and demonstrate improvements in panel fidelity / minority-opinion preservation / set efficiency on NIH and VinDr.

If no such X survives literature audit, stop the rescue rather than relabeling a generic conformal method.

## Current status

- TPAMI theory route: STOP.
- CPC theorem as novelty: STOP.
- Pattern Recognition algorithmic rescue: **ALIVE but requires one multi-rater-specific algorithmic novelty before coding.**
