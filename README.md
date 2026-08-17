# Panel-Adaptive Conformal Sets (PACS)

Research code and reproducibility materials for **Panel-Adaptive Conformal Sets for Multi-Expert Classification**.

PACS targets classification problems in which repeated expert annotations define an empirical plausibility distribution rather than a single definitive label. The main method predicts the model-ranked top-k set size required to capture a user-specified expert-mass level and applies a one-sided split-conformal residual correction.

## Repository layout

- `src/` — PACS and comparison-method implementations.
- `scripts/` — experiment, statistical-analysis, figure-generation, and integrity-audit scripts.
- `experiments/pr_rescue/results/` — frozen experiment summaries and raw run-level result tables used in the Pattern Recognition analysis.
- `experiments/pr_rescue/statistics/` — paired-effect and repeated-split statistical summaries.
- `tests/` — unit tests for PACS-related and earlier conformal components.
- `docs/` — method, figure, and analysis notes.
- `configs/` — experiment configuration files.
- `third_party/uncertain_ground_truth/` — third-party ambiguity-aware conformal utilities retained with their upstream license.

Large source datasets, downloaded public data, model checkpoints, local build products, and journal submission bundles are intentionally not versioned.

## Main PACS implementation

The primary method used in the final analysis is `PACSTopK` in:

```text
src/pacs_v2.py
```

The difficulty-stratified implementation in `src/pacs_mondrian.py` is an ablation/sensitivity variant and is not the defining PACS method.

## Key reproducibility scripts

Primary dermatology ablation and q-sensitivity:

```bash
python scripts/run_dermatology_pacs_ablation.py
python scripts/run_dermatology_pacs_global_residual_q_sensitivity.py
```

NIH and CIFAR-10H validation:

```bash
python scripts/run_pacs_global_residual_validation.py
```

Paired statistical summaries:

```bash
python scripts/analyze_pr_main_pacs_statistics.py
```

Submission-result integrity audit:

```bash
python scripts/audit_pr_submission_integrity.py
```

## Data

The repository does not redistribute the large source datasets. The experiments use public assets described in the manuscript for the ambiguous-ground-truth dermatology benchmark, CIFAR-10H, and the NIH expert-label sensitivity analysis. Downloaded datasets and derived local data are excluded by `.gitignore`.

## Scope

The code supports the paper's bounded claim: PACS is intended to reduce upper-tail prediction-set inflation in regimes where demanding expert-mass requirements create heterogeneous case-level set demands. It is not presented as a universally smaller conformal predictor or as a new generic conformal theorem.
