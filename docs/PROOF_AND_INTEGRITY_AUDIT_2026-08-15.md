# Proof and integrity audit — 2026-08-15

## Verdict

**Conditional GO.** No exponent-changing contradiction was found in the current thinned-sequential minimax argument, but the theorem is **not yet camera-ready**. Three proof-completion items must be written explicitly before the minimax theorem can be treated as submission-grade.

## Proof audit

### P0/P1 items

1. **Weighted confidence-failure remainder must be explicit.**
   The Horvitz--Thompson contribution is bounded by `1/S_J`, which is as large as order `1/epsilon` (up to a log improvement at gamma=1). Therefore statements of the form `O(exp(-cL))` are insufficient unless the proof controls the weighted terms that enter bias and second moment. It is enough to choose `L = K log(1/(epsilon delta))` with `K` large enough so that the weighted remainders are lower order. This changes only logarithmic factors, not the epsilon exponent.

2. **Budget remainder from stochastic non-resolution must be explicit.**
   A far-from-threshold case can remain unresolved at a stage only on a confidence-band failure event. The expected stage cost contains terms of the form `S_j exp(-cL) d_j^{-2}`. Summing these terms through the deepest stage must be shown to be lower order under the same logarithmic choice of `L`.

3. **Define the minimax budget formally.**
   The manuscript must define the admissible fully adaptive policy class, uniform `(epsilon,delta)` accuracy, and
   `B*(epsilon,delta)=inf_pi sup_G E_G B_pi`
   (or the chosen equivalent definition) before stating the lower and upper bounds. The current use of `B*(epsilon)` is semantically clear but too informal for a theorem-level claim.

### Claim-boundary item

4. **The procedure is margin-aware, not adaptive to unknown gamma.**
   `h`, `A_J`, and `S_j` depend on the prespecified margin parameters `(C,gamma)`. Do not call the method `margin-adaptive` unless a separate adaptation theorem is proved. The manuscript has been patched to state this boundary explicitly.

### Lower-bound checks

5. **Parametric lower bound:** sound in structure. A two-point mixture supported away from the threshold reduces the problem to estimating a Bernoulli mixing weight, giving `Omega(epsilon^-2)`.

6. **Soft-margin shell lower bound:** sound in exponent. The shell mass is order `epsilon`, shell distance is `h ~ epsilon^(1/gamma)`, both alternatives satisfy the margin restriction, and even an oracle revealing shell membership leaves per-reader conditional KL at `O(h^2)`. Hence constant testing power requires `Omega(h^-2)=Omega(epsilon^-2/gamma)` judgments. The final proof must use alternatives whose tail values differ by strictly more than `2 epsilon` to invoke the estimation-to-testing reduction cleanly.

### Upper-bound checks

7. **HT unbiasedness mechanism:** sound. Thinning is independent of the potential judgment trajectory and a stage-`j` positive resolution is weighted by `1/S_j`, so stage contributions are recovered in expectation, up to terminal unresolved mass and confidence errors.

8. **Second-moment schedule:** the algebra is internally consistent. With `S_j=min(1,A_J d_j/d0)` and `A_J=1+sum d_j^(gamma-1)`, the sum `sum q_j/S_j` is uniformly bounded when `q_j <= C d_j^gamma` plus controlled confidence-failure remainders.

9. **Expected-cost schedule:** the dyadic sums give the claimed exponents: `epsilon^-2/gamma` for `gamma<1`, `epsilon^-2` with logarithmic overhead at `gamma=1`, and `epsilon^-2` for `gamma>1`.

## Result provenance audit

### TRACEABLE real-world claims

- NIH reader-count contraction table: source `experiments/pilot/results/tail_identification/nih_m_contraction_summary_beta05.csv`.
- NIH/VinDr 95% honest intervals and margin sensitivity: source `experiments/pilot/results/tail_honest_ci/real_data_honest_ci_sensitivity.csv`.
- NIH sample size 810 / five-reader design: source dataset audit `docs/NIH_EXPERT_LABELS_AUDIT.md`.
- VinDr sample size 15,000 / three-reader design: source dataset audit `docs/VINDR_KAGGLE_CSV_AUDIT.md`.

### TRACEABLE synthetic claims

- Population identified-width curves and truth containment: source `experiments/pilot/results/tail_identification/synthetic_population_bounds.csv`.
- 100-replicate finite-sample CI implementation check: source `experiments/pilot/results/tail_honest_ci/core_mc_100_summary.csv`.
- Thinned sequential simulation outputs: source `experiments/pilot/results/thinned_sequential_simulation.csv`.

### Important interpretation boundary

The controlled NIH `m=1,...,5` curves are descriptive identified-set calculations based on symmetric enumeration of subsets of the observed five-reader panel. They are not multinomial finite-sample confidence intervals for each subsampled `m`; formal finite-sample intervals are reported separately from the full count data.

## Manuscript consistency corrections already applied

- Removed stale statement that the soft-margin lower bound remained open.
- Updated contribution paragraph to include the thinned sequential minimax theorem.
- Replaced potentially misleading `margin-adaptive` interpretation with `margin-aware / prespecified margin class` wording.
- Preserved the distinction between descriptive projection diagnostics and the formal confidence-region model-compatibility test.

## Submission gate

Before labeling the manuscript `TPAMI-ready`, complete all of the following:

1. Write the weighted confidence-failure and expected-budget remainder bounds in the supplement with explicit constants/log choices.
2. Give a formal definition of the adaptive policy/minimax budget class.
3. Replace empirical use of the word `sharp` by `grid-discretized approximation to the sharp identified interval` unless a numerical convergence/error certificate is provided.
4. Add direct comparison to Basu--Brill--Yekutieli on synthetic/binomial-mixture settings where their interval is computationally reproducible.
5. Expand the TPAMI-facing relevance beyond a medical-label case study: explain how the framework applies to repeated human labels in pattern recognition benchmarks generally, while keeping medical radiology as the primary real-data validation.
