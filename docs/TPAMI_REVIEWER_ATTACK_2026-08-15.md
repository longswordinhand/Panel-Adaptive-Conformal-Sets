# TPAMI reviewer attack — 2026-08-15

## Reviewer 1 — theory / statistical learning emphasis

### Strengths
- Clear target functional: population tail of latent expert-positive probability.
- Finite-sample honest CI is simple, auditable, and does not rely on asymptotics.
- Thinned sequential construction has a nontrivial fully-adaptive minimax exponent with a gamma=1 phase transition.
- Hard-gap prior work is explicitly separated rather than misclaimed.

### Major concerns

**R1-M1 — proof completion is not yet publication-grade.**
The upper-bound supplement currently suppresses weighted confidence-failure terms even though HT weights can be order 1/epsilon.
Close criterion: explicit bias, second-moment, and expected-budget remainder inequalities under a stated L=K log(1/(epsilon delta)) choice.

**R1-M2 — minimax object is underspecified.**
B*(epsilon) appears before a formal policy class and accuracy criterion.
Close criterion: define admissible adaptive policies, stopping rules, expected budget, and uniform (epsilon,delta)-accuracy before Theorems 2--3.

**R1-M3 — margin parameters are assumed known.**
The allocation schedule uses C and gamma.
Close criterion: call the procedure margin-aware, not adaptive to unknown gamma; optionally add a short discussion of adaptation as future work.

**R1-M4 — direct closest-CI baseline is missing.**
Basu--Brill--Yekutieli provide exact CDF/quantile intervals for the same binomial-mixture observation family.
Close criterion: reproduce their public implementation or provide a technically justified apples-to-apples comparison on shared synthetic settings.

## Reviewer 2 — empirical / annotation-quality emphasis

### Strengths
- Uses real individual-reader labels rather than consensus labels.
- NIH fixed five-reader panel supports a clean controlled reader-depth calculation.
- VinDr provides a larger external dataset.
- Model-compatibility test is separated from descriptive projection error.

### Major concerns

**R2-M1 — reader exchangeability is empirically questionable.**
NIH reader-specific positive rates differ sharply for five of six highlighted endpoints after a Bonferroni diagnostic; e.g. Consolidation 4.7--47.5%, Pleural Thickening 3.0--27.2%.
Close criterion: present NIH as a panel-averaged controlled-depth study, report reader-specific heterogeneity, and do not claim that count-law compatibility verifies reader exchangeability.

**R2-M2 — subset averaging can hide reader identity effects.**
At small m, subset-specific identified widths have wide ranges.
Close criterion: include subset-specific distributions/ranges in supplement and explain why the averaged curve answers a different question from reader-specific robustness.

**R2-M3 — sequential theory is only synthetic.**
The adaptive algorithm cannot be retrospectively executed on a dataset capped at 3 or 5 readers/case.
Close criterion: state this boundary plainly; use real data for finite-panel inference and synthetic data for sequential-budget validation.

**R2-M4 — practical constants are large.**
Current Hoeffding-based simulation uses very large reader budgets.
Close criterion: report this rather than hiding it; distinguish minimax exponent contribution from practical constant optimization.

## Reviewer 3 — TPAMI fit / impact emphasis

### Strengths
- Human-label uncertainty is relevant to benchmark construction, evaluation, and pattern-recognition datasets.
- The paper offers a principled alternative to majority-vote collapse.
- The phase transition gives a memorable theoretical result.

### Major concerns

**R3-M1 — current framing is too radiology-specific for TPAMI.**
The theory is generic but the empirical narrative reads primarily as a medical-label paper.
Close criterion: reframe the introduction around repeated human judgments in pattern recognition, with radiology as a high-stakes validation domain; add at least one non-radiology or general annotation discussion if possible without diluting the paper.

**R3-M2 — current manuscript is underdeveloped for a Transactions paper.**
Five pages of main text is not enough to establish the theory, algorithms, diagnostics, baselines, and experimental detail.
Close criterion: expand to a self-contained 9--12 page main paper with theorem statements, algorithm pseudocode, proof intuition, complete experimental protocol, robustness, and limitations; place long derivations in supplement.

**R3-M3 — numerical 'sharpness' must be qualified.**
Empirical LPs use finite support grids.
Close criterion: call reported values grid-discretized approximations and provide grid convergence checks or certified measure-LP bounds.

**R3-M4 — novelty must be triangulated against both exact-CI and hard-gap adaptive work.**
Close criterion: a dedicated related-work paragraph/tabular comparison distinguishing target, assumptions, guarantee, adaptivity, and budget objective.

## Cross-review synthesis

### Consensus strengths
- Target functional is interpretable and different from consensus accuracy.
- Finite-sample confidence construction is valid and transparent.
- Thinned sequential allocation is the main theory contribution.
- NIH/VinDr provide real reader-level evidence that finite panel depth matters.

### Consensus blockers before TPAMI submission
1. Finish weighted-error proof details and formal minimax definitions.
2. Add Basu et al. direct comparator.
3. Make reader heterogeneity a first-class limitation/diagnostic.
4. Expand the paper substantially and improve TPAMI/general pattern-recognition framing.
5. Preserve grid-discretization and margin-assumption boundaries everywhere.

## Current recommendation posture
**Major revision before submission, not reject / abandon.** The project is now coherent enough to continue, but the current five-page build is not TPAMI-ready.
