# Theory Gap Audit — finite repeated expert judgments

Date: 2026-08-15

## Current model
For case i, latent expert-positive propensity theta_i in [0,1], observed K_i | theta_i, m_i ~ Binomial(m_i, theta_i), theta_i ~ G. Primary functional tau_beta(G)=P_G(theta>beta).

## What is already known / must not be claimed as novel
1. Fixed-m binomial mixtures only identify finitely many moments of G; G itself is generally nonidentified (classical binomial-mixture / moment-problem literature; e.g. Wood 1999).
2. Sharp extrema of a CDF/tail under finitely many moments belong to the truncated Hausdorff / Chebyshev-Markov-Stieltjes moment problem.
3. Exact confidence intervals for mixing-distribution CDF/quantiles from binomial-mixture samples now exist (Basu, Brill, Yekutieli; JCGS 2025).
4. Generic n-versus-repeated-measurement allocation under fixed budget is old in reliability/longitudinal design; generic annotation-budget allocation is old in crowdsourcing.
5. Partial-identification-aware statistical decision theory is a mature and active field (Manski; Dominitz & Manski 2017; recent 2026 work).
6. A very recent Chen & Tamer 2026 paper studies partial identification from repeated/named LLM binary reports. It is not our same estimand, but it makes generic "panels of noisy binary judges imply partial identification" claims unsafe.

## Candidate gap that survived this search
### Threshold-tail certification under binomially repeated latent propensities
Instead of estimating G in general, target a clinically interpretable threshold functional
    tau_beta(G)=P(theta>beta)
and ask for the annotation design needed to certify tau_beta <= alpha with confidence 1-delta.

The key distinction from existing fixed-m work is to let the data-collection design choose n and m (or m_i) under total expert-read budget B, while the estimand remains a discontinuous latent tail functional.

### Why a structural assumption is unavoidable
Without any restriction on mass arbitrarily close to beta, uniform certification is expected to be impossible/non-informative: distributions can move small amounts of mass across beta while producing nearly indistinguishable finite-repeat observations.

A natural local regularity class is a margin condition
    G([beta-t, beta+t]) <= C t^gamma,  t in (0,t0].
This is not a parametric model for G; it only controls how much latent expert propensity piles up at the clinical threshold.

## First theorem program (candidate, not yet proved)

### T1. No-margin impossibility
Over all probability distributions G on [0,1], prove that no budget-B design/procedure can uniformly certify tau_beta to vanishing error at a nontrivial rate, because mass can be placed arbitrarily close to beta. Stronger version: derive a two-point Le Cam lower bound for any adaptive allocation policy.

### T2. Margin-class upper bound
For balanced m repeats per case, n=B/m cases, the simple estimator
    hat tau = n^{-1} sum_i 1{K_i/m > beta}
has two error sources:
- threshold misclassification bias controlled by binomial concentration + margin mass, heuristically O(m^{-gamma/2});
- between-case sampling fluctuation O(n^{-1/2}) = O(sqrt(m/B)).
Hence a first achievable envelope is
    R(B,m) <= c1 m^{-gamma/2} + c2 sqrt(m/B).
Balancing gives the candidate scaling
    m* ~ B^{1/(gamma+1)},
    n* ~ B^{gamma/(gamma+1)},
    R* ~ B^{-gamma/(2(gamma+1))}.
This is only an upper-bound heuristic until constants and the exact estimator/functional are proved.

### T3. Matching minimax lower bound
Construct pairs G0,G1 in the margin class with separated tau_beta but small KL/TV between induced Binomial-mixture experiments under budget B. If the lower bound matches T2, this would yield a genuine design theorem rather than another estimator.

### T4. Adaptive allocation
Study whether unequal/adaptive m_i can improve the minimax rate or only constants. A strong result would be either:
- balanced allocation is minimax-rate optimal under the margin class; or
- a sequential rule allocating extra readers only to cases whose current K_i/m_i lies near beta is strictly better, with a provable budget saving.
The second outcome would connect directly to practical expert-review workflows.

### T5. Finite-sample certification
Construct a one-sided certificate U_beta such that
    P_G(tau_beta <= U_beta) >= 1-delta
uniformly over the margin class, then choose/stop annotation when U_beta <= alpha.
This is a stronger practical endpoint than point estimation.

## Literature status from targeted search
- Found many fixed-budget repeated-measures designs, but objectives were ICC variance, power, Fisher information, treatment-effect estimation, or generic label accuracy—not minimax error/one-sided certification of a discontinuous tail of a nonparametric random-effects distribution under binomial repeats.
- Found general partial-identification design/decision theory, but not this exact binomial-repeated latent-tail budget problem.
- Found deconvolution/minimax CDF theory, but primarily additive measurement error; not an obvious equivalent to the binomial-repeat design problem.
- Found exact binomial-mixture CDF intervals, but not a theorem optimizing n versus m under a total repeat budget for threshold-tail certification.

This is therefore a *surviving candidate gap*, not yet a novelty claim. It must survive a deeper theorem-level literature audit before manuscript positioning.

## Empirical bridge
- VinDr-CXR: m=3, 15,000 images, strong finding-level disagreement.
- NIH all-findings expert labels: m=5, 810 images, exactly the same 5 readers per image, no missingness.
NIH can be subsampled to m=1,...,5 on the same cases/readers to test monotonic shrinking of empirical uncertainty while holding case mix fixed.

## Immediate Go/No-Go gates
1. Prove T1 rigorously.
2. Prove a nonasymptotic T2 bound under the margin condition.
3. Try to prove a matching lower bound T3.
4. Search specifically for prior work proving the same B-dependent n-vs-m minimax rate for random-effects/binomial-mixture tail functionals.
5. Only after 1-4 pass, implement NIH/VinDr experiments.

Current status: GO for theorem exploration; NOT YET safe to claim novelty.
