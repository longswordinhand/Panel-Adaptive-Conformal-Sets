# Novelty closure: thinned soft-margin minimax theorem

Date: 2026-08-15

## Frozen theorem claim

For repeated Bernoulli judgments with latent case probability theta and local margin

P(|theta-beta| <= t) <= C t^gamma,

the total expert-judgment budget required to estimate

tau_beta = P(theta > beta)

to additive error epsilon has minimax exponent

B*(epsilon) = tilde-Theta(epsilon^{-max{2,2/gamma}}),

where the upper bound is achieved by dyadic sequential confidence filtering plus random thinning of unresolved cases and Horvitz--Thompson reweighting, and the lower bound combines a parametric mixing-proportion subproblem with a sparse soft-margin shell construction. The critical margin exponent is gamma=1.

## Direct prior-work boundary

### Lee and Valiant, SODA 2021 / arXiv:1904.09228

They estimate the fraction rho of positive coins when every positive coin has bias at least 1/2+Delta and every negative coin has bias at most 1/2-Delta. They obtain tight fully-adaptive complexity Theta(rho/(epsilon^2 Delta^2) log(1/delta)). Their fully-adaptive lower bound is the essential adaptive baseline.

This is a fixed hard-gap class. It does not impose or analyze a polynomial soft-margin law allowing arbitrarily small gaps, does not derive a gamma-dependent phase transition, and does not use the unresolved-population thinning / Horvitz--Thompson construction developed here.

### Adjacent adaptive coin / bandit literature

Work on identifying the most biased coin, top-k coins, thresholding bandits, and level-set estimation targets individual arms/points or a selected set rather than the population fraction of latent Bernoulli parameters exceeding a threshold under a soft-margin distributional law.

### Tsybakov/margin active learning

Margin/noise conditions and phase transitions are classical in active learning, but the observation model and estimand differ: those works learn a decision boundary/classifier from covariates and labels. They do not directly solve the repeated-judgment population-tail problem here.

## Search outcome

Targeted searches combining:
- soft margin / Tsybakov margin;
- adaptive Bernoulli / coin sampling;
- population proportion above a threshold;
- repeated measurements / mixture of coin biases;
- Horvitz--Thompson thinning / sequential subsampling;

did not locate an equivalent theorem as of 2026-08-15. This is evidence for a defensible novelty claim, not a proof that no equivalent result exists.

## Safe novelty wording

Do not claim novelty for adaptive coin sampling, hard-gap mixture-fraction estimation, margin conditions, Horvitz--Thompson estimation, or sequential confidence intervals separately.

Defend only the narrow package:

> Minimax-optimal (up to logarithmic factors) reader-budget scaling for a latent threshold-tail functional under a polynomial soft-margin class, together with a thinned sequential estimator that achieves the rate and fully adaptive lower bounds that establish the phase transition at gamma=1.

## Remaining audit before submission

1. Independent proof audit of the thinning variance/budget calculations and adaptive KL lower bound.
2. Expand search around noisy population recovery / multilevel Monte Carlo / adaptive stratified sampling for equivalent rate results.
3. Verify all citations and DOI/year metadata.
4. Run reviewer-style novelty attack before calling the manuscript TPAMI-ready.
