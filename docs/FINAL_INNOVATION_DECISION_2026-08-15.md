# Final Innovation Decision — 2026-08-15

## Decision

Freeze the project around **margin-adaptive inference for a latent expert-disagreement tail under repeated binomial judgments, with confidence-set optimality and annotation-budget design**.

Do **not** claim novelty for generic multi-rater conformal prediction, finite-rater non-identifiability, Hausdorff-moment tail bounds, generic binomial-mixture CDF inference, smooth mixing-density estimation, heterogeneous trial counts, or Bernstein approximation alone.

## Statistical target

For case i, let

\[
K_i\mid \theta_i,m_i\sim\mathrm{Binomial}(m_i,\theta_i),\qquad \theta_i\overset{iid}{\sim}G,
\]

where \(\theta_i\) is the latent probability that a randomly sampled expert gives the positive/discordant judgment for case i.

The clinical target is the tail functional

\[
\tau_\beta(G)=G((\beta,1])=P_G(\theta>\beta),
\]

not the full mixing density.

Impose a local, nonparametric margin condition near the clinically chosen threshold \(\beta\):

\[
G([\beta-t,\beta+t])\le C t^\gamma.
\]

This permits arbitrary distributional shape away from \(\beta\), including multimodality and atoms away from the threshold.

## Final claimed contribution package

1. **Tail-functional minimax theory.** Derive upper and lower minimax bounds for estimating/certifying \(\tau_\beta(G)\) in the triangular binomial-mixture experiment with heterogeneous \(m_i\), under the local margin class. The target is a discontinuous functional, unlike existing smooth-density work.

2. **Shortest honest confidence interval under a margin class.** Construct a confidence interval for \(\tau_\beta(G)\) that is honest over the margin class and prove an expected-length lower bound / asymptotic optimality result. This directly sharpens the no-shape exact-CI problem rather than re-solving it.

3. **Budget-optimal expert allocation.** Under \(\sum_i m_i\le B\), characterize the allocation that minimizes worst-case CI length (or certification risk), including whether heterogeneous/adaptive allocation can improve the minimax order. The design criterion is tail-certification uncertainty, not density loss, majority-vote accuracy, or Fisher information.

4. **Finite-sample implementation.** Use test inversion / convex moment constraints for honest coverage, with margin restrictions imposed as linear constraints on a discretized measure and with sensitivity analysis over \((C,\gamma)\).

5. **Real-data validation.** NIH all-findings (810 cases, exactly five fixed readers per image) is primary because it supports controlled reader subsampling \(m=1,\ldots,5\). VinDr (three readers/image) is external validation. No images or predictive CNNs are required for the core paper.

## Literature boundary

- Wood (1999): fixed-m binomial-mixture geometry and non-identifiability — therefore non-identifiability is background only.
- Classical Hausdorff/Chebyshev-Markov moment theory: sharp bounds from finite moments — background only.
- Basu, Brill, Yekutieli (JCGS 2026): exact pointwise CDF/quantile confidence intervals for binomial mixtures without shape restrictions — baseline and direct predecessor. Their paper explicitly raises whether smaller intervals are possible without parametric shape assumptions.
- Loh & Zhang (1996), Zhang (1995), Roueff & Rydén (2006): nonparametric mixing density/distribution estimation and slow/logarithmic rates in discrete exponential-family mixtures — rate theory background.
- Lee, Baćak, Kennedy (2025/2026 preprint): smooth binomial mixing-density estimation with heterogeneous trial counts and Bernstein-error analysis — therefore smoothness, heterogeneous m_i, and Bernstein arguments are not novelty by themselves.

## Novelty claim to defend

The paper's novelty must be stated narrowly:

> **Nonparametric, margin-adaptive, minimax-optimal inference and annotation design for a threshold tail of the latent expert-disagreement distribution from repeated binomial judgments.**

The novelty is the combination of (i) a discontinuous tail functional, (ii) local margin rather than global parametric density assumptions, (iii) honest CI-length optimality/minimax theory in the repeated-binomial experiment, and (iv) fixed-budget expert-allocation consequences.

## Kill criteria

Abandon or substantially reframe this project if any of the following is found before manuscript drafting:

1. A prior paper proves minimax rates or shortest honest CIs for a binomial-mixture CDF/tail under an equivalent local margin/Hölder-at-threshold class.
2. A prior paper solves the same \(\sum_i m_i\le B\) minimax allocation problem for the tail/CDF functional.
3. The lower bound cannot separate this functional problem from existing full-CDF/density minimax results.
4. NIH reader subsampling does not empirically exhibit the predicted uncertainty-versus-m tradeoff.

Until one of these kill criteria is met, **do not pivot to another project idea**. All subsequent theory and experiments should test this frozen formulation.
