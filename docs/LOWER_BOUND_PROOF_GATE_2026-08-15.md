# Lower-Bound Proof Gate — 2026-08-15

## Status

The previous candidate theorem

\[
R_B^\star \asymp B^{-\gamma/[2(\gamma+1)]}
\]

based on a hoped-for KL bound of order \(q^2mh^2\) is **not yet justified**. A direct two-point calculation shows why that step is delicate: under a margin class, the observation-law denominator near \(\beta\) cannot contain order-one mass without violating the margin restriction, so the naive \(q^2\) scaling is generally unavailable.

This does **not** kill the project. It reveals that two different limits must be separated:

1. **Identification limit**: what can be recovered if the binomial-mixture law is known exactly (equivalently, infinitely many cases for fixed \(m\))?
2. **Stable statistical limit**: what can be recovered from finitely many cases when the inversion from the \((m+1)\)-cell count distribution to a discontinuous tail functional is noisy/ill-conditioned?

That distinction now appears to be the central theoretical object.

## Model

\[
K_i\mid \theta_i \sim \mathrm{Binomial}(m,\theta_i),\qquad \theta_i\sim G,
\]

with target

\[
\tau_\beta(G)=G((\beta,1]).
\]

Assume a local margin class around the clinical threshold \(\beta\):

\[
G([\beta-t,\beta+t])\le C t^\gamma,
\qquad 0<t\le t_0.
\]

## Gate 1: exact-law identification width

A numerically stable LP was formulated directly in the Bernstein/binomial basis. Two distributions \(G_0,G_1\) were constrained to:

- be probability distributions on \([0,1]\),
- satisfy the same binomial-mixture probabilities
  \[
  \int {m\choose k}\theta^k(1-\theta)^{m-k}\,dG_0
  =
  \int {m\choose k}\theta^k(1-\theta)^{m-k}\,dG_1,
  \quad k=0,\ldots,m,
  \]
- satisfy the margin constraints,
- maximize \(\tau_\beta(G_1)-\tau_\beta(G_0)\).

For \(\beta=1/2,\gamma=1,C=2\), the worst-case tail gap was approximately:

| m | max tail gap | m × gap |
|---:|---:|---:|
| 5 | 0.2182 | 1.091 |
| 8 | 0.1727 | 1.382 |
| 10 | 0.1430 | 1.430 |
| 12 | 0.1220 | 1.464 |
| 16 | 0.0943 | 1.509 |
| 20 | 0.0770 | 1.540 |
| 24 | 0.0650 | 1.560 |
| 30 | 0.0526 | 1.579 |

The stable range is strongly consistent with

\[
\boxed{\text{identification width}\ \asymp m^{-\gamma}}
\]

for \(\gamma=1\), rather than the earlier \(m^{-\gamma/2}\) guess.

The larger-m LP begins to encounter grid-resolution limits, so this is evidence, not yet a theorem.

## Gate 2: why the simple plug-in estimator is slower

The natural per-case estimator

\[
\widehat\tau_{\rm plug}=
\frac1n\sum_i \mathbf 1\{K_i/m>\beta\}
\]

has crossing error controlled by Hoeffding:

\[
P\!\left(\mathbf 1\{K_i/m>\beta\}\neq \mathbf 1\{\theta_i>\beta\}\mid \theta_i\right)
\le e^{-2m(\theta_i-\beta)^2}.
\]

Under the margin condition this yields bias of order

\[
O(m^{-\gamma/2}),
\]

and sampling error \(O(n^{-1/2})\).

Hence the simple stable estimator obeys

\[
R_{n,m}\lesssim m^{-\gamma/2}+n^{-1/2}.
\]

Under \(B=nm\), balancing gives

\[
m\asymp B^{1/(\gamma+1)},\qquad
R_B\lesssim B^{-\gamma/[2(\gamma+1)]}.
\]

For \(\gamma=1\), this is \(B^{-1/4}\).

But the exact-law LP suggests that, with noiseless knowledge of all degree-\(m\) Bernstein moments, the intrinsic identification width can be as small as order \(m^{-\gamma}\). Therefore the gap between \(m^{-\gamma}\) and \(m^{-\gamma/2}\) is a **stability/inversion gap**, not an identification gap.

## Gate 3: likely lower-bound mechanisms

### A. Identification lower bound

If one can construct two margin-class distributions with identical first \(m\) binomial/Bernstein moments and tail separation

\[
|\tau_\beta(G_1)-\tau_\beta(G_0)|\gtrsim m^{-\gamma},
\]

then no amount of additional cases can beat \(m^{-\gamma}\) for fixed \(m\).

This should be attacked with truncated-moment extremal theory / polynomial duality rather than the earlier local KL argument.

### B. Finite-sample stability lower bound

For finite \(n\), recovering a discontinuous tail from noisy estimates of the \((m+1)\)-cell mixture pmf is an inverse problem. The relevant question is not merely whether the target is identified, but whether a degree-\(m\) polynomial approximant to the step function can be represented with Bernstein coefficients small enough to keep estimator variance controlled.

A bounded-coefficient Bernstein statistic has the form

\[
\widehat T_a=\frac1n\sum_{i=1}^n a_{K_i},
\qquad |a_k|\le A,
\]

with expectation

\[
E_G\widehat T_a
=
\int \sum_{k=0}^m a_k {m\choose k}\theta^k(1-\theta)^{m-k}\,dG(\theta).
\]

Thus the statistical problem is equivalent to a **stable polynomial approximation problem**:

> How sharply can the step function \(1\{\theta>\beta\}\) be approximated by degree-\(m\) polynomials whose Bernstein coefficients are variance-controlled?

This is the more credible source of the \(m^{-\gamma/2}\) scale.

## Literature audit relevant to this distinction

1. Basu, Brill & Yekutieli, *Exact Confidence Intervals for the Mixing Distribution from Binomial Mixture Distribution Samples*, JCGS 2026: exact pointwise CDF/quantile confidence intervals without shape assumptions; this confirms that fixed-\(m\) CDF inference itself is not a new problem.
2. Classical Hausdorff/truncated moment theory addresses reconstruction and extremal CDF bounds from finitely many moments; therefore exact-law identification bounds alone are not enough for novelty.
3. Recent approximation-theory work on Bernstein-form polynomials shows that coefficient constraints materially change achievable approximation rates. In particular, bounded/integer Bernstein coefficients can exhibit the characteristic \(n^{-s/2}\) approximation scale rather than unrestricted-polynomial rates.
4. This supports treating the finite-sample problem as a stability-constrained inverse problem, not merely a moment-identification problem.

## Current theorem target

The strongest defensible target is now a two-regime theorem of the form

\[
R_{n,m}^\star(\mathcal G_{\gamma,C})
\asymp
\underbrace{\mathfrak I_m}_{\text{identification floor}}
\vee
\underbrace{\mathfrak S_{n,m}}_{\text{statistical stability floor}},
\]

where

\[
\mathfrak I_m\asymp m^{-\gamma}
\]

and \(\mathfrak S_{n,m}\) is determined by stable Bernstein approximation / inverse-noise amplification.

The previous \(B^{-1/4}\) rate for \(\gamma=1\) remains a valid **achievable upper bound for the simple plug-in estimator**, but it is **not yet established as minimax**.

## Decision

**PROOF-GATE REMAINS OPEN.**

The project should continue only if we can prove one of the following genuinely new statements:

1. a matching finite-sample minimax lower bound for stable latent-tail estimation under \(B=nm\), or
2. a two-regime phase diagram showing when the exact-moment identification floor and the noisy-inversion floor dominate, including an optimal allocation law for \((n,m)\).

If neither can be proved, the theory core should be abandoned rather than weakened into a method-combination paper.
