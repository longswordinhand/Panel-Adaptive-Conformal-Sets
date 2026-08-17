# Minimax Certification Gate

Date: 2026-08-15

## Question under test

Given a total expert-label budget `B = sum_i m_i`, can allocation across cases and repeated readers be optimized to certify the latent expert-risk tail

\[
\tau_\beta(G)=P_G(\theta>\beta),
\]

under

\[
K_i\mid \theta_i,m_i\sim \mathrm{Binomial}(m_i,\theta_i),\qquad \theta_i\sim G?
\]

The initial proposed minimax target was the worst-case uncertainty/risk over an unrestricted class of mixing distributions `G`.

---

## Gate 1: unrestricted-G formulation is degenerate

### Two-point construction

For any interior threshold \(\beta\in(0,1)\), choose

\[
G_0=\delta_{\beta-\varepsilon},\qquad
G_1=\delta_{\beta+\varepsilon}.
\]

Then

\[
\tau_\beta(G_0)=0,\qquad \tau_\beta(G_1)=1.
\]

For case `i`, under `G_j`,

\[
K_i\sim\mathrm{Binomial}(m_i,p_j),
\qquad p_0=\beta-\varepsilon,\ p_1=\beta+\varepsilon.
\]

The KL divergence between two binomials with the same number of trials is

\[
D_{KL}(\mathrm{Bin}(m_i,p_0)\Vert\mathrm{Bin}(m_i,p_1))
= m_i\, d_{Ber}(p_0\Vert p_1).
\]

For independent cases, summing over all cases gives

\[
D_{KL}(P_0\Vert P_1)
=\sum_i m_i\,d_{Ber}(p_0\Vert p_1)
=B\,d_{Ber}(p_0\Vert p_1).
\]

**The allocation \((m_1,\ldots,m_n)\) cancels exactly.**

For small \(\varepsilon\),

\[
d_{Ber}(\beta-\varepsilon\Vert\beta+\varepsilon)
=\frac{2\varepsilon^2}{\beta(1-\beta)}+O(\varepsilon^3).
\]

Set

\[
\varepsilon_B=c/\sqrt{B}.
\]

Then total KL remains bounded as \(B\to\infty\):

\[
D_{KL}(P_0\Vert P_1)
\to \frac{2c^2}{\beta(1-\beta)}.
\]

Yet the estimand remains separated by one:

\[
|\tau_\beta(G_1)-\tau_\beta(G_0)|=1.
\]

By a standard two-point Le Cam argument, the worst-case absolute-error risk is therefore bounded away from zero for every allocation rule when the parameter class permits distributions arbitrarily close to the threshold.

### Consequence

The unrestricted minimax certification problem is **NO-GO**. There is no meaningful uniformly consistent allocation rule over all mixing distributions. More cases cannot solve this local threshold pathology, and more readers per case cannot solve it either; for this least-favorable pair only the total number of Bernoulli judgments matters.

This is not claimed as a new general statistical theorem. It is a project-specific impossibility gate obtained by applying a standard two-point lower-bound argument to the finite-expert latent-tail target.

---

## Gate 2: a margin restriction makes the problem non-degenerate

A minimal structural restriction is a separation/margin condition around the clinically relevant tail threshold:

\[
G((\beta-\gamma,\beta+\gamma))=0,
\qquad \gamma>0.
\]

Thus every case has latent expert probability either at most \(\beta-\gamma\) or at least \(\beta+\gamma\).

For equal `m` and `n=B/m`, consider the simple case classifier

\[
Z_i=\mathbf 1\{K_i/m>\beta\},
\qquad
\widehat\tau=\frac1n\sum_i Z_i.
\]

By Hoeffding's inequality, under the margin condition,

\[
P(Z_i\neq \mathbf 1\{\theta_i>\beta\}\mid\theta_i)
\le e^{-2m\gamma^2}.
\]

Hence the population bias satisfies

\[
|E\widehat\tau-\tau_\beta(G)|\le e^{-2m\gamma^2}.
\]

Also, with probability at least \(1-\delta\),

\[
|\widehat\tau-E\widehat\tau|
\le
\sqrt{\frac{\log(2/\delta)}{2n}}
=
\sqrt{\frac{m\log(2/\delta)}{2B}}.
\]

Therefore a valid simple upper bound is

\[
|\widehat\tau-\tau_\beta(G)|
\le
\underbrace{e^{-2m\gamma^2}}_{\text{within-case reader resolution}}
+
\underbrace{\sqrt{\frac{m\log(2/\delta)}{2B}}}_{\text{between-case sampling}}
\]

with probability at least \(1-\delta\).

This exhibits a genuine allocation trade-off: increasing `m` exponentially suppresses threshold-crossing error but decreases the number of independent cases and therefore increases sampling error.

Balancing the two terms gives the rough scaling

\[
m^\star = O(\gamma^{-2}\log B),
\]

up to logarithmic corrections and constants. The resulting upper-bound rate is of order

\[
O\!\left(\sqrt{\frac{\log B}{B\gamma^2}}\right)
\]

for fixed confidence level.

### Numerical sanity check for the simple bound

Using \(\delta=0.05\), brute-force minimization of the displayed upper bound gives examples:

| B | gamma | bound-optimal m | bound value |
|---:|---:|---:|---:|
| 1,000 | 0.10 | 115 | 0.561 |
| 10,000 | 0.10 | 184 | 0.209 |
| 45,000 | 0.10 | 227 | 0.107 |
| 100,000 | 0.10 | 250 | 0.075 |
| 1,000,000 | 0.10 | 313 | 0.026 |
| 45,000 | 0.20 | 66 | 0.057 |

These are deliberately conservative Hoeffding-bound optima, not recommended practical reader counts.

---

## Current verdict

1. **Unrestricted minimax allocation/certification:** NO-GO. The target is uniformly non-estimable because mass can approach the threshold at `1/sqrt(B)` scale, and the least-favorable KL depends only on total budget, not its allocation.
2. **Margin-restricted allocation:** mathematically nontrivial and alive as a candidate. A simple estimator already produces an internal `n` versus `m` optimum and logarithmic repeated-reader scaling.
3. **Not yet a paper claim:** no minimax lower bound matching the margin-restricted upper rate has been proved, and novelty relative to active learning, repeated-measure optimal design, noisy-label/crowdsourcing allocation, and nonparametric functional estimation remains to be audited.

## Next theorem gate

Before any new dataset or experiment work, establish or refute both:

- a minimax lower bound under a clearly specified margin class showing that `m = Theta(gamma^{-2} log B)` (or another nontrivial allocation law) is necessary up to constants/log factors;
- a literature kill-test for this exact margin-restricted latent-tail allocation problem.

If either fails, stop the project rather than rebranding the same idea.
