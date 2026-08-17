# Thinned sequential inference under a soft margin

Status: **theorem-level upper-bound derivation; paired with the proved lower bounds in `SOFT_MARGIN_MINIMAX_LOWER_BOUND_PARTIAL.md`, this closes the minimax exponent for all gamma > 0 up to logarithmic factors.**

## 1. Setting

Cases have iid latent probabilities `theta_i ~ G`; judgments from a case are conditionally iid Bernoulli(`theta_i`). The target is

\[
\tau_\beta(G)=P_G(\theta>\beta).
\]

Assume, for fixed `beta` in the interior of `(0,1)`,

\[
P_G(|\theta-\beta|\le t)\le C t^\gamma,\qquad 0<t\le t_{\max}.
\]

We measure total expert judgments. Introducing a new case is free until its first requested judgment, which costs one judgment exactly as any other observation.

## 2. Why the previous sequential algorithm is suboptimal

The earlier procedure continued **every** unresolved case to the next depth. With `N ~ epsilon^{-2}` base cases and terminal resolution `h ~ epsilon^{1/gamma}`, this wastes effort on the entire near-threshold subset. The key observation is that the contribution of a narrow unresolved band is itself small, so only a random subsample of that band must be investigated more deeply. Inverse-probability weighting recovers its population contribution.

## 3. Dyadic stages and thinning probabilities

Let

\[
d_j=d_0 2^{-j},\qquad j=0,1,\ldots,J,
\]

where `d_0` is a fixed constant smaller than the distance from beta to the boundary and `d_J = h` up to a factor of two. Set

\[
h\asymp (\epsilon/C)^{1/\gamma}
\]

so the terminal unresolved mass is at most order epsilon.

At stage `j`, a retained case is sampled to cumulative depth

\[
s_j\asymp L d_j^{-2},
\]

where `L` is logarithmic in the target confidence level and in `1/epsilon`. The decision rule resolves the case positive when a Hoeffding confidence interval lies strictly above beta, negative when it lies strictly below beta, and otherwise calls the case unresolved.

Define

\[
A_J = 1+\sum_{j=1}^J d_j^{\gamma-1},
\]

and a nonincreasing survival schedule

\[
S_j=\min\{1, A_J d_j/d_0\},\qquad S_0=1.
\]

A case that is retained and unresolved after stage `j-1` is independently retained for stage `j` with conditional probability `S_j/S_{j-1}`. Thus an originally sampled case reaches stage `j`, conditional on having remained unresolved, with thinning probability `S_j`.

## 4. Horvitz--Thompson tail estimator

Couple each case to an infinite sequence of potential Bernoulli judgments and independent thinning uniforms. Let `R_j^+` be the event that, under the full unthinned stage sequence, a case would first resolve positive at stage `j`.

For base case `i`, define

\[
Z_i=\sum_{j=0}^J \frac{1\{\text{case i survives to j and first resolves positive at j}\}}{S_j}.
\]

A case contributes at most one summand. The estimator is

\[
\widehat\tau_{HT}=\frac1N\sum_{i=1}^N Z_i.
\]

Because thinning is independent of the case trajectory,

\[
E[Z_i]=\sum_{j=0}^J P(R_j^+),
\]

apart from the controlled probability of confidence-bound misclassification. Therefore the only systematic truncation bias is the positive mass still unresolved at terminal width h plus the confidence-bound failure probability.

## 5. Bias bound

With `s_j = c L d_j^{-2}` and sufficiently large fixed c, Hoeffding gives an exponentially small probability of either a wrong sign decision or failure to resolve a case whose gap is a constant multiple larger than `d_j`.

At the terminal stage,

\[
P(\text{unresolved at J})\le C' h^\gamma + O(e^{-cL}).
\]

Choose `h` above and `L >= c log(J/epsilon)`. Then

\[
|E\widehat\tau_{HT}-\tau_\beta(G)|\le c_1\epsilon.
\]

## 6. Variance bound

Let `q_j=P(R_j^+)`. A case first resolving at stage j must, up to exponentially small confidence errors, have remained within a constant multiple of `d_{j-1}` of beta. Hence

\[
q_j\le C' d_j^\gamma + O(e^{-cL}).
\]

Since survival to stage j has probability `S_j` independently of the potential trajectory,

\[
E[Z_i^2]
\le \sum_{j=0}^J \frac{q_j}{S_j} + o(1).
\]

For stages with `A_J d_j/d_0 >= 1`, `S_j=1` and the dyadic sum of `d_j^gamma` is bounded. For deeper stages,

\[
\frac{d_j^\gamma}{S_j}
\asymp \frac{d_j^{\gamma-1}}{A_J}.
\]

By definition of `A_J`, their sum is also bounded. Thus

\[
\sup_{G\in\mathcal G_{\gamma,C}} Var(Z_i)\le C_2.
\]

Moreover

\[
\|Z_i\|_\infty\le S_J^{-1}.
\]

For `0<gamma<1`, `A_J ~ h^{gamma-1}` and therefore `S_J ~ h^gamma ~ epsilon`; for `gamma=1`, `S_J ~ h log(1/h) >= c epsilon`; and for `gamma>1`, `S_J ~ h = epsilon^{1/gamma} >= epsilon`. Hence `S_J^{-1} <= C/epsilon` up to a log improvement at gamma=1.

Bernstein's inequality with

\[
N\asymp \epsilon^{-2}\log(1/\delta)
\]

therefore gives stochastic error `O(epsilon)` with probability at least `1-delta` (with only logarithmic adjustments when delta varies).

## 7. Expected budget

The incremental cost of reaching stage j is order `L d_j^{-2}`. The probability that a base case is both retained to stage j and still unresolved is at most

\[
C' S_j d_j^\gamma + O(S_j e^{-cL}).
\]

Therefore the expected number of judgments per base case satisfies

\[
E[M_i]\lesssim L\sum_{j=0}^J S_j d_j^{\gamma-2}.
\]

We evaluate this dyadic sum.

### 7.1 `gamma > 1`

Here

\[
A_J=O(1).
\]

Except for a constant number of coarse stages, `S_j ~ d_j`, hence

\[
\sum_j S_j d_j^{\gamma-2}
\lesssim 1+\sum_j d_j^{\gamma-1}=O(1).
\]

Thus

\[
E[B]\lesssim \widetilde O(\epsilon^{-2}).
\]

### 7.2 `gamma = 1`

Now `A_J ~ J ~ log(1/h)`. Splitting at `d_* ~ 1/A_J`, coarse stages have `S_j=1` and deep stages have `S_j ~ A_J d_j`. Both portions are bounded by `O(A_J^2)`, giving

\[
E[B]\lesssim \widetilde O(\epsilon^{-2}),
\]

more explicitly `O(epsilon^{-2} log^2(1/epsilon))` before the additional confidence logarithm.

### 7.3 `0 < gamma < 1`

Here

\[
A_J\asymp h^{\gamma-1}.
\]

Again split at `d_* ~ 1/A_J`. The deep-stage contribution is

\[
A_J\sum_{d_j<d_*} d_j^{\gamma-1}
\asymp A_J^2
\asymp h^{2\gamma-2},
\]

and dominates the coarse-stage contribution. Hence

\[
E[B]\lesssim \widetilde O\left(\epsilon^{-2}h^{2\gamma-2}\right)
=\widetilde O\left(\epsilon^{-2/\gamma}\right).
\]

## 8. Theorem: minimax exponent

Combining this thinned sequential upper bound with the fully adaptive lower bounds

\[
B^*(\epsilon)\gtrsim \epsilon^{-2}
\quad\text{and}\quad
B^*(\epsilon)\gtrsim \epsilon^{-2/\gamma}
\]

gives, for every fixed `gamma>0`,

\[
\boxed{
B^*(\epsilon)
=\widetilde\Theta\left(\epsilon^{-\max\{2,2/\gamma\}}\right).
}
\]

Thus the minimax exponent has a phase transition at

\[
\boxed{\gamma=1.}
\]

- `0<gamma<1`: threshold-near mass is sufficiently heavy that resolving the latent tail is nonparametric and costs `epsilon^{-2/gamma}` up to logs.
- `gamma=1`: critical regime; the exponent is parametric but logarithmic overhead remains in the present construction.
- `gamma>1`: threshold-near mass is thin enough that the parametric `epsilon^{-2}` case-sampling barrier dominates.

## 9. Relation to hard-gap adaptive coin estimation

A fixed hard gap corresponds to zero mass in a neighborhood of beta and is therefore much more regular than any finite-gamma boundary case. Lee--Valiant establish exact dependence on the hard gap and the positive fraction for that class. The present result addresses a different regime in which arbitrarily small gaps are allowed but their population mass decays polynomially. Their work remains the essential adaptive baseline and motivates the fully adaptive lower-bound standard; it is not claimed as a special case of the finite-gamma rate formula.

## 10. Remaining proof-polish tasks

Before calling this camera-ready theorem, the manuscript should spell out:

1. constants for the confidence radii and the terminal h;
2. the exponentially small wrong-resolution terms in the bias and second-moment calculations;
3. the Bernstein high-probability statement with explicit delta dependence;
4. expected-budget versus high-probability-budget variants;
5. a formal definition of the admissible adaptive algorithm class used by the lower bound.

These are proof-completion details, not unresolved exponent questions.
