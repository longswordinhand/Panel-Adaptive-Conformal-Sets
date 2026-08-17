# Soft-margin minimax lower bound: rigorous partial closure

Status: **proved lower bounds**. Together with the later thinned-sequential upper construction in `THINNED_SEQUENTIAL_MINIMAX_THEOREM.md`, these lower bounds match the minimax exponent for all fixed `gamma > 0` up to logarithmic factors. The older corollary below documents the pre-thinning proof stage and is retained only for provenance.

## Setup

For cases `i`, latent expert-positive probabilities are iid `theta_i ~ G`. An algorithm may adaptively choose a case and request another Bernoulli(`theta_i`) judgment, may revisit old cases, and may decide when to introduce new cases. Let `B` denote the total number of expert judgments. The target is

\[
\tau_\beta(G)=P_G(\theta>\beta).
\]

Fix `beta` in the interior of `(0,1)`, and let the local margin class satisfy

\[
G([\beta-t,\beta+t])\le C t^\gamma,\qquad 0<t\le t_{\max}.
\]

A procedure is `(epsilon,delta)`-accurate if

\[
P_G(|\hat\tau-\tau_\beta(G)|\le\epsilon)\ge 1-\delta
\]

uniformly over the class.

Throughout, constants depend only on fixed `beta,C,gamma,t_max,delta` unless stated otherwise.

---

## Theorem L1: parametric lower bound

For every `gamma>0`, any fully adaptive `(epsilon,delta)`-accurate procedure with fixed `delta<1/4` must satisfy

\[
E[B]\ge c\,\epsilon^{-2}
\]

for sufficiently small `epsilon`.

### Proof

Choose two mixing distributions supported at two points a fixed distance away from `beta`, for example `theta_-<beta-t_max` and `theta_+>beta+t_max`:

\[
G_0=(1/2)\delta_{\theta_-}+(1/2)\delta_{\theta_+},
\]

\[
G_1=(1/2-2\epsilon)\delta_{\theta_-}+(1/2+2\epsilon)\delta_{\theta_+}.
\]

Both satisfy the margin condition because they have zero mass in the local threshold neighborhood. Their tail functionals differ by `2 epsilon`.

Strengthen the experiment by revealing the latent support point of a newly introduced case after its first expert judgment (or, equivalently for a lower bound, consider deterministic endpoints `theta_-=0`, `theta_+=1` when the chosen beta/t_max allow them). Repeated judgments on the same case then give no additional information about the mixture weight; at most `B` distinct case types can be learned from `B` judgments. Distinguishing Bernoulli mixture weights `1/2` and `1/2+2 epsilon` from at most `B` iid type draws has KL divergence at most `c B epsilon^2`. Le Cam/Pinsker therefore requires `B >= c' epsilon^{-2}` for constant error probability.

---

## Theorem L2: soft-margin shell lower bound

For every `gamma>0`, any fully adaptive `(epsilon,delta)`-accurate procedure with fixed `delta<1/4` must satisfy

\[
E[B]\ge c\,\epsilon^{-2/\gamma}
\]

for sufficiently small `epsilon`.

Consequently,

\[
E[B]\ge c\,\epsilon^{-\max\{2,2/\gamma\}}.
\]

### Construction

Take a fixed background point `theta_0` with `|theta_0-beta|>t_max`; choose it below beta so it contributes no tail mass. Let

\[
r=4\epsilon,
\qquad
h=(r/C_0)^{1/\gamma},
\]

where `0<C_0<=C` is fixed and epsilon is small enough that `h<t_max` and `beta +/- h` lie in `(0,1)`.

Define

\[
G_0=(1-r)\delta_{\theta_0}
+r\left[\tfrac12\delta_{\beta-h}+\tfrac12\delta_{\beta+h}\right],
\]

and

\[
G_1=(1-r)\delta_{\theta_0}
+r\left[\tfrac14\delta_{\beta-h}+\tfrac34\delta_{\beta+h}\right].
\]

Then

\[
\tau_\beta(G_1)-\tau_\beta(G_0)=r/4=\epsilon.
\]

To use a standard two-point testing reduction for estimation within error epsilon, replace epsilon above by a fixed multiple (e.g. use `r=8 epsilon`) so the two target values are separated by at least `2 epsilon`; this changes only constants.

### Margin verification

For `t<h`, both distributions have no mass in `[beta-t,beta+t]`. For `h<=t<=t_max`, the mass in the threshold neighborhood is exactly `r`. Since `r=C_0 h^gamma<=C t^gamma`, both distributions belong to the margin class.

### Adaptive KL bound

We make the experiment strictly easier for the algorithm: before it spends any expert judgment on a case, reveal whether the case is a background case or a hard-shell case. Shell membership has the same Bernoulli(`r`) law under both hypotheses and therefore contributes zero KL. The algorithm may consequently focus all of its budget on shell cases if it wishes.

For a hard-shell case, its latent sign `Z in {-1,+1}` has prior probability `P_0(Z=+1)=1/2` under `G_0` and `P_1(Z=+1)=3/4` under `G_1`; conditional on Z, every judgment is Bernoulli(`beta+Z h`).

Consider any fully adaptive transcript and apply the KL chain rule one judgment at a time. Conditional on the entire past and on choosing a hard-shell case, the predictive success probability under either hypothesis is a posterior convex combination of `beta-h` and `beta+h`. Hence both predictive probabilities lie in `[beta-h,beta+h]`, and their absolute difference is at most `2h`.

Because beta is bounded away from 0 and 1 and h is sufficiently small, Bernoulli KL obeys uniformly

\[
D_{KL}(Bern(p)\Vert Bern(q))\le c_\beta (p-q)^2\le 4c_\beta h^2.
\]

Background judgments have zero KL because their law is identical under the two hypotheses. Therefore for any adaptive policy whose total number of judgments is B,

\[
D_{KL}(P_0^{\mathcal T}\Vert P_1^{\mathcal T})
\le c h^2 E_0[B].
\]

The same conclusion holds for a random stopping budget by the chain rule/Wald-style summation of conditional KL increments, provided `E_0[B]` is finite.

If an estimator is epsilon-accurate under both alternatives whose tail values are separated by more than `2 epsilon`, it induces a test with constant error probability. Le Cam's two-point lemma and Pinsker require transcript KL bounded away from zero, hence

\[
E_0[B]\ge c h^{-2}
= c (r/C_0)^{-2/\gamma}
\asymp \epsilon^{-2/\gamma}.
\]

This proves Theorem L2.

---

## Corollary: minimax exponent is closed for gamma >= 2

The proved sequential upper bound gives, up to logarithmic factors,

\[
B^*(\epsilon)\lesssim
\begin{cases}
\epsilon^{-1-2/\gamma}, & 0<\gamma<2,\\
\epsilon^{-2}, & \gamma\ge2.
\end{cases}
\]

Theorems L1--L2 give

\[
B^*(\epsilon)\gtrsim \epsilon^{-\max\{2,2/\gamma\}}.
\]

Therefore for `gamma>=2`,

\[
B^*(\epsilon)=\widetilde\Theta(\epsilon^{-2}),
\]

where the tilde hides only logarithmic factors from the current upper construction.

For `0<gamma<2`, the remaining gap is genuine:

\[
\epsilon^{-\max\{2,2/\gamma\}}
\lesssim B^*(\epsilon)
\lesssim \widetilde O(\epsilon^{-1-2/\gamma}).
\]

No matching claim should be made in this regime until a multiscale fully-adaptive lower bound or a sharper upper algorithm closes the gap.

---

## Why a single rare-component Lee--Valiant reduction does not close the gap

Lee--Valiant's hard-gap lower bound places the negative population itself at distance Delta below the threshold, so essentially all coins are in the hard band. Such instances violate a fixed soft-margin condition as Delta tends to zero. Embedding only a rare O(epsilon)-mass hard shell into an easy background does satisfy the margin condition, but it yields only the `epsilon^{-2/gamma}` shell lower bound above: once shell membership is made easy, adaptivity can concentrate its effort there. A matching lower bound for `0<gamma<2` must therefore exploit a genuinely multiscale family (or prove that the current upper bound is not minimax).

## Next proof target

Construct a nested/multiscale family with threshold-neighborhood mass saturating `t^gamma` over dyadic scales and use a fully-adaptive information decomposition (Lee--Valiant-style or an equivalent martingale/Assouad argument). The purpose is to determine whether the true exponent for `0<gamma<2` is the current upper exponent `1+2/gamma` or a smaller rate attainable by a more efficient algorithm.
