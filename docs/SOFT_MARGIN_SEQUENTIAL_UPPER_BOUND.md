# Soft-margin sequential upper bound

Status: proved upper bound; **not** a matching minimax theorem.

## Setting

Draw cases independently with latent expert-positive probabilities `theta_i ~ G`. For each case, reader judgments are conditionally iid Bernoulli(`theta_i`). Let `D_i=|theta_i-beta|` and assume the local margin condition

\[
P(D_i\le t)\le C t^\gamma,\qquad 0<t\le t_{\max}.
\]

The target is `tau_beta=P(theta_i>beta)`.

## Sequential procedure

Sample `n` cases. For each case `i`, collect reader judgments sequentially up to a cap `T`. After `s` readers let `barY_{i,s}` be the empirical positive fraction and

\[
r_s=\sqrt{\frac{L}{2s}},\qquad L=\log\frac{4nT}{\delta}.
\]

Stop and label the case positive if `barY-r_s>beta`, negative if `barY+r_s<beta`, otherwise continue until `T`. Cases still unresolved at `T` are left unresolved.

Let `P_n` and `U_n` be the numbers of resolved-positive and unresolved cases. Define

\[
\eta_n=\sqrt{\frac{\log(4/\delta)}{2n}},
\]

and the interval

\[
CI=[(P_n/n-\eta_n)_+,\ ((P_n+U_n)/n+\eta_n)\wedge1].
\]

## Theorem A: honest coverage

With probability at least `1-delta`, the interval contains `tau_beta(G)`.

### Proof

For any fixed `(i,s)`, Hoeffding gives

\[
P(|\bar Y_{i,s}-\theta_i|>r_s)\le 2e^{-2sr_s^2}=\frac{\delta}{2nT}.
\]

A union bound over all `nT` case-time pairs gives an event `E_CS` of probability at least `1-delta/2` on which every confidence interval is valid. On this event every resolved label has the correct sign relative to `beta`, so if `S_i=1{theta_i>beta}`,

\[
P_n\le \sum_i S_i\le P_n+U_n.
\]

Independently, Hoeffding across the iid cases gives

\[
P\left(\left|n^{-1}\sum_iS_i-\tau_\beta\right|>\eta_n\right)\le \delta/2.
\]

Intersecting the two events proves coverage by another union bound.

## Theorem B: unresolved-mass width bound

On `E_CS`, any case unresolved at time `T` obeys

\[
D_i\le2r_T=\sqrt{\frac{2L}{T}}.
\]

Therefore

\[
P(U_i=1, E_{CS})\le C\left(\frac{2L}{T}\right)^{\gamma/2}.
\]

Combining this with concentration across cases yields, up to universal confidence terms,

\[
|CI|\lesssim
C\left(\frac{L}{T}\right)^{\gamma/2}
+\sqrt{\frac{\log(1/\delta)}{n}}.
\]

Thus choosing `n` of order `epsilon^{-2}` and `T` of order `L epsilon^{-2/gamma}` makes the interval width `O(epsilon)` up to logarithmic factors.

## Theorem C: expected reader budget

On a valid confidence-sequence path, a case with gap `D>0` stops once `r_s<D/2`, hence after at most

\[
1+\frac{2L}{D^2}
\]

readers, capped by `T`. Consequently

\[
E[M_i]\lesssim E\left[\min\left\{T,\frac{L}{D^2}\right\}\right]+T\frac{\delta}{2n}.
\]

Under the margin condition, a standard split/integration argument gives

\[
E\left[\min\left\{T,\frac{L}{D^2}\right\}\right]
\lesssim
\begin{cases}
L^{\gamma/2}T^{1-\gamma/2},&0<\gamma<2,\\
L\log(1+T/L),&\gamma=2,\\
L,&\gamma>2,
\end{cases}
\]

where constants may depend on `C`, `gamma`, `beta`, and the fixed local radius `t_max`.

Plugging `n ~ epsilon^{-2}` and `T ~ L epsilon^{-2/gamma}` gives the **achievable expected total reader budget**, up to logarithmic factors,

\[
E[B]\lesssim
\begin{cases}
\epsilon^{-1-2/\gamma},&0<\gamma<2,\\
\epsilon^{-2},&\gamma\ge2.
\end{cases}
\]

The transition at `gamma=2` is an upper-bound phenomenon at this stage. It must not be called minimax until a matching fully-adaptive lower bound is proved.

## Relation to Lee--Valiant (SODA 2021)

Lee and Valiant solve a different hard-gap class in which every coin is at least `Delta` away from the threshold and prove tight fully-adaptive complexity. Their least-favorable hard-gap populations do **not** satisfy a fixed soft-margin condition as `Delta -> 0`, so their lower bound cannot simply be substituted for the missing soft-margin lower bound. Their work remains the essential hard-gap baseline and source of lower-bound techniques.
