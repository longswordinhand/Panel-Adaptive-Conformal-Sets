# Candidate minimax theorem: finite-rater tail certification under a margin condition

## Model
For i=1,...,n,
- theta_i iid~G on [0,1]
- K_i | theta_i ~ Binomial(m,theta_i)
- total annotation budget B=nm
- target tau_beta(G)=P_G(theta>beta)

Assume beta is bounded away from 0 and 1 and, for some gamma>0, C>0, t0>0,

M(gamma,C,t0):  G([beta-t,beta+t]) <= C t^gamma,  0<t<=t0.

This controls only local mass near the clinically meaningful threshold beta; it is not a parametric model for G.

## Candidate estimator

hat_tau = (1/n) sum_i 1{K_i/m > beta}.

## Upper-bound derivation
Let D=|theta-beta|. Conditional Hoeffding gives, on either side of beta,

P( 1{K/m>beta} != 1{theta>beta} | theta ) <= exp(-2m D^2),

up to the harmless lattice convention at m beta when beta*m is integer.
Hence

|E_G[hat_tau]-tau_beta(G)| <= E_G exp(-2mD^2).

For phi(t)=exp(-2mt^2), integration by parts / layer-cake yields

E phi(D) = integral_0^infty 4mt exp(-2mt^2) P(D<=t) dt.

Using P(D<=t)<=C t^gamma for t<=t0 and <=1 afterward,

bias <= C * 4m * integral_0^infty t^(gamma+1) exp(-2mt^2) dt + exp(-2m t0^2)
     = C_gamma,C * m^(-gamma/2) + exp(-2m t0^2),

where an explicit admissible constant is

C_gamma,C = C * 2^(-gamma/2) * Gamma((gamma+2)/2).

Since the summands are Bernoulli, Hoeffding across cases gives with probability >=1-delta,

|hat_tau-E hat_tau| <= sqrt(log(2/delta)/(2n)).

Therefore a finite-sample one-sided certificate is

U_beta = hat_tau
       + C_gamma,C m^(-gamma/2)
       + exp(-2m t0^2)
       + sqrt(log(1/delta)/(2n)),

and uniformly over G in M(gamma,C,t0),

P_G(tau_beta(G) <= U_beta) >= 1-delta.

Ignoring the exponentially small term and constants,

risk/certificate width <= m^(-gamma/2)+n^(-1/2).

With B=nm,

R(B,m) <= m^(-gamma/2)+sqrt(m/B).

Balancing the two terms gives

m* ~ B^(1/(gamma+1)),
n* ~ B^(gamma/(gamma+1)),
R* ~ B^(-gamma/(2(gamma+1))).

For gamma=1 (bounded density around beta):

m* ~ B^(1/2), n* ~ B^(1/2), R* ~ B^(-1/4).

This is qualitatively different from ordinary repeated-measures design: the objective is a discontinuous latent-tail functional, and the slow B^-1/4 rate arises from balancing threshold-resolution error against population-sampling error.

## Matching lower-bound construction: current proof sketch

Goal: show no procedure/design with total budget B can beat order B^(-gamma/(2(gamma+1))) uniformly over the margin class.

Take a common background distribution H supported far from beta. Let h>0 be small and q=a h^gamma. Define two alternatives that differ only by a q-sized local component shifted across beta:

G0=(1-q)H + q delta_{beta-h},
G1=(1-q)H + q delta_{beta+h}.

Then

|tau_beta(G1)-tau_beta(G0)| = q = a h^gamma.

Both can be chosen to satisfy the margin condition by selecting a<=C times a fixed constant and keeping H outside [beta-t0,beta+t0].

For one case with m Bernoulli repeats, the two induced count laws are mixtures

P0=(1-q)P_H + q Bin(m,beta-h),
P1=(1-q)P_H + q Bin(m,beta+h).

When the common background gives positive mass across the relevant count support, a chi-square/KL expansion should yield

KL(P0||P1) <= c q^2 m h^2

for beta bounded away from 0,1 and h small. Across n independent cases,

KL(P0^n||P1^n) <= c n q^2 m h^2 = c B h^(2gamma+2).

Choose h ~ B^(-1/(2gamma+2)); then total KL is O(1), while tail separation is

q ~ h^gamma ~ B^(-gamma/(2(gamma+1))).

Le Cam's two-point lemma would then imply the matching minimax lower bound.

### Important caveat
The q^2 scaling in mixture KL must be proved carefully by an explicit common background H; if H is chosen poorly or has insufficient overlap, the divergence can scale like q rather than q^2 and the rate changes. This is the main technical point still needing a rigorous proof.

## Adaptive allocation question
For arbitrary adaptive m_i with sum m_i<=B, a lower bound should be formulated against the full sequential experiment. If the likelihood-ratio information contributed by a case receiving m_i reads is at most c q^2 m_i h^2, then the KL chain rule gives total information <=c q^2 h^2 sum_i m_i <=c q^2 h^2 B, implying the same lower bound for *every adaptive policy*. If true, balanced designs are minimax-rate optimal, although adaptive designs may improve constants.

This would be a strong theorem: adaptivity cannot beat the fundamental B-exponent for latent-tail certification under the margin class.

## Novelty status
Targeted searches on 2026-08-15 found:
- classical and modern binomial-mixture identification/CDF inference;
- repeated-measures optimal designs for ICC, power, Fisher information, longitudinal effects;
- generic crowdsourcing budget allocation for majority-vote accuracy;
- general partial-identification decision theory;
- minimax CDF deconvolution under additive measurement error;
- margin-condition theory in classification and partially identified causal bounds.

No located source stated this exact binomial-repeat latent-tail certification problem, the B=nm minimax rate above, or an adaptive-policy lower bound. This remains a candidate gap, not a confirmed novelty claim until the lower bound and broader literature audit are complete.
