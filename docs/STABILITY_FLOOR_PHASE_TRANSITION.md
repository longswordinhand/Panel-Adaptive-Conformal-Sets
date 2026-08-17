# Stability-floor phase transition (2026-08-15)

## Status
Numerical/theoretical gate. Do not cite as a proved theorem yet.

## Observation operator
For one case with m readers,

K | theta ~ Binomial(m, theta).

Any linear estimator based on the count K has expectation

p_a(theta) = sum_{k=0}^m a_k B_{k,m}(theta),

where B_{k,m} is the Bernstein/binomial basis. The coefficient vector a controls statistical noise, while p_a controls approximation to the tail indicator 1{theta > beta}.

## Endpoint-separation lower bound
If p_a(beta-delta) <= eps and p_a(beta+delta) >= 1-eps, then

1-2 eps <= |E_{beta+delta} a_K - E_{beta-delta} a_K|
          <= 2 ||a||_infty TV(Bin(m,beta-delta), Bin(m,beta+delta)).

For beta bounded away from 0 and 1, Pinsker plus the Bernoulli KL expansion gives

TV <= C_beta delta sqrt(m)

when delta sqrt(m) is small. Hence

||a||_infty >= c_beta,eps / (delta sqrt(m)).

This is rigorous up to writing the constants carefully.

## Full off-band approximation is much harder
For tail estimation over a margin class, endpoint separation alone is insufficient. To keep worst-case bias small for arbitrary mass outside the threshold neighborhood, p_a must approximate 0 on [0,beta-delta] and 1 on [beta+delta,1].

LP experiments minimizing ||a||_infty under these uniform constraints show a severe instability when c = delta sqrt(m) becomes small. For eps=0.05 and m=40--120:

- c=0.50: ||a||_infty ~ 1.13
- c=0.40: ~ 2.0--2.2
- c=0.35: ~ 4.9--5.7
- c=0.30: ~ 26--37
- c=0.25: ~ 5e2--1.3e3
- c=0.20: ~ 1.4e5--2.1e6

Across m, the dependence is approximately a function of c=delta sqrt(m), and log ||a|| grows roughly linearly in 1/c^2 over the probed range. This suggests

||a||_infty ~ exp{ C_eps / (m delta^2) }

in the super-resolution regime delta << m^{-1/2}. This is a conjecture, not yet a theorem.

## Consequence if the conjecture is proved
The identification floor under a gamma-margin class is numerically consistent with m^{-gamma}. Reaching delta ~ 1/m would then require coefficients of order exp(C m), so finite-sample variance would require n exponential in m.

Under total budget B = n m, balancing an identification error m^{-gamma} against exponential inversion instability suggests

m* ~ c log B,

and therefore a possible budget minimax rate

R_B* ~ (log B)^{-gamma}

up to log-log factors/constants.

This would be qualitatively different from the earlier naive B^{-gamma/[2(gamma+1)]} calculation based on the plug-in threshold estimator.

## Proof tasks
1. Prove a lower bound on Bernstein coefficient norm for uniform two-interval approximation of a step, ideally exp{c/(m delta^2)} for delta sqrt(m) -> 0.
2. Construct a matching upper approximation with coefficient norm exp{C/(m delta^2)}.
3. Convert coefficient norm to minimax finite-n risk, not only linear-estimator risk.
4. Combine with the m^{-gamma} identification lower bound.
5. Optimize over m under B=nm; verify whether the sharp result is (log B)^{-gamma} and identify log-log corrections.

## Numerical artifacts
- experiments/pilot/results/bernstein_stability_probe.csv
- experiments/pilot/results/bernstein_stability_probe_relaxed.csv
- experiments/pilot/results/bernstein_scale_probe.csv
- experiments/pilot/results/bernstein_full_probe.csv
