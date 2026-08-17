# Finite-Rater Latent-Tail Identification Gate

Date: 2026-08-15

## Question

For a binary expert judgment, let each case have latent expert-positive probability

\[
\theta_i = P_H(Y_{iH}=1\mid i),
\]

and observe only \(m\) expert judgments through

\[
K_i\mid \theta_i \sim \mathrm{Binomial}(m,\theta_i), \qquad \theta_i\sim G.
\]

The target is the latent tail

\[
\tau_\beta(G)=P_G(\theta>\beta).
\]

The gate asks whether fixed finite \(m\) fundamentally prevents point identification of \(\tau_\beta\), and whether this ambiguity is practically material on real multi-reader data.

## Result 1 — explicit non-identifiability construction

For any fixed \(m\), choose \(m+2\) equally spaced support points

\[
x_j=a+jh,\qquad j=0,\ldots,m+1,
\]

with \(\beta\) strictly between two adjacent support points. Define signed coefficients

\[
c_j=(-1)^j {m+1\choose j}.
\]

The finite-difference identity gives

\[
\sum_{j=0}^{m+1} c_j p(x_j)=0
\]

for every polynomial \(p\) of degree at most \(m\). Hence the signed measure

\[
\nu=\sum_{j=0}^{m+1}c_j\delta_{x_j}
\]

annihilates all moments of order \(0,\ldots,m\).

Take a strictly positive base mass vector \(w_j=1/(m+2)\), and for sufficiently small \(\varepsilon>0\), define

\[
G_+=\sum_j (w_j+\varepsilon c_j)\delta_{x_j},\qquad
G_-=\sum_j (w_j-\varepsilon c_j)\delta_{x_j}.
\]

Then \(G_+\neq G_-\), both are probability distributions, and their moments agree through order \(m\). Since each binomial cell probability

\[
P(K=k)=\int {m\choose k}\theta^k(1-\theta)^{m-k}\,dG(\theta)
\]

is a polynomial functional of degree \(m\), \(G_+\) and \(G_-\) induce exactly the same distribution of \(K\).

However, because \(\beta\) lies between adjacent support points, the upper-tail signed mass is a nonzero partial alternating binomial sum. Therefore

\[
P_{G_+}(\theta>\beta)\neq P_{G_-}(\theta>\beta).
\]

This proves that \(\tau_\beta(G)\) is not point identifiable from fixed-\(m\) binomial-mixture observations, even as the number of cases tends to infinity.

### Concrete checks

For \(m=3\), \(\beta=0.5\), support \((0.35,0.45,0.55,0.65,0.75)\), coefficients \((1,-4,6,-4,1)\), and \(\varepsilon=0.02\):

- all moments through order 3 match to numerical precision;
- all four \(P(K=k)\) values match to numerical precision;
- \(P_{G_+}(\theta>0.5)=0.66\);
- \(P_{G_-}(\theta>0.5)=0.54\);
- latent-tail gap = 0.12.

For \(m=5\), \(\beta=0.5\), support \((0.25,0.33,0.41,0.49,0.57,0.65,0.73)\), coefficients \((1,-6,15,-20,15,-6,1)\), and \(\varepsilon=0.005\):

- all moments through order 5 match to numerical precision;
- all six \(P(K=k)\) values match to numerical precision;
- latent-tail values are 0.47857 versus 0.37857;
- latent-tail gap = 0.10.

This is the key structural statement: **more cases cannot remove finite-rater identification uncertainty when \(m\) is fixed.**

## Result 2 — real VinDr m=3 plug-in identified intervals

Using the previously audited image-level vote-count distributions from VinDr-CXR and solving a linear program over a dense grid on \([0,1]\), the following are grid-approximated extremal intervals. They are not yet a formal continuous-support certificate, so they should be treated as numerical evidence for the sharp-bound problem rather than the final theorem statement.

### Pleural thickening

Observed counts \((K=0,1,2,3)=(13019,1099,517,365)\).

- \(\beta=0.2\): approximately [0.04481, 0.22433], width 0.17952.
- \(\beta=0.3\): approximately [0.02316, 0.18891], width 0.16575.
- \(\beta=0.5\): approximately [0.01304, 0.10333], width 0.09029.
- \(\beta=0.7\): approximately [0, 0.07077], width 0.07077.
- \(\beta=0.8\): approximately [0, 0.04469], width 0.04469.

### Lung Opacity

Observed counts \((13678,775,380,167)\).

- \(\beta=0.2\): approximately [0.03163, 0.14941], width 0.11778.
- \(\beta=0.3\): approximately [0.01190, 0.12796], width 0.11606.
- \(\beta=0.5\): approximately [0.00278, 0.06963], width 0.06685.

### Nodule/Mass

Observed counts \((14174,421,224,181)\).

- \(\beta=0.2\): approximately [0.02218, 0.08991], width 0.06773.
- \(\beta=0.3\): approximately [0.01370, 0.07667], width 0.06297.
- \(\beta=0.5\): approximately [0.00711, 0.04676], width 0.03965.

Grid refinement from 1001 to 5001 to 20001 support points changed the reported bounds only at roughly the 1e-4 to 1e-3 level, supporting numerical stability of the observed gaps.

## Gate decision

**GO for the mathematical phenomenon.**

The fixed-rater non-identifiability claim is supported by an explicit finite-difference construction, not only simulation. Real VinDr vote distributions show practically large identification intervals, especially for high-disagreement findings.

This is not yet a claim of publication-level novelty. The next gate is literature/theory verification against truncated Hausdorff moment problems, binomial-mixture identifiability, finite exchangeability/de Finetti results, and partial-identification literature. The numerical LP must also be upgraded to a continuous-support sharp-bound formulation with a dual certificate before calling the bounds formally sharp.
