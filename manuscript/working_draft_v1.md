# Certifying Latent Diagnostic Disagreement from Finite Expert Panels

## Abstract

Clinical benchmark labels often compress several expert judgments into a single consensus label, although persistent expert disagreement may itself be clinically relevant. We study repeated binary expert judgments with a latent case-specific positive-judgment probability \(\theta_i\), observed through \(K_i\mid\theta_i,m_i\sim\mathrm{Binomial}(m_i,\theta_i)\). The target is the tail \(\tau_\beta(G)=P_G(\theta>\beta)\): the population fraction of cases whose probability of a positive judgment exceeds a prespecified threshold. We separate identification uncertainty from finite-sample uncertainty. For a fixed reader count, sharp identified intervals are obtained by linear optimization over mixing distributions compatible with the observable count law. We then construct conservative finite-sample confidence intervals by combining simultaneous exact confidence bounds for the multinomial count probabilities with the same measure optimization; optional local margin restrictions near \(\beta\) preserve coverage while reducing interval width. A corresponding confidence-region feasibility test provides a conservative diagnostic for the exchangeable binomial-mixture observation model. Controlled analysis of 810 NIH chest radiographs read by the same five radiologists shows strong contraction of descriptive identified intervals as reader count increases. At \(\beta=0.5\), pleural-thickening width decreases from 0.236 with one reader to 0.0225 with five readers. VinDr-CXR shows substantial residual uncertainty with three readers despite 15,000 cases. Synthetic experiments with known latent distributions verify interval containment and illustrate the difficulty created by mass concentrated near the threshold.

## 1. Introduction

Expert disagreement is commonly treated as annotation noise to be removed by consensus. This is often convenient for supervised learning, but it erases a different scientific quantity: the prevalence of cases that would remain controversial if additional qualified experts were asked to judge them. In radiology, pathology, dermatology, and other perceptual diagnostic tasks, such persistent ambiguity can matter independently of the final consensus label.

We formalize this quantity using a population-of-experts model. For case \(i\), let \(\theta_i\in[0,1]\) be the probability that a randomly sampled expert from a specified target population gives a positive binary judgment. If \(m_i\) experts judge the case, we observe a count
\[
K_i\mid\theta_i,m_i\sim\mathrm{Binomial}(m_i,\theta_i),
\qquad \theta_i\stackrel{iid}{\sim}G.
\]
The object of interest is not necessarily the full mixing distribution \(G\), but the threshold tail
\[
\tau_\beta(G)=P_G(\theta>\beta),
\]
which is directly interpretable as the fraction of cases for which expert support exceeds a prespecified level. At \(\beta=0.5\), for example, this is the fraction of cases for which a randomly sampled expert is more likely than not to return the positive judgment.

Finite expert panels make this problem fundamentally different from ordinary prevalence estimation. For fixed \(m\), the distribution of \(K\) depends on only finitely many Bernstein-moment functionals of \(G\). Hence the latent tail is generally only partially identified even if the number of cases tends to infinity. This fact is classical in binomial-mixture and truncated-moment theory and is background rather than a novelty claim. Recent work has also developed exact pointwise confidence intervals for the mixing-distribution CDF and quantiles without shape restrictions. Separate work studies smooth mixing-density estimation, including heterogeneous trial counts. Our objective is narrower and application-driven: quantify what a finite expert panel can certify about a clinically chosen tail functional, distinguish identification from sampling uncertainty, and make the assumptions that shorten the interval explicit and auditable.

The paper contributes four components. First, we formulate the sharp identified interval for \(\tau_\beta(G)\) as a convex optimization problem over probability measures. Second, we give a finite-sample conservative confidence construction: simultaneous exact confidence bounds for the observable multinomial probabilities are propagated through the measure optimization, yielding an honest interval over any prespecified class of mixing distributions. Third, the same confidence-region geometry gives a conservative level-\(\alpha\) model-compatibility test for the exchangeable binomial-mixture observation model. Fourth, we provide controlled multi-reader evidence from NIH ChestX-ray14 and external evidence from VinDr-CXR showing that the number of readers, not only the number of cases, materially controls what can be learned about persistent expert support.

A local margin class is used as an optional sensitivity model,
\[
\mathcal G_{\gamma,C}(\beta)=\left\{G: G([\beta-t,\beta+t])\le Ct^\gamma\text{ for all admissible }t\right\}.
\]
This restriction is deliberately local: it limits the amount of latent mass arbitrarily close to the threshold while leaving the distribution unrestricted elsewhere. The margin parameters are not estimated from five votes per case and are therefore reported as sensitivity assumptions rather than fitted truths.

The present manuscript makes a strict distinction between closed and open theory. The finite-sample coverage results below are proved. A stronger minimax characterization of interval length and annotation-budget allocation remains under active derivation and is not used to justify any empirical claim in the current experiments. This separation prevents conjectured rates from being presented as established results.

## 2. Related Work

### 2.1 Binomial mixtures and finite-trial identification

A binomial mixture has probability mass function
\[
p_k=\int_0^1 {m\choose k}\theta^k(1-\theta)^{m-k}\,dG(\theta),\qquad k=0,\ldots,m.
\]
For fixed \(m\), these probabilities determine only finitely many polynomial moments of \(G\), so the mixing distribution is not generally point identified. Geometric treatments of binomial mixtures and the broader truncated Hausdorff moment problem make this limitation explicit. We therefore do not claim finite-reader non-identifiability as new.

### 2.2 Inference for mixing distributions

Nonparametric estimation of mixing distributions and densities in discrete exponential-family models has a long history, including slow-rate inverse problems. Recent work by Basu, Brill, and Yekutieli constructs exact pointwise confidence intervals for the CDF and quantiles of a binomial mixing distribution without shape restrictions. That work is the closest inferential predecessor to ours. Our construction differs operationally in using a transparent simultaneous observable-law confidence region that can be intersected with local margin restrictions and can also be used for model-compatibility testing. We treat their exact unstructured procedure as an important baseline rather than claiming that CDF confidence inference itself is new.

### 2.3 Smoothness and heterogeneous trial counts

Recent work by Lee, Baćak, and Kennedy studies smooth binomial mixing-density estimation under heterogeneous numbers of trials. Consequently, neither smoothness, heterogeneous \(m_i\), nor Bernstein approximation alone constitutes novelty here. The present target is a discontinuous threshold functional under a local, rather than global, structural restriction.

### 2.4 Multi-reader medical imaging

Multi-reader datasets are usually used to create consensus labels, estimate inter-rater agreement, or train models robust to annotation variability. Our use is different: the individual judgments are the data of interest, and no image model is required. This isolates the statistical information supplied by repeated experts from predictive-model error.

## 3. Statistical Formulation

### 3.1 Observation model

For cases \(i=1,\ldots,n\),
\[
K_i\mid\theta_i,m_i\sim\mathrm{Binomial}(m_i,\theta_i),\qquad \theta_i\stackrel{iid}{\sim}G.
\]
When all \(m_i=m\), the observable count law is \(p=(p_0,\ldots,p_m)\) with
\[
p_k=\int B_{k,m}(\theta)\,dG(\theta),
\quad
B_{k,m}(\theta)={m\choose k}\theta^k(1-\theta)^{m-k}.
\]
The model presumes a scientifically meaningful target population of experts and conditional exchangeability of repeated draws from that population. Named-reader effects can violate this interpretation even in a balanced panel; model compatibility must therefore be checked rather than assumed.

### 3.2 Target tail

For a prespecified \(\beta\in(0,1)\),
\[
\tau_\beta(G)=\int \mathbf 1\{\theta>\beta\}\,dG(\theta).
\]
The primary descriptive threshold in this paper is \(\beta=0.5\). A lower threshold can be used as a sensitivity analysis when the scientific question concerns any persistent expert support rather than majority support.

### 3.3 Sharp identified interval

For an exact observable law \(p\), let
\[
\mathcal M_m(p)=\left\{G:\int B_{k,m}\,dG=p_k,\ k=0,\ldots,m\right\}.
\]
The sharp identified interval is
\[
\mathcal I_\beta(p)=
[\tau^-_\beta(p),\tau^+_\beta(p)],
\]
where
\[
\tau^-_\beta(p)=\inf_{G\in\mathcal M_m(p)}\tau_\beta(G),
\qquad
\tau^+_\beta(p)=\sup_{G\in\mathcal M_m(p)}\tau_\beta(G).
\]
The endpoints are sharp by definition: every feasible value is attained or approached by a compatible probability measure. Numerically, we approximate the measure problem on a dense grid and verify convergence by increasing grid resolution.

### 3.4 Local margin sensitivity class

For \(\gamma>0\) and \(C>0\), define
\[
\mathcal G_{\gamma,C}(\beta)
=
\left\{G:G([\beta-t,\beta+t])\le Ct^\gamma,\ 0<t\le t_{\max}\right\}.
\]
The corresponding identified and confidence intervals are obtained by intersecting the feasible measure set with \(\mathcal G_{\gamma,C}(\beta)\). Because the assumption concerns the unobserved distribution near a discontinuity, \((C,\gamma)\) is treated as a sensitivity parameter.

## 4. Finite-Sample Confidence Inference

### 4.1 Simultaneous confidence region for the observable law

For homogeneous reader count \(m\), let \(N_k=\sum_i\mathbf 1\{K_i=k\}\). Marginally,
\[
N_k\sim\mathrm{Binomial}(n,p_k).
\]
For each \(k\), construct a two-sided Clopper--Pearson interval \([L_k,U_k]\) at error level \(\alpha/(m+1)\). By the union bound,
\[
P_p\{p_k\in[L_k,U_k]\ \forall k\}\ge1-\alpha.
\]
Let
\[
\mathcal P_\alpha(N)=\left\{q\in\Delta_m:L_k\le q_k\le U_k\ \forall k\right\}.
\]

### Theorem 1 (finite-sample honest tail interval)

Let \(\mathcal H\) be any prespecified class of probability measures on \([0,1]\), including the unrestricted class or a margin class. Define
\[
\underline\tau_\alpha
=
\inf_G \tau_\beta(G),
\qquad
\overline\tau_\alpha
=
\sup_G \tau_\beta(G),
\]
where the optimization is over all \(G\in\mathcal H\) whose induced count probabilities satisfy
\[
\left(\int B_{0,m}\,dG,\ldots,\int B_{m,m}\,dG\right)
\in\mathcal P_\alpha(N).
\]
If the true \(G_0\in\mathcal H\), then
\[
P_{G_0}\{\tau_\beta(G_0)\in[\underline\tau_\alpha,\overline\tau_\alpha]\}\ge1-\alpha.
\]

**Proof.** With probability at least \(1-\alpha\), all true observable probabilities \(p_{0k}\) lie in their simultaneous intervals. On this event the true mixing distribution \(G_0\) is feasible for the optimization because \(G_0\in\mathcal H\) and its induced count law belongs to \(\mathcal P_\alpha(N)\). Therefore the optimized lower endpoint cannot exceed \(\tau_\beta(G_0)\), and the optimized upper endpoint cannot be below it. \(\square\)

This theorem is deliberately conservative. It requires no asymptotic normal approximation and no parametric model for \(G\). More efficient simultaneous multinomial regions can replace the Bonferroni--Clopper--Pearson box without changing the propagation argument.

### Corollary 1 (margin information cannot widen the interval)

If \(\mathcal H_1\subseteq\mathcal H_0\) and the true \(G_0\in\mathcal H_1\), then the confidence interval obtained under \(\mathcal H_1\) remains \((1-\alpha)\)-honest and is a subset of the interval obtained under \(\mathcal H_0\).

**Proof.** Coverage follows from Theorem 1. The feasible set under \(\mathcal H_1\) is contained in the feasible set under \(\mathcal H_0\), so the infimum cannot decrease and the supremum cannot increase. \(\square\)

### 4.2 Confidence-region model compatibility

Let \(\mathfrak P_m\) be the set of all count laws induced by probability measures on \([0,1]\). We reject the exchangeable binomial-mixture observation model if
\[
\mathcal P_\alpha(N)\cap\mathfrak P_m=\varnothing.
\]

### Proposition 1 (conservative model-compatibility test)

If the data are generated by a binomial mixture, the probability that the above rule rejects is at most \(\alpha\).

**Proof.** Under the null, the true count law \(p_0\in\mathfrak P_m\). Whenever \(p_0\in\mathcal P_\alpha(N)\), the intersection is nonempty. Hence rejection implies failure of the simultaneous confidence event, whose probability is at most \(\alpha\). \(\square\)

This test distinguishes formal incompatibility from a nonzero projection distance of the empirical histogram. A finite empirical histogram can lie outside the mixture polytope even when the population law is compatible.

## 5. Computation

We discretize \([0,1]\) on a dense grid \(\theta_1,\ldots,\theta_J\) and represent a candidate mixing distribution by nonnegative weights \(w_j\) summing to one. Observable-law constraints are linear in \(w\), as are the tail objective and margin constraints. Thus all identified-set, confidence-interval, and compatibility calculations reduce to linear programs.

For descriptive identified intervals we first compute the closest grid mixture to the empirical count histogram in \(L_1\) distance, then optimize the tail under exact reproduction of that projected law. Projection error is reported only as a descriptive diagnostic. For formal finite-sample inference we do not project the empirical histogram; instead, the induced count probabilities are constrained to lie inside the simultaneous confidence region.

## 6. Experiments

### 6.1 Datasets

**NIH ChestX-ray14 additional expert labels.** The all-findings test file contains 4,050 reader-image rows from 810 images. Exactly five radiologists read every image, with no missing reader judgments. This balanced panel permits controlled descriptive subsampling from one to five readers while holding the cases fixed.

**VinDr-CXR.** The training annotations contain 15,000 images and exactly three reader-image pairs per image, across 17 readers. Reader panels are heterogeneous; VinDr is therefore used as an external repeated-reader cohort rather than as the cleanest random-expert design. Binary finding indicators are formed after deduplicating multiple boxes of the same finding from the same reader.

### 6.2 Controlled NIH identification experiment

For each NIH endpoint and each \(m=1,\ldots,5\), we enumerate all \({5\choose m}\) reader subsets, compute the positive-vote count per image, and average the resulting count histograms across subsets. This symmetrizes over the five named readers and preserves the same 810 cases. The resulting intervals are descriptive identified sets rather than finite-sample confidence intervals because the subset-averaged histograms are dependent transformations of the same panel.

At \(\beta=0.5\), six representative endpoints show monotone contraction:

| Endpoint | width \(m=1\) | width \(m=3\) | width \(m=5\) |
|---|---:|---:|---:|
| Abnormal | 0.651 | 0.346 | 0.203 |
| Consolidation | 0.467 | 0.375 | 0.084 |
| Pleural Thickening | 0.236 | 0.0765 | 0.0225 |
| Nodule | 0.339 | 0.180 | 0.101 |
| Pneumothorax | 0.307 | 0.172 | 0.0841 |
| Cardiomegaly | 0.184 | 0.0903 | 0.0507 |

Pleural thickening shows the strongest relative contraction: the five-reader width is about 9.5% of the one-reader width. These results isolate an information effect of repeated judgments: the cases do not change while the number of observed readers does.

### 6.3 Formal NIH model compatibility

Using the full five-reader counts and the 95% and 99% simultaneous exact observable-law regions, all eight audited NIH endpoints remain compatible with the exchangeable binomial-mixture count model. This includes atelectasis and effusion, which had larger empirical projection distances than the other endpoints. The formal result corrects a misleading interpretation of projection error: those endpoints show descriptive tension but are not rejected at conventional confidence levels by the conservative compatibility test.

### 6.4 Honest real-data intervals

At \(m=5\), \(\beta=0.5\), and 95% confidence, unrestricted NIH intervals include:

| Endpoint | lower | upper | width |
|---|---:|---:|---:|
| Abnormal | 0.469 | 0.863 | 0.394 |
| Cardiomegaly | 0.0028 | 0.160 | 0.157 |
| Consolidation | 0.000 | 0.465 | 0.465 |
| Nodule | 0.0259 | 0.294 | 0.268 |
| Pleural Thickening | 0.000 | 0.135 | 0.135 |
| Pneumothorax | 0.0492 | 0.276 | 0.227 |

These are wider than the corresponding descriptive identified intervals because they account for finite-case uncertainty. Under the sensitivity assumption \(\gamma=1,C=2\), all six problems are feasible and the intervals contract, for example from width 0.394 to 0.340 for Abnormal and from 0.465 to 0.355 for Consolidation. The stricter choice \(C=1\) is infeasible for all six endpoints at the 95% observable-law confidence level, illustrating why the margin assumption must be reported as a sensitivity model rather than tuned for favorable widths.

For VinDr at \(m=3\), \(\beta=0.5\), unrestricted 95% widths are approximately 0.108 for Pleural thickening, 0.080 for Lung Opacity, 0.051 for Nodule/Mass, 0.104 for Cardiomegaly, and 0.131 for Aortic enlargement. The large number of cases substantially reduces sampling uncertainty, but finite-reader ambiguity remains visible.

### 6.5 Synthetic truth-known validation

We generate latent probabilities from Beta distributions and mixtures, sample binomial reader counts, and compare the computed intervals with the known \(\tau_{0.5}(G)\). In population-level identification experiments, all tested true tails are contained in the computed identified intervals for \(m\in\{1,2,3,5,8,12,20\}\). Interval contraction depends strongly on mass near the threshold. For a distribution concentrated near 0.5, \(\mathrm{Beta}(20,20)\), the identified width remains about 0.404 even at \(m=20\), whereas an asymmetric mixture with less mass near the threshold contracts to about 0.033.

A finite-sample sanity simulation of the exact-confidence implementation also contains the true tail in every tested replicate. Because the current run uses only 20 replicates per configuration, these values are treated as implementation checks rather than estimates of 95% empirical coverage. Larger Monte Carlo experiments are required before reporting coverage frequencies to publication precision.

## 7. Open Minimax Theory and Annotation Design

The finite-sample coverage theory above is closed, but the stronger TPAMI-level theory target is not yet closed. We seek a characterization of the minimax expected length
\[
\inf_{CI}\sup_{G\in\mathcal G_{\gamma,C}(\beta)}E_G|CI|
\]
under repeated binomial observations, and then the allocation minimizing this quantity subject to \(\sum_i m_i\le B\). Numerical experiments indicate a severe stability cost when a Bernstein-form estimator is forced to approximate the threshold step uniformly outside a band narrower than the natural binomial resolution. These experiments motivate, but do not prove, a possible severely ill-posed regime.

A rigorous elementary bound is already available. If a count-based linear estimator has expectation \(p_a(\theta)=\sum_k a_k B_{k,m}(\theta)\) and satisfies
\[
p_a(\beta-\delta)\le\varepsilon,\qquad p_a(\beta+\delta)\ge1-\varepsilon,
\]
then
\[
1-2\varepsilon
\le 2\|a\|_\infty\,\mathrm{TV}\{\mathrm{Bin}(m,\beta-\delta),\mathrm{Bin}(m,\beta+\delta)\}.
\]
For \(\beta\) bounded away from 0 and 1, Pinsker's inequality and the Bernoulli KL expansion imply
\[
\|a\|_\infty\gtrsim (\delta\sqrt m)^{-1}
\]
when \(\delta\sqrt m\) is small. This establishes a necessary stability cost but is not yet the matching minimax lower bound for arbitrary estimators.

Until that lower bound and a matching construction are established, the manuscript will not claim a minimax rate or an optimal budget scaling.

## 8. Discussion

The main empirical message is simple but easily obscured by consensus labeling: more cases and more readers answer different statistical questions. A large number of cases can estimate the observable distribution of a three-reader vote count very precisely while leaving a nontrivial range of latent expert-tail values compatible with that observable law. Conversely, adding readers to the same cases can contract that identified range substantially.

The framework also forces assumptions to be visible. The latent \(\theta_i\) interpretation requires a target expert population and approximate conditional exchangeability. The conservative compatibility test can reject gross violations at the observable-law level, but failure to reject is not proof of exchangeability or absence of named-reader effects. Extensions with explicit reader random effects would address a different, richer model and are not claimed here.

The local margin condition should likewise be interpreted carefully. It is not learned reliably from five votes per case. Its scientific meaning is that only limited population mass lies arbitrarily close to the decision threshold. Reporting intervals over a grid of \((C,\gamma)\) values is therefore more defensible than selecting one pair after observing which gives the narrowest result.

The current exact confidence construction prioritizes transparency and guaranteed coverage over efficiency. The Bonferroni--Clopper--Pearson observable-law box can be replaced by a smaller simultaneous multinomial region or by direct inversion of a sharper test. Basu et al.'s exact CDF procedure is an important comparison for the unrestricted case. The contribution of the present framework is the explicit propagation of observable-law uncertainty through a clinically targeted tail functional, optional local structural information, and a model-compatibility geometry that uses the same optimization machinery.

## 9. Conclusion

Finite expert panels do not reveal a single deterministic label; they provide repeated samples from a case-dependent judgment process. By targeting the fraction of cases whose latent expert-positive probability exceeds a prespecified threshold, we make persistent diagnostic disagreement an explicit inferential object. Sharp descriptive identified intervals quantify the information supplied by reader count, while conservative finite-sample intervals separate that identification uncertainty from sampling uncertainty. Controlled NIH experiments show large contractions in compatible tail ranges as the number of readers increases, and VinDr shows that many cases cannot fully compensate for only three judgments per case. The remaining theoretical challenge is to characterize the shortest achievable margin-adaptive interval and convert that limit into an optimal expert-allocation law. Those rates are deliberately left unclaimed until matching proofs are complete.

## Reproducibility

Primary scripts:

- `scripts/run_tail_identification_experiments.py`
- `scripts/run_synthetic_tail_sanity.py`
- `scripts/run_honest_tail_ci_simulation.py`
- `scripts/run_exact_model_compatibility.py`
- `scripts/run_real_honest_tail_ci.py`

Primary outputs:

- `experiments/pilot/results/tail_identification/identified_tail_bounds.csv`
- `experiments/pilot/results/tail_identification/nih_m_contraction_summary_beta05.csv`
- `experiments/pilot/results/tail_identification/synthetic_population_bounds.csv`
- `experiments/pilot/results/model_compatibility/nih_m5_exact_compatibility.csv`
- `experiments/pilot/results/tail_honest_ci/real_data_honest_ci_sensitivity.csv`
- `experiments/pilot/results/tail_honest_ci/simulation_summary.csv`

## Submission gate

The manuscript is **not yet TPAMI-ready** until the following are closed:

1. a defensible minimax lower bound for the margin-tail functional, or an explicit editorial decision to submit the finite-sample method without a minimax claim;
2. a matching or near-matching upper construction sufficient to justify an annotation-budget design theorem;
3. a publication-scale Monte Carlo coverage study;
4. final citation verification and comparison with Basu--Brill--Yekutieli and Lee--Baćak--Kennedy;
5. IEEE two-column typesetting, figures, supplement, and final reviewer-style audit.
