# Working manuscript v0 — 2026-08-15

## Working title

**Finite Expert Panels Limit Inference on Latent Diagnostic Disagreement**

Alternative theory-forward title:

**Minimax Tail Inference from Finite Expert Panels**

## One-sentence argument

In repeated expert diagnosis, we study the latent fraction of cases whose positive-judgment probability exceeds a clinically chosen threshold, show that finite expert panels induce a nontrivial identification and stability barrier for this tail functional, and develop a margin-adaptive inference framework whose theory is evaluated on NIH and VinDr multi-reader labels without using images or predictive models.

## Terminology ledger

- case: one diagnostic item/image
- expert / reader: one clinician providing a binary judgment
- reader count: `m` or case-specific `m_i`
- latent expert-positive probability: `theta_i`
- mixing distribution of case difficulty/disagreement: `G`
- tail threshold: `beta`
- latent disagreement tail: `tau_beta(G) = P_G(theta > beta)`
- local margin class: `G([beta-t,beta+t]) <= C t^gamma`
- observable count: `K_i`
- identified interval: sharp range of `tau_beta(G)` over mixing distributions compatible with the observable count law
- projection error: L1 distance from the empirical count distribution to the grid-discretized binomial-mixture model; used only as a model diagnostic

---

# Abstract — provisional; rewrite after the main theorem is closed

Clinical labels are often treated as if a finite panel of experts revealed a single underlying truth, even when expert judgments vary systematically across cases. We consider repeated binary expert judgments in which case `i` has an unobserved expert-positive probability `theta_i`, and only `m_i` expert votes are observed. Our target is not the full distribution of `theta_i`, but the clinically interpretable tail `tau_beta(G)=P(theta_i>beta)`: the proportion of cases for which a randomly drawn expert is more likely than a specified threshold to give the positive judgment. Finite panels make this tail only partially identified, and sharper recovery requires an increasingly unstable inversion of the binomial observation operator. We formulate margin-adaptive inference for `tau_beta(G)` under a local restriction on probability mass near the threshold and derive the corresponding minimax and confidence-set problem. In public multi-reader chest-radiograph labels, controlled subsampling of the same five NIH readers shows substantial contraction of the identified tail interval as the number of readers increases. At `beta=0.5`, the interval width for pleural thickening decreases from 0.236 with one reader to 0.0225 with five readers; cardiomegaly decreases from 0.184 to 0.0507. VinDr-CXR independently exhibits substantial finite-reader ambiguity with three readers per case, including interval widths of 0.090 for pleural thickening and 0.067 for lung opacity. These results motivate inference and study design that quantify what finite expert panels can certify about population-level diagnostic disagreement rather than collapsing disagreement to a consensus label.

**STATUS:** the abstract intentionally does not yet claim a proved minimax rate or optimal confidence interval. Insert theorem-level claims only after matching upper and lower bounds are closed.

---

# 1. Introduction

Expert disagreement is not merely annotation noise. In radiology, pathology, dermatology, and other diagnostic specialties, two qualified readers may reach different conclusions on the same case because the case itself lies near a perceptual or clinical decision boundary. A finite panel therefore reveals only a small number of draws from an underlying case-specific distribution of expert judgments. Yet most benchmark construction and downstream statistical analysis compress these repeated judgments into a majority label, an average score, or a consensus reference. This compression discards a clinically relevant question: how many cases would remain genuinely controversial if additional experts were asked to read them?

We study this question through a latent expert-positive probability. For case `i`, let `theta_i` denote the probability that a randomly sampled expert gives a positive binary judgment, and suppose `m_i` experts are observed so that `K_i | theta_i,m_i ~ Binomial(m_i,theta_i)`. Across cases, `theta_i` follows an unknown mixing distribution `G`. Rather than estimating the full mixing density, we focus on the threshold tail

`tau_beta(G) = P_G(theta_i > beta)`,

which directly quantifies the fraction of cases whose expert-positive probability exceeds a clinically chosen level `beta`. For `beta=0.5`, for example, `tau_beta(G)` is the fraction of cases for which a randomly sampled expert is more likely than not to return a positive judgment. Other choices of `beta` express stricter or more permissive notions of persistent expert support.

This functional is difficult for two distinct reasons. First, with a fixed number of readers, the observable distribution of `K_i` contains only finitely many moment functionals of `G`; therefore, even infinitely many cases need not point identify `tau_beta(G)`. This finite-rater non-identifiability is a consequence of classical binomial-mixture and moment theory and is not itself our novelty. Second, exploiting the additional information delivered by larger reader panels requires inversion of the binomial observation operator. Numerical evidence in this project shows that uniformly approximating the discontinuous threshold functional with the associated Bernstein basis becomes rapidly unstable when resolution is pushed below the natural binomial scale. The resulting distinction between an identification limit and a finite-sample stability limit motivates the inference problem studied here.

Existing work establishes several adjacent pieces but does not resolve the target problem in the form needed for finite expert panels. Classical binomial-mixture theory characterizes non-identifiability at fixed trial count, and moment-problem theory gives extremal distribution-function bounds from finitely many moments. Recent work provides exact pointwise confidence intervals for mixing-distribution CDFs and quantiles without shape restrictions, while separate work studies smooth binomial mixing-density estimation, including heterogeneous trial counts and Bernstein approximation. Our target differs in three linked ways: it is a discontinuous threshold-tail functional; structural information is imposed only locally through a margin condition near the threshold rather than through a global parametric density model; and the inferential loss is tied directly to reader allocation under a finite annotation budget. [CITATIONS TO INSERT FROM VERIFIED BIBLIOGRAPHY]

We therefore consider the local class

`G([beta-t,beta+t]) <= C t^gamma`,

which limits the amount of latent case mass arbitrarily close to the decision threshold while leaving the distribution unrestricted away from it. The paper has three theoretical goals. First, we seek matching minimax upper and lower bounds for estimating `tau_beta(G)` from repeated binomial judgments. Second, we construct honest confidence intervals and characterize the shortest achievable worst-case expected length over the local margin class. Third, under a total expert-reading budget `sum_i m_i <= B`, we characterize how reader effort should be allocated across cases to minimize tail-certification uncertainty. These results are complemented by public multi-reader experiments in which the images themselves are unnecessary: only the individual expert labels are used.

Our empirical analysis uses the NIH ChestX-ray14 additional expert labels as the primary controlled panel. The all-findings test set contains 810 images, each read by the same five radiologists, allowing exact subsampling from `m=1` to `m=5` without changing cases or reader composition. VinDr-CXR provides an external three-reader cohort. The controlled NIH analysis shows that additional readers can sharply reduce the identified range of the latent tail for model-compatible endpoints. At `beta=0.5`, pleural-thickening width decreases from 0.236 at `m=1` to 0.0225 at `m=5`; cardiomegaly decreases from 0.184 to 0.0507; and nodule decreases from 0.339 to 0.101. We also find endpoint-specific departures from the exchangeable-binomial observation model, most notably for atelectasis and effusion at five readers. We treat these departures as a model-checking result rather than excluding them silently.

The central claim of the paper is consequently bounded. We do not argue that expert disagreement has a single ground-truth probability for every diagnostic task, nor that experts are automatically exchangeable. Instead, we provide a formal framework for the subset of repeated-reader problems in which a latent population-of-experts interpretation is scientifically defensible, make the model testable at the level of the observable count law, and quantify how much can be inferred about persistent disagreement from a finite panel.

---

# 2. Problem formulation

## 2.1 Repeated expert judgments

For cases `i=1,...,n`, let

`K_i | theta_i,m_i ~ Binomial(m_i,theta_i)`,

and let `theta_i iid~ G` on `[0,1]`. The reader count `m_i` may be fixed or heterogeneous. The interpretation of `theta_i` is the probability of a positive judgment under repeated sampling from a specified expert population. This interpretation requires a scientifically meaningful target expert population and approximate exchangeability conditional on the case; it is not implied merely by the presence of multiple named readers.

The observable law for fixed `m` is

`P(K=k) = integral B_{k,m}(theta) dG(theta)`,

where

`B_{k,m}(theta) = choose(m,k) theta^k (1-theta)^(m-k)`.

Thus the binomial count probabilities are Bernstein-moment functionals of `G`.

## 2.2 Target tail functional

For a prespecified threshold `beta in (0,1)`, define

`tau_beta(G) = integral 1{theta>beta} dG(theta)`.

The threshold must be selected before looking at endpoint-specific results when used for confirmatory inference. We use `beta=0.5` as the primary descriptive threshold and `beta=0.2` as a sensitivity threshold in the current experiments.

## 2.3 Local margin class

Define

`G_{gamma,C}(beta) = {G: G([beta-t,beta+t]) <= C t^gamma for all t in an admissible range}`.

This condition controls only probability mass close to the functional discontinuity. It allows arbitrary multimodality and atoms away from `beta`.

## 2.4 Identification set

For an observable count distribution `p=(p_0,...,p_m)`, define

`I_beta(p) = {tau_beta(G): integral B_{k,m} dG = p_k, k=0,...,m}`.

The sharp identified interval is

`[tau_beta^-(p), tau_beta^+(p)]`,

with endpoints obtained by minimizing and maximizing the tail mass over compatible probability measures.

**Background proposition, not a novelty claim.** For fixed finite `m`, `tau_beta(G)` is not point identified over unrestricted `G` in general. The proof follows from finite-moment/binomial-mixture non-identifiability.

---

# 3. Theory roadmap

## 3.1 Identification under a margin condition

**Theorem 1 — PLACEHOLDER.** Establish upper and lower bounds on the diameter of the identified set over `G_{gamma,C}(beta)` as `m` increases.

Current numerical evidence is consistent with a power-law contraction close to `m^{-gamma}` in the examined `gamma=1` setting, but this is not yet a proved result and must not appear as a final theorem until the analytic bound is closed.

## 3.2 Stability of the binomial inverse

For a linear count estimator

`T_hat_a = n^{-1} sum_i a_{K_i}`,

its conditional expectation at latent probability `theta` is the Bernstein polynomial

`p_a(theta) = sum_{k=0}^m a_k B_{k,m}(theta)`.

To estimate `tau_beta(G)`, `p_a` must approximate a step function away from a threshold band. A rigorous endpoint-separation inequality already obtained in this project gives

`1-2 eps <= 2 ||a||_infty TV(Bin(m,beta-delta), Bin(m,beta+delta))`,

and hence, for `beta` bounded away from the endpoints and small `delta sqrt(m)`,

`||a||_infty >= c / (delta sqrt(m))`.

**Theorem 2 — PLACEHOLDER.** Prove the stronger full off-band stability bound needed for uniform tail approximation. Numerical linear programs suggest much faster coefficient growth in the super-resolution regime, approximately exponential in `1/(m delta^2)`, but this remains a conjecture.

## 3.3 Honest confidence intervals

**Theorem 3 — PLACEHOLDER.** Construct an honest `(1-alpha)` confidence interval for `tau_beta(G)` over `G_{gamma,C}(beta)` and prove a matching lower bound on worst-case expected interval length.

The implementation will be based on finite-sample constraints on the observable multinomial/binomial count law combined with convex constraints representing the local margin class. The final method must report sensitivity to `(C,gamma)` rather than selecting these hyperparameters post hoc for favorable interval widths.

## 3.4 Budget-optimal expert allocation

Let the total number of expert judgments satisfy

`sum_i m_i <= B`.

**Theorem 4 — PLACEHOLDER.** Characterize the reader allocation minimizing worst-case expected confidence-interval length or certification risk. The theorem must be derived from the closed minimax theory above; no power-law or logarithmic allocation rate is claimed at this draft stage.

---

# 4. Experiments

## 4.1 Goals

The experiments are designed to test three empirical questions rather than to establish the minimax theorem numerically.

1. Does the observable count law approximately satisfy the exchangeable-binomial mixture model for a given diagnostic endpoint?
2. Conditional on approximate model compatibility, how strongly does the sharp descriptive identified interval contract as the number of readers increases on the same cases?
3. Are large finite-reader identified intervals also present in an independent multi-reader dataset?

## 4.2 NIH ChestX-ray14 additional expert labels

We use the all-findings individual-reader test labels. The file contains 4,050 reader-image rows corresponding to 810 images read by exactly five radiologists. There are no missing reader labels or duplicate reader-image rows, and the same five-reader panel reads every image. We analyze binary findings separately.

For each endpoint and each `m=1,...,5`, we enumerate all `choose(5,m)` reader subsets. For every subset we compute the positive-vote count `K`, aggregate across the 810 images, and average across reader subsets to obtain the controlled `m`-reader observable count distribution. This preserves the same cases and uses the full five-reader panel symmetrically rather than selecting a favorable subset.

Because a finite-sample empirical count distribution need not lie exactly in the convex hull of binomial-mixture distributions, we first compute its L1 projection onto a fine-grid binomial-mixture model. We report this projection distance as an explicit model diagnostic. Tail bounds are then obtained by linear programming over the projected observable distribution. These intervals are descriptive identified sets, not finite-sample confidence intervals.

The grid contains 4,001 equally spaced points on `[0,1]`. We minimize and maximize mass above `beta` subject to exact reproduction of the projected count probabilities. Repeating the calculation after increasing the grid from 2,001 to 4,001 points changes the reported model-compatible bounds only slightly.

## 4.3 NIH controlled-reader results

For six endpoints with negligible projection error at every reader count—Abnormal, Consolidation, Pleural Thickening, Nodule, Pneumothorax, and Cardiomegaly—the sharp descriptive tail interval contracts substantially as `m` increases.

At `beta=0.5`:

| Endpoint | width m=1 | width m=3 | width m=5 |
|---|---:|---:|---:|
| Abnormal | 0.651 | 0.346 | 0.203 |
| Consolidation | 0.467 | 0.375 | 0.084 |
| Pleural Thickening | 0.236 | 0.0765 | 0.0225 |
| Nodule | 0.339 | 0.180 | 0.101 |
| Pneumothorax | 0.307 | 0.172 | 0.0841 |
| Cardiomegaly | 0.184 | 0.0903 | 0.0507 |

The contraction is monotone over `m=1,...,5` for all six endpoints in the current grid calculation. Pleural thickening shows the strongest relative contraction among these examples: the width at five readers is about 9.5% of the one-reader width. Consolidation decreases to about 18% of its one-reader width, while cardiomegaly decreases to about 28%.

Two endpoints reveal model tension rather than clean contraction. Atelectasis has L1 projection error 0.0016 at `m=4` and 0.0516 at `m=5`; effusion has error 0.0122 at `m=5`. These deviations cannot be attributed to grid resolution because the other endpoints fit to numerical precision under the same grid. They are consistent with endpoint-specific reader heterogeneity or other violations of the iid-expert interpretation. We therefore exclude these endpoints from primary binomial-mixture inference and retain them as negative model-checking examples.

## 4.4 VinDr-CXR external three-reader analysis

VinDr-CXR contains 15,000 training images with exactly three reader-image pairs per image. We deduplicate image-reader-class records to account for multiple bounding boxes of the same finding and form binary reader-level finding indicators. Reader assignment is heterogeneous, so VinDr is used as external evidence for finite-reader ambiguity rather than as the cleanest population-of-experts design.

At `m=3` and `beta=0.5`, the descriptive identified intervals are:

| Endpoint | lower | upper | width |
|---|---:|---:|---:|
| Pleural thickening | 0.0130 | 0.1032 | 0.0902 |
| Lung Opacity | 0.0028 | 0.0695 | 0.0668 |
| Nodule/Mass | 0.0071 | 0.0467 | 0.0396 |
| Cardiomegaly | 0.0783 | 0.1640 | 0.0858 |
| Aortic enlargement | 0.1029 | 0.2097 | 0.1068 |

These ranges remain substantial despite 15,000 cases, illustrating the difference between accumulating more cases and observing more readers per case. The statement is descriptive rather than causal: VinDr differs from NIH in task prevalence, reader assignment, and case mix.

## 4.5 Reproducibility

Current experiment entry point:

`python scripts/run_tail_identification_experiments.py --grid 4001`

Primary outputs:

- `experiments/pilot/results/tail_identification/identified_tail_bounds.csv`
- `experiments/pilot/results/tail_identification/nih_m_contraction_summary_beta05.csv`

The script uses deterministic subset enumeration and linear programming; no random seed is required for the current descriptive analysis.

---

# 5. Discussion — provisional

The empirical results show why repeated expert labels should not be summarized only by consensus. For several NIH findings, moving from one reader to five readers shrinks the range of latent expert-tail values compatible with the observable vote distribution by factors of three to ten. This is an identification effect: the number of cases is held fixed while the number of repeated judgments changes. Conversely, the VinDr analysis shows that a very large number of cases does not by itself eliminate uncertainty about the latent distribution when each case receives only three judgments.

The analysis also exposes the importance of validating the observation model. The same five NIH readers produce count distributions that are essentially compatible with a binomial mixture for several findings but not for atelectasis and effusion. A model that treats named experts as iid draws from a common expert population can fail even in a balanced panel. Future versions of the framework should therefore distinguish three layers: a target population of experts, case-specific latent propensity, and systematic reader effects. The present paper will not claim robustness to arbitrary reader heterogeneity unless that extension is formally developed.

A second limitation is that the current real-data intervals are identification intervals conditional on a projected observable distribution, not honest sampling confidence intervals. They demonstrate the information supplied by additional readers, but they do not yet include finite-`n` uncertainty. Closing that gap is precisely the purpose of the margin-adaptive confidence-set theory under development.

Finally, the local margin condition is a sensitivity assumption, not an empirically verifiable fact from five votes per case. Its role is to encode how much latent probability mass may lie arbitrarily close to the clinically chosen threshold. Final analyses must therefore report results across scientifically interpretable values of `(C,gamma)` and must separate conclusions that hold without a margin condition from those that rely on it.

---

# 6. Conclusion — provisional

Finite expert panels contain two kinds of uncertainty that consensus labels obscure: uncertainty about case-level expert propensity and uncertainty about the population fraction of persistently controversial cases. By targeting a threshold tail of the latent expert-judgment distribution, the proposed framework turns repeated labels into an explicit inference problem rather than a nuisance to be averaged away. Controlled NIH reader subsampling already shows that additional readers can greatly contract the compatible tail range for several diagnostic findings, while model diagnostics reveal endpoints for which an exchangeable-binomial interpretation is inadequate. The remaining theoretical task is to characterize the minimax and honest-confidence limits of this tail functional under local margin information and to translate those limits into principled expert-allocation rules.

---

# Manuscript completion checklist

## Must be proved before any submission claim

- [ ] Identification-diameter upper and lower bound under the chosen margin class.
- [ ] Full off-band Bernstein stability theorem or a replacement route if the conjectured exponential bound is false.
- [ ] Minimax lower bound for the tail functional, not only for linear estimators.
- [ ] Honest finite-sample or asymptotically honest CI construction.
- [ ] Matching CI-length lower bound.
- [ ] Budget-allocation theorem derived from the closed risk bound.

## Must be completed experimentally

- [x] NIH data audit.
- [x] VinDr data audit.
- [x] NIH controlled `m=1,...,5` identified-set experiment.
- [x] VinDr `m=3` external identified-set experiment.
- [x] Observable-model projection diagnostic.
- [ ] Multinomial finite-sample uncertainty layer.
- [ ] Margin sensitivity experiment over prespecified `(C,gamma)` values.
- [ ] Synthetic experiments with known `G` for coverage and minimax-rate verification.
- [ ] Formal model goodness-of-fit test / bootstrap for reader heterogeneity.

## Citation work still required

Insert verified bibliographic entries for Wood (1999), classical Hausdorff/Chebyshev-Markov moment bounds, Basu-Brill-Yekutieli, Loh-Zhang / Zhang / Roueff-Ryden, Lee-Bacak-Kennedy, and repeated-reader design literature. Do not cite the current working notes as evidence.
