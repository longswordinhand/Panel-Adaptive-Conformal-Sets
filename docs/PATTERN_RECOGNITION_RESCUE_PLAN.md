# Pattern Recognition Rescue Plan

## Status

Target journal: Pattern Recognition (not TPAMI).

Current decision: salvageable only if the paper is rebuilt around a new **case-level panel-coverage** guarantee and a concrete algorithm. The previous latent-tail / moment-identification theory is retained only as background motivation, not as the principal novelty claim.

## Prior-art kill test completed on 2026-08-15

### Directly covered by prior work — DO NOT claim as novel

1. **Conformal prediction under ambiguous / multi-expert ground truth**
   - Stutz et al., TMLR 2023, *Conformal Prediction under Ambiguous Ground Truth*.
   - Already shows majority-vote CP can under-cover expert annotations.
   - Already proposes Monte Carlo CP by sampling pseudo-labels from an aggregated expert label distribution and gives theoretical coverage with respect to that aggregate distribution.
   - Also extends to multi-label settings.

2. **Conformal prediction with set-valued / partial labels**
   - Javanmardi et al., COPA 2023, *Conformal Prediction with Partially Labeled Data*.
   - Gong et al., AAAI 2025, *Conformal Prediction for Partial Label Learning*.

3. **Prediction-set similarity to human disagreement is weak**
   - Hagos & Lundström, WACV 2026, *Performance of Conformal Prediction in Capturing Aleatoric Uncertainty*.
   - Shows ordinary CP set size is generally weakly correlated with human annotator disagreement.

4. **Medical multi-rater calibration / ambiguity modeling**
   - Large existing literature in multi-rater segmentation and calibration (MRNet, Multi-rater Prism, MRNet+, TwinTrack, etc.).
   - Therefore the paper must stay in diagnostic/classification disagreement and cannot claim generic multi-rater calibration as new.

### Candidate gap still alive after first kill test

Existing ambiguous-ground-truth CP primarily targets a **random expert label** or an aggregate label distribution. That guarantee is not the same as requiring that, on a new case, the prediction set cover a specified fraction of the observed expert panel.

We therefore define a new target.

## Core target: Case-level Panel Coverage (CPC)

For case i with m_i expert labels Y_{i1},...,Y_{im_i}, and a prediction set C(X_i), define panel recall

R_i(C) = (1/m_i) sum_{h=1}^{m_i} 1{Y_{ih} in C(X_i)}.

For user-selected q in (0,1], the desired guarantee is

P( R_{new}(C) >= q ) >= 1 - alpha.

Interpretation:

> With probability at least 1-alpha over a new case (and its expert panel), the model's prediction set contains diagnoses endorsed by at least a q fraction of that case's experts.

This differs from marginal random-rater coverage

P_{X,H}(Y_H in C(X)) >= 1-alpha,

because E[R_i] coverage can be high while a non-negligible fraction of cases have very poor within-case expert coverage.

### Special cases

- q = 1/m: at least one panel vote represented.
- q = 1/2: at least half of expert votes represented.
- q = 1: all observed expert votes represented.

For binary/multiclass labels, repeated labels are counted by vote frequency; this is intentionally different from merely covering the set of distinct labels.

## Minimal conformal construction to test

Assume a trained classifier provides class scores s(x,y), lower is better.

For calibration case i, define the q-panel nonconformity score as the q-quantile of the expert-specific scores:

A_i^{(q)} = Quantile_q( { s(X_i,Y_{ih}) : h=1,...,m_i } ).

Calibrate a split-conformal threshold

T = Quantile_{ceil((n_cal+1)(1-alpha))/n_cal}( A_1^{(q)},...,A_{n_cal}^{(q)} ).

Return

C_q(x) = { y : s(x,y) <= T }.

Under exchangeability of **cases-with-panels** (not individual reader labels), the event A_new^{(q)} <= T implies that at least a q fraction of the new panel's expert labels have score <= T, hence

P( R_new(C_q) >= q ) >= 1-alpha.

This is the first theorem to formalize and verify carefully.

### Why this is potentially useful

Random-rater marginal coverage can hide concentrated failures. For example, if 90% of cases have 100% panel coverage and 10% have 0%, then average random-rater coverage is 90%, but 10% of cases are completely unrepresented. CPC directly controls the tail event of poor panel representation.

## Candidate contribution package for Pattern Recognition

### Contribution 1 — New validity target

Define and distinguish:

1. consensus-label coverage;
2. random-rater marginal coverage;
3. distinct-label-set coverage;
4. **case-level q-panel coverage**.

Provide counterexamples showing (2) does not imply (4).

### Contribution 2 — Panel-Conformal Prediction (PCP)

A model-agnostic split-conformal wrapper using per-case order statistics / quantiles of expert-specific nonconformity scores.

The primary theorem should be finite-sample and distribution-free at the case-panel level.

### Contribution 3 — Efficiency under disagreement

Study the q-alpha efficiency frontier:

- set size versus q;
- set size versus alpha;
- minority-diagnosis retention;
- panel-recall distribution across cases;
- disagreement-stratified efficiency.

Optional extension only if genuinely novel after audit: adaptive q(x) or weighted-expert CPC. Do not include unless a second novelty audit clears it.

### Contribution 4 — Real multi-expert medical validation

Primary datasets already available:

- NIH all-findings expert labels: 810 cases, fixed complete 5-reader panel. This is the cleanest CPC benchmark.
- VinDr-CXR: 15,000 cases, 3 readers/case; use disease-level labels, not normal-vs-abnormal endpoint.

No new dataset should be downloaded until the base CPC theorem and a label-only numerical sanity test are complete.

## Critical distinction from Stutz et al. 2023

Stutz target (schematically):

P_{X,Y~P_agg(.|X)}(Y in C(X)) >= 1-alpha.

CPC target:

P_X,panel( (1/m) sum_h 1{Y_h in C(X)} >= q ) >= 1-alpha.

The former controls an expectation over an expert draw. The latter controls a **case-level lower tail of panel representation**.

This distinction must survive a second targeted literature audit before being promoted to a novelty claim.

## Immediate Go/No-Go gates

### Gate A — theorem correctness

Prove the finite-sample CPC guarantee under exchangeable case-panels. Check discrete quantile conventions for finite-sample coverage.

### Gate B — non-equivalence example

Construct distributions where random-rater coverage is >= 1-alpha but CPC at a clinically meaningful q is arbitrarily poor.

### Gate C — novelty audit

Search explicitly for:

- conformal panel coverage;
- conformal fraction-of-annotators coverage;
- conformal quantile across repeated labels;
- repeated-measure conformal order-statistic targets;
- conformal risk control for within-group coverage fractions;
- set-valued response recall guarantees.

If an existing paper already proves essentially the same target and algorithm, STOP this rescue route immediately.

### Gate D — label-only real-data sanity test

Before training any image model, use the existing NIH/VinDr votes and synthetic class scores / oracle-style score constructions to verify the metric behaves nontrivially.

### Gate E — model experiment

Only after Gates A-C pass, train or reuse image classifiers and compare:

- majority-vote split CP;
- random-rater / pooled-rater CP;
- Stutz-style Monte Carlo CP;
- partial-label/set baselines where applicable;
- proposed PCP/CPC.

## Primary evaluation metrics

1. q-panel coverage probability: P(R_i >= q).
2. Mean panel recall E[R_i].
3. Worst disagreement-stratum panel coverage.
4. Mean prediction-set size.
5. Singleton rate.
6. Minority-diagnosis retention.
7. Consensus-label accuracy/coverage as a secondary metric only.

## Claims explicitly prohibited unless later proven

- "first conformal method for multi-rater labels" — false.
- "first method to model expert disagreement" — false.
- "ordinary CP cannot handle ambiguous labels" — too strong; Stutz et al. already addresses this.
- "finite-rater non-identifiability is novel" — false as a general statistical statement.
- "CPC is novel" — not yet established; currently a candidate gap only.

## Current decision

The article is not dead for Pattern Recognition, but it requires a **new problem definition + algorithm + finite-sample guarantee**, not a downgraded TPAMI submission.

The only route currently authorized for further work is the CPC/PCP route above, subject to the immediate theorem and novelty gates.
