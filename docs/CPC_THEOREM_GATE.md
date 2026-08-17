# CPC Theorem Gate

## Candidate target

For case i with m_i expert labels Y_{i1},...,Y_{im_i}, prediction set C_t(X_i)={y:s(X_i,y)<=t}, and target panel fraction q in (0,1], define

R_i(t) = (1/m_i) sum_{h=1}^{m_i} 1{s(X_i,Y_{ih}) <= t}.

Desired guarantee:

P( R_{n+1}(T) >= q ) >= 1-alpha.

## Exact score definition

Let

r_i = ceil(q m_i).

Sort the expert nonconformity scores

s_{i,(1)} <= ... <= s_{i,(m_i)}.

Define the case-panel score

A_i^(q) = s_{i,(r_i)}.

Then the events are exactly equivalent:

A_i^(q) <= t  <=>  R_i(t) >= r_i/m_i >= q.

This removes ambiguity associated with generic software quantile conventions.

## Split-conformal calibration

Given n calibration case-panels, let

k = ceil((n+1)(1-alpha)).

If k <= n, let T be the k-th smallest value among A_1^(q),...,A_n^(q). If k=n+1, use T=+infinity (the usual conservative finite-sample convention).

Assume the n calibration case-panels and the new case-panel are exchangeable as whole observational units. Then the rank of A_{n+1}^(q) among the n+1 scores is exchangeable, giving

P(A_{n+1}^(q) <= T) >= 1-alpha.

By the equivalence above,

P(R_{n+1}(T) >= q) >= 1-alpha.

Thus CPC is a valid finite-sample guarantee for the observed/future panel target.

## Variable panel sizes

The proof still works when m_i varies only when the complete observational units (X_i, m_i, Y_{i1:m_i}) are exchangeable and A_i^(q) is computed by the same deterministic rule r_i=ceil(q m_i).

If panel size or panel composition changes systematically between calibration and deployment, ordinary exchangeability does not justify the guarantee. NIH all-findings (fixed m=5 panel) is especially clean; VinDr (mostly/exactly m=3 in the audited train CSV) is also usable for a fixed-m experiment but reader-composition heterogeneity must be disclosed.

## Relationship to existing theory

This theorem is mathematically a standard split-conformal argument applied to a derived case-panel order-statistic score. It should NOT be claimed as a new conformal theorem in isolation.

It is also closely related to Conformal Risk Control (Angelopoulos et al.), which can control expected monotone losses and explicitly treats multilabel false-negative rate / fraction of missed true classes. CPC differs in the target quantity: CPC controls the probability of a case-level panel-recall failure event, whereas standard CRC examples control mean missed-label fraction. However, this distinction is an application/problem-definition distinction unless accompanied by additional methodological novelty.

## Non-equivalence with random-rater marginal coverage

Let R in [0,1] denote per-case panel recall. Random-rater marginal coverage controls E[R]. CPC controls P(R>=q).

These are not equivalent.

Example: take q=0.95 and let R=0.90 almost surely. Then random-rater marginal coverage is 0.90, but P(R>=0.95)=0.

More generally, knowing only E[R]>=mu does not imply P(R>=q)>=mu. For q<mu, the sharp generic lower bound from boundedness is

P(R>=q) >= (mu-q)/(1-q)

(up to endpoint conventions), which can be far below mu. Thus case-level panel guarantees address a concentrated-failure mode that marginal expert-draw coverage can hide.

## Current verdict

- Correctness: PASS.
- Standalone theorem novelty: NO-GO.
- Potential Pattern Recognition contribution: ALIVE only as part of a stronger panel-aware method/benchmark package.

## Next required novelty gate

Before implementation, search explicitly for methods that already control any of:

1. probability that per-instance recall exceeds q;
2. quantile / VaR of per-instance false-negative rate;
3. conformal control of lower-tail sample-wise recall;
4. panel-level repeated-label order-statistic conformal scores;
5. human-annotator fraction coverage under conformal prediction.

If an existing method directly provides the same target and construction, CPC/PCP should not be presented as the primary algorithmic contribution.
