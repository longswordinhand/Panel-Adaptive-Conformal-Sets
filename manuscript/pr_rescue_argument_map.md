# Argument Map — Panel-Adaptive Conformal Sets for Multi-Expert Classification

## One-sentence argument

In classification problems with persistent expert disagreement, we show that calibrating prediction sets to a case-level expert-mass target exposes a tail-efficiency failure of global and marginal ambiguity-aware conformal procedures, and that Panel-Adaptive Conformal Sets (PACS) can redistribute set size across case difficulty to reduce upper-tail set inflation while maintaining the desired fraction of expert plausibility mass on most cases, supported by the original dermatology ambiguous-ground-truth benchmark, CIFAR-10H human judgments, and a five-radiologist NIH sensitivity study.

## Reader questions

1. **Why care?** A prediction set can have acceptable average expert coverage yet fail badly on individual ambiguous cases, or satisfy a strong global target by returning extremely large sets for a subset of cases.
2. **What is new?** The paper targets the *fraction of expert plausibility mass captured per case*, learns the minimum model-ranked top-k needed for that target, and calibrates the adaptive requirement across difficulty strata. The novelty is the multi-expert objective and resulting adaptive set construction, not split conformal or Mondrian calibration themselves.
3. **Why trust it?** We compare against voted-label CP, Stutz-style MCCP, and a global panel-mass calibration under repeated held-out splits; the primary benchmark is the exact Stutz dermatology data/predictions, a second categorical benchmark is CIFAR-10H with ~50 human labels per image, and NIH is retained only as a medical sensitivity analysis because its findings are multilabel.
4. **Can I reuse it?** PACS is a post-hoc calibrator requiring only model class scores and panel-derived class plausibilities on a calibration set. It does not require retraining the base classifier.
5. **What does it mean?** Adaptivity matters most when expert-mass requirements are high enough that global calibration develops a heavy tail in set size, but not so high that all methods saturate at the number of classes.

## Paragraph/section jobs

### Introduction
P1 context — multi-expert disagreement is often collapsed to consensus even though downstream prediction sets are used to communicate uncertainty.
P2 gap — ordinary CP and MCCP mainly guarantee marginal/average coverage; acceptable averages do not imply that most cases capture a desired fraction of expert diagnostic mass.
P3 practical failure — global thresholds can meet the target by creating a heavy tail of very large sets, especially at high ambiguity-coverage requirements.
P4 approach — define case-level expert-mass capture and PACS, which predicts required top-k then calibrates within difficulty strata.
P5 evidence — dermatology + CIFAR-10H + NIH sensitivity, with the headline being P90 set-size reduction at matched/improved case-level success.
P6 boundary — no claim of new generic conformal theorem; difficulty/Mondrian calibration and ambiguous-ground-truth CP are established prior art.

### Related work
RW1 standard/adaptive conformal prediction and efficiency metrics.
RW2 ambiguous ground truth: Stutz MCCP and conformalized credal/plausibility regions.
RW3 difficulty-stratified/Mondrian adaptive CP, explicitly crediting Jang & Lee 2026.
RW4 multi-annotator validation showing weak correspondence between ordinary conformal set size and human disagreement.
RW5 position: PACS differs by optimizing a per-case expert-mass requirement and its tail-size behavior rather than generic marginal coverage or generic difficulty adaptivity.

### Method
M1 notation: base scores s(x,y), expert plausibility lambda_x(y), prediction set C(x).
M2 target: M_x(C)=sum lambda_x(y)1[y in C], success M_x(C)>=q.
M3 oracle training target: K_x(q), minimum top-k in model-score order needed to reach q expert mass.
M4 predictor: quantile regression for log(1+K_x(q)) using model-output features only.
M5 difficulty strata: define strata from predicted K using training data only.
M6 calibration: untouched calibration cases provide one-sided residual quantiles per stratum; test k is ceil(predicted K + correction).
M7 validity boundary: this is standard group-conditional split-conformal validity for the observed derived target under exchangeability; empirical finite expert panels are not claimed to identify a latent expert population.
M8 optional safety variants PTCP/PanelCert are conservative references, not default predictors.

### Experiments
E1 datasets and why each is included.
E2 baselines: top-1/voted CP, MCCP-10, global panel-mass threshold, PACS.
E3 protocol: repeated disjoint train/cal/test splits; base classifier independence from panel labels where relevant.
E4 metrics: case-level q-mass success, mean set size, P90/P95 set size, mean captured mass, minority-opinion mass.
E5 dermatology main q=0.9 result.
E6 CIFAR-10H q sensitivity (pending final numbers).
E7 NIH q sensitivity and saturation as external medical stress test.
E8 ablations/boundaries: q=0.7/0.8 dermatology non-improvement; q=0.9 NIH saturation; optional nested hyperparameter selection; number of difficulty bins.

### Discussion
D1 main finding: tail, not mean, is where adaptive benefit appears.
D2 why: one global threshold transfers the hardest-case calibration burden to easier cases; strata localize this burden.
D3 regime dependence: too-low q gives little need for adaptation; too-high q saturates class support.
D4 finite-panel caveat: lambda is empirical/aggregated expert evidence, not an identified latent posterior.
D5 dataset caveat: NIH is multilabel and therefore secondary.
D6 practical implication: PACS is a post-hoc calibration layer for systems that already collect repeated expert labels.

## Forbidden narrative moves

- Do not describe PACS as the first adaptive conformal method.
- Do not describe difficulty grouping/Mondrian calibration as novel.
- Do not claim universal mean-size improvement.
- Do not claim latent expert-population guarantees from finite panels.
- Do not use the old TPAMI minimax/partial-identification theory as the headline contribution.
