# Pilot Protocol Amendment V2: Morphological Prediction Bands

Status: frozen **after** the V1 pixel-interval family failed the pre-method set-family adequacy check, and **before** inspecting any V2 conformal results.

## Why this amendment exists

V1 used probability-threshold bands around 0.5. Across the already-trained OOF predictors, calibrated q values were typically 0.48-0.50, making the upper mask include roughly 98-100% of the 512x512 image. Coverage remained computable, but set-efficiency metrics became practically uninterpretable. This is treated as a prediction-set-family failure, not as a Go/No-Go result.

All V1 files/results are retained unchanged for provenance.

## What does NOT change

- Same 55 QUBIQ prostate labeled cases.
- Same fixed 5-fold patient-level splits.
- Same trained deterministic U-Net probability maps; **no retraining**.
- Same Task01 low-disagreement control and Task02 high-disagreement stress test.
- Same four calibration targets: Consensus, Naive Annotation, Random-Rater, All-Rater.
- Same alpha values: 0.10, 0.20, 0.30.
- Same 200 fixed-seed random-rater calibration replicates.
- Same primary OOF case-level evaluation and Task02 sensitivity excluding case07/case50 from evaluation summaries.

## V2 nested set family

Let P(x) = {v : p_v(x) >= 0.5} be the deterministic hard segmentation. The adequacy check confirmed P(x) is non-empty for all 55 OOF cases in both tasks.

For radius r >= 0 (in pixels on the common 512x512 letterboxed analysis grid):

- L_r(x) = Euclidean erosion of P(x) by radius r.
- U_r(x) = Euclidean dilation of P(x) by radius r.
- C_r(x) = {y : L_r(x) subseteq y subseteq U_r(x)}.

As r increases, C_r is nested. At sufficiently large radius the lower mask is empty and the upper mask is the full image, so both empty and non-empty expert masks can be represented.

For a binary target y, the minimal inclusion radius is computed exactly from Euclidean distance transforms:

- removing false-positive core pixels requires r at least their depth inside P;
- including target-positive pixels outside P requires r at least their distance to P;
- the score is the maximum of these two requirements.

This is a deliberately simple morphological conformal baseline and is **not claimed as a methodological novelty**.

## Efficiency metrics

For each calibrated radius r:

- ambiguity area = |U_r \ L_r| pixels;
- ambiguity fraction = ambiguity area / 512^2;
- relative ambiguity = ambiguity area / max(consensus foreground area, 1);
- normalized radius = r / 512.

## Interpretation gate

V2 is considered adequate only if calibrated ambiguity fractions are materially below the near-universal V1 values and vary meaningfully across calibration targets. Only then can the original Go/No-Go hypothesis be interpreted.

The scientific Go signal remains unchanged: Task02 should exhibit a stable larger efficiency cost for all-rater coverage than Task01, associated with independently audited inter-rater disagreement and remaining material after excluding case07/case50 from evaluation summaries.
