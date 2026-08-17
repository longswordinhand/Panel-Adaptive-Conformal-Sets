# Paired split-level statistical analysis for PACS vs global panel calibration

## Analysis unit

- Differences are always PACS minus Global.
- Dermatology: the four released base-model matrices are averaged within each repeated split before inference; n=12 split units at q=0.9 and n=6 at q=0.7/0.8.
- NIH: n=30 patient-level repeated split units at each q.
- CIFAR-10H: n=20 repeated split units at q=0.7.
- 95% intervals are percentile bootstrap intervals for the mean paired split-level difference (20,000 resamples, fixed seed).
- Paired t and Wilcoxon p-values are saved as sensitivity summaries only. Because repeated splits reuse observations, the manuscript should emphasize paired effect sizes and resampling intervals rather than treat these p-values as independent-sample population inference.

## Headline paired effects

### Dermatology, q=0.9 (n=12 split units)
- Success delta: -0.08 [-0.43, 0.26] percentage points; global=0.9048, PACS=0.9040.
- Mean-size delta: 1.098 [-0.241, 2.608] classes (+2.05% relative).
- P90-size delta: -13.750 [-17.233, -9.958] classes (-13.91% relative).

### NIH, q=0.7 (n=30 split units)
- Success delta: 2.18 [1.43, 2.96] percentage points; global=0.9104, PACS=0.9323.
- Mean-size delta: 0.030 [-0.185, 0.239] classes (+0.24% relative).
- P90-size delta: -1.743 [-2.097, -1.400] classes (-11.47% relative).

### NIH, q=0.8 (n=30 split units)
- Success delta: 2.98 [1.89, 4.04] percentage points; global=0.9134, PACS=0.9431.
- Mean-size delta: 0.037 [-0.128, 0.198] classes (+0.27% relative).
- P90-size delta: -0.967 [-1.200, -0.767] classes (-6.04% relative).

### NIH, q=0.9 (n=30 split units)
- Success delta: 5.14 [4.18, 6.05] percentage points; global=0.9112, PACS=0.9626.
- Mean-size delta: 0.422 [0.299, 0.546] classes (+2.89% relative).
- P90-size delta: -0.067 [-0.167, 0.000] classes (-0.42% relative).

### CIFAR-10H, q=0.7 (n=20 split units)
- Success delta: 7.18 [6.89, 7.49] percentage points; global=0.8990, PACS=0.9709.
- Mean-size delta: 0.690 [0.658, 0.722] classes (+63.43% relative).
- P90-size delta: 1.900 [1.700, 2.100] classes (+165.22% relative).

### Dermatology, q=0.8 (n=6 split units)
- Success delta: 0.71 [0.26, 1.11] percentage points; global=0.9020, PACS=0.9091.
- Mean-size delta: 3.573 [2.718, 4.193] classes (+13.36% relative).
- P90-size delta: 0.750 [-2.250, 2.383] classes (+1.56% relative).

### Dermatology, q=0.7 (n=6 split units)
- Success delta: 1.52 [1.33, 1.72] percentage points; global=0.8915, PACS=0.9067.
- Mean-size delta: 4.229 [3.800, 4.602] classes (+25.74% relative).
- P90-size delta: 5.375 [3.700, 6.975] classes (+18.68% relative).

