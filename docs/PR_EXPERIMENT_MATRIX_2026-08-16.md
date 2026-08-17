# PR Rescue Experiment Matrix — Active / Finalizing

## Primary benchmark A: Dermatology ambiguous ground truth (Stutz et al.)
- 1,947 cases.
- 419 mutually exclusive skin-condition classes.
- Expert partial rankings aggregated with the official IRN procedure.
- Four released model-prediction matrices; no retraining required.
- Main target: q=0.90 expert plausibility mass, alpha=0.10 case-level calibration.
- Fixed PACS configuration: model quantile 0.90, 3 difficulty bins, outer calibration untouched.
- Repeated evaluation: 12 splits x 4 released models = 48 runs.
- Main result: global success 0.9048, mean size 53.54, P90 98.82; PACS success 0.9040, mean size 54.64, P90 85.07.
- Interpretation: ~13.9% P90 reduction at essentially unchanged success, with ~2.1% higher mean size.
- Boundary: q=0.7/0.8 do not show efficiency gains and must be reported as a limitation/sensitivity result.

## Primary benchmark B: CIFAR-10H (in progress)
- 10,000 CIFAR-10 test images.
- 10 mutually exclusive classes.
- ~50 human judgments per image; released 10000 x 10 count/probability arrays.
- Base classifier trained only on original CIFAR-10 50k training images, never on CIFAR-10H soft labels.
- Planned evaluation: q in {0.7,0.8,0.9}, alpha=0.10, repeated 40/30/30 PACS-train/cal/test splits.
- Baselines: top-1 CP, MCCP-10, global expert-mass threshold.
- Purpose: clean categorical multi-annotator replication with much denser human panels than medical data.

## External medical sensitivity: NIH five-radiologist panel
- Complete 810/810 expert-labeled images now available; 810 unique image hashes.
- Same fixed panel of five radiologists for every image.
- Image features: ImageNet ResNet50 frozen features, 810 x 2048, no missing values.
- Caveat: original annotations are multilabel chest findings, so the 16-way normalized finding-mass construction is a stress test rather than a canonical categorical outcome.
- Fixed PACS, 30 patient-level repeated splits.
- q=0.70: global success 0.9104 / P90 15.20; PACS 0.9323 / 13.46.
- q=0.80: global success 0.9134 / P90 16.00; PACS 0.9431 / 15.03.
- q=0.90: global success 0.9112 / P90 16.00; PACS 0.9626 / 15.93; near ceiling, little efficiency room.
- Role in paper: external medical sensitivity only, not the sole proof of method validity.

## Safety/theory baselines
- PTCP: exact finite-panel tolerance correction. Truth-known synthetic experiments verify target protection but show severe conservatism for small m.
- PanelCert: population partial-identification certification layer. Also conservative; retained as a safety/reference method rather than default predictor.
- These methods demonstrate why a practically useful algorithm cannot simply demand worst-case latent-panel guarantees with m=5.

## Main paper claim under evaluation
PACS targets case-level expert-mass capture and redistributes prediction-set size across difficulty strata. The principal efficiency benefit is reduction of upper-tail set inflation, not universal reduction of mean set size. This claim is considered publishable only if it reproduces on CIFAR-10H in addition to the dermatology benchmark; NIH is secondary because of its multilabel structure.
