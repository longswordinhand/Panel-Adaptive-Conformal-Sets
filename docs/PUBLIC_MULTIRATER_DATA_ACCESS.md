# Public Multi-Rater Data Access Plan

Date: 2026-08-15

Purpose: public real-data validation for finite-rater latent expert-risk theory. Priority is individual-reader categorical/diagnostic labels, not segmentation disagreement.

## 1. VinDr-CXR — primary large-n / small-m dataset

Official source: PhysioNet VinDr-CXR v1.0.0 / latest DOI.

What we need:
- 15,000 training studies.
- Each training image independently labeled by 3 radiologists.
- `image_labels_train.csv` with `image_id`, radiologist ID, and image-level finding/diagnosis labels.
- DICOM train images corresponding to those labels.

Access status:
- Restricted-access PhysioNet resource.
- Requires the user to be a credentialed PhysioNet user.
- Requires CITI Data or Specimens Only Research training.
- Requires signing the PhysioNet credentialed data use agreement.
- These steps must be completed by the user; they cannot be accepted by an automated agent.

When access is granted, place original files under:
`data/public/vindr_cxr/raw/`

Do not alter raw files. Record checksums after transfer.

## 2. NIH Chest X-ray expert labels — secondary public expert-reader dataset

Official source: Google Cloud Healthcare API NIH Chest X-ray public dataset documentation.

What we need first:
- `four_findings_expert_labels/individual_readers.csv`
- Preferably also `all_findings_expert_labels/test_individual_readers.csv`.
- Corresponding chest radiographs for image-level modeling after label audit.

Documented label structure:
- Four-findings subset: 4,374 images with adjudicated labels; individual-reader file contains one row per reader per image. Nodule/mass and pneumothorax allow `YES`, `NO`, `HEDGE`; opacity/fracture use `YES`, `NO`.
- All-findings subset: 810 PA images, five board-certified radiologists per image, 4,050 individual-reader rows, 14 findings plus normal/abnormal and `Other`.

Access status:
- Expert labels require completing Google's access form before download.
- NIH image access through the Cloud Healthcare API also requires an access request form.
- These one-time user actions cannot be completed by an automated agent.

When obtained, place original files under:
`data/public/nih_cxr_expert/raw/`

## 3. MCR-SL — ungated public dataset

Official source: Zenodo record 17306338, DOI 10.5281/zenodo.17306338.
License: CC-BY.
Archive size: approximately 4.4 GB.

Dataset content relevant to this project:
- 240 lesions from 60 subjects.
- Four dermatologists independently diagnosed every lesion.
- Dataset includes individual dermatologist diagnoses; 29 excised lesions additionally have histopathology.
- Diagnostic categories include NEV, SK, BCC, AK, ATY, MEL, SCC, ANG, DF, UNK.

Target location after download:
`data/public/mcr_sl/raw/`

Because the archive is large, do not download silently. Download only after explicit approval in the conversation.

## Planned audit after files arrive

For every dataset:
1. Verify archive/file checksums and preserve raw files unchanged.
2. Identify true experimental unit (patient/study/lesion) and repeated-reader nesting.
3. Count unique cases, readers per case, stable reader IDs, missing labels, and label vocabulary.
4. Quantify disagreement without collapsing to consensus:
   - exact agreement,
   - vote entropy,
   - minority-label prevalence,
   - presence of clinically meaningful alternative labels,
   - per-reader prevalence/bias.
5. Check whether any patient contributes multiple images/cases and enforce patient-level splits where needed.
6. Build a normalized manifest with one row per case x reader x target.
7. Only after the data audit, decide which target(s) are valid for conformal experiments.

## Current decision

Primary real-data plan:
1. VinDr-CXR for large-n, m=3.
2. NIH expert labels, especially the 810-image five-reader subset, for a stronger within-case reader-count experiment.
3. MCR-SL as an ungated cross-domain expert-diagnosis dataset.
4. ReiScan later as independent private institutional validation, not as the sole real-data evidence.
