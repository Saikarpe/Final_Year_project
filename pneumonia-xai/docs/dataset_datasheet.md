# Dataset datasheet (P-9) -- Kermany paediatric chest X-ray corpus (D-1)

Follows the spirit of Gebru et al.'s "Datasheets for Datasets" -- the
questions that matter for this project, not the full published
questionnaire. Only covers D-1; D-2/D-3/D-4/D-5/D-6 get their own datasheets
if/when they're added (all currently deferred -- see decisions_log.md).

## Motivation

Collected by Kermany et al. (2018, *Cell*) to train and validate a
deep-learning system for diagnosing paediatric pneumonia from chest
radiographs, and released publicly. Widely reused (including here) as a
transfer-learning benchmark, which this project's model card is careful to
flag rather than treat as equivalent to a general adult-population dataset.

## Composition

- 5,856 anterior-posterior chest X-rays, labelled `NORMAL` or `PNEUMONIA`
  (bacterial or viral, collapsed to a single positive class here).
- Source: paediatric patients aged 1-5, Guangzhou Women and Children's
  Medical Center. **Every image in this dataset is paediatric.** Nothing
  trained on it should be assumed to generalize to adult radiographs.
- Class imbalance: roughly 3:1 PNEUMONIA:NORMAL (handled in training via
  `xai_cxr.data.class_weights`, not by discarding examples).
- No demographic metadata (sex, exact age, comorbidities) ships with the
  image files in this repo -- this is why A-2 (subgroup performance) is
  marked `not_available` rather than computed.
- No patient-ID column. Patient/study grouping used for the splits in this
  repo (`configs/splits/*.txt`) is derived from filename patterns -- see
  `docs/decisions_log.md` for exactly how, and its confirmed-vs-best-effort
  reliability per class.

## Collection process

Radiographs were originally graded by two expert physicians before being
cleared for training use, with a third adjudicating disagreements (per the
original publication). This repo did not re-verify any label.

## Preprocessing in this repo

Images are read at their native resolution and resized to 224x224 by
`xai_cxr.data.load_image` / `load_split_dataset`. Model-specific
normalization (`vgg16.preprocess_input`, not a manual `/255` rescale -- see
`xai_cxr.models.baseline`) is applied inside the model graph, not at load
time, so every consumer (training, evaluation, explanation methods, the app)
reads the same raw-pixel representation.

## Known biases / limitations

- Paediatric-only (see Composition). The single largest external-validity
  limitation of any result produced from this dataset.
- Two-centre origin narrows the range of scanner/imaging-protocol variation
  represented -- a known source of shortcut learning in chest X-ray models
  (motivating the shortcut audit, A-1).
- No radiologist-drawn bounding boxes or segmentation masks -- localisation
  metrics (E-6) and a real lung-field ROI for the shortcut audit (A-1) both
  need D-2 (VinDr-CXR) instead.

## Licence / distribution

Redistributed by Kaggle/Mendeley under CC BY 4.0 (see the original Kermany
et al. release). This repo does not commit the dataset to git (see
`.gitignore` and `docs/decisions_log.md`'s G-2 note) -- download it
separately per the README and place it under `dataset/chest_xray/`.
