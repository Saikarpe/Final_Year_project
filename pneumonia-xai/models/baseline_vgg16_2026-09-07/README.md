# Archived M-1 baseline (frozen VGG16) — evaluated 2026-09-07

Preserved before the M-1b re-train overwrote them. These are the **only**
numbers this repo has for the pre-M-1b model, and they are not being
regenerated (no VGG16 re-run — time constraint).

Contents: `metrics.json`, `training_history.png`, `explanations_grid.png`,
plus `docs/baseline_vgg16_2026-09-07/` for the failure gallery and its images.

## Reading these against the M-1b numbers

**The data is the same. This is a controlled comparison on the test set.**

Verified 2026-09-20, not assumed. The M-1b run re-downloaded the corpus from
Mendeley (the Kermany authors' own publication data) rather than the Kaggle
redistribution these numbers came from, and `build_splits.py` was re-run. The
resulting manifests were then compared image-by-image against the committed
ones:

    split         images  same_files  same_paths  moved
    train           4120  True        False       11
    val              591  True        False       2
    calibration      563  True        False       2
    test             582  True        False       1

Every split holds **exactly the same images**. Only 16 path strings differ,
because the Kaggle copy ships a 16-image `val/` folder that the Mendeley
original distributes inside `train/`/`test/` -- so those images carry a
different source prefix while being the same files. `build_splits.py` groups
by patient id derived from the filename and splits with a fixed seed, so
identical patients produce identical splits.

So: **same 582 test images, same patients, patient-disjointness re-verified.**
An AUROC difference between this baseline and M-1b is a real difference, not
a resampling artefact.

## What still differs (so do not over-claim)

The *model and recipe* changed, which is the point, but note these moved
together rather than one at a time:

- backbone VGG16 -> DenseNet121, and the backbone is now fine-tuned rather
  than frozen forever;
- head: GAP -> attention pooling, plus the learnable window filter and
  spatial dropout;
- input resolution 224 -> 320, so the CAM grid is 10x10 here vs 14x14 for
  VGG16's pre-pool `block5_conv3`;
- augmentation, optimiser (Adam -> AdamW), LR schedule, label smoothing;
- batch size 32 -> 8 (a 4 GB VRAM constraint, not a modelling choice).

This is an A/B between two complete configurations, not an ablation. It
cannot attribute a gain to any single change. An ablation would need one
run per change against these same manifests.

Headline figures here: AUROC 0.994 (95% CI 0.989-0.998), sensitivity 95.6%,
specificity 96.7%, Brier 0.047, decision threshold 0.217, n_test 582.

To make it a controlled comparison later, set the VGG16-reproduction block in
the README, re-run `build_splits.py` once and train both configurations
against the *same* manifests.
