"""CLI: train the M-1 classifier (M-1, M-1b).

Two stages, both driven by configs/model.yaml:

  1. Head warmup -- backbone frozen, high LR, warmup-cosine schedule. This is
     what the previous version did, and all it did.
  2. Fine-tune   -- the top `unfreeze_fraction` of the backbone unfrozen
     (BatchNormalization kept frozen), LR two orders of magnitude lower.

Stage 2 is the substantive change. A permanently-frozen ImageNet backbone
never adapts to radiographs: it is matching natural-image texture statistics
and the head is doing all the work, which is why the old model's Grad-CAM
maps were diffuse. `finetune_epochs: 0` reproduces the old single-stage
behaviour if you want the comparison.

Usage: python scripts/train.py
"""
import dataclasses
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from xai_cxr.config import DataConfig, ModelConfig, MODELS_DIR
from xai_cxr.data import load_split_dataset, class_weights, read_manifest
from xai_cxr.models.baseline import (
    build_model, describe, finetune, tune_threshold, tune_threshold_at_sensitivity,
    save_threshold, WarmupCosine, build_optimizer, build_loss, build_metrics,
)
from xai_cxr import tracking

MONITOR = 'val_auc'


def _callbacks(model_cfg, patience, best_so_far=None):
    """Fresh callbacks for a stage.

    `best_so_far` carries stage 1's best val_auc into stage 2's
    ModelCheckpoint. Without it the checkpoint's internal best resets to -inf
    between stages, so stage 2's *first* epoch overwrites the file no matter
    how it compares -- and if fine-tuning degraded the model, the better
    stage-1 checkpoint would be silently replaced by a worse one.
    EarlyStopping's restore_best_weights does not cover this: it only tracks
    the stage it is running in.
    """
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_cfg.model_path, save_best_only=True, monitor=MONITOR, mode='max',
        verbose=1, initial_value_threshold=best_so_far)
    return [
        checkpoint,
        tf.keras.callbacks.EarlyStopping(patience=patience, monitor=MONITOR, mode='max',
                                         restore_best_weights=True, verbose=1),
    ]


def _merge_history(*histories):
    """Concatenate the per-stage Keras histories into one series per metric,
    so training_history.png shows the whole run rather than only stage 2."""
    merged = {}
    for h in histories:
        if h is None:
            continue
        for key, values in h.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


if __name__ == '__main__':
    data_cfg = DataConfig.load()
    model_cfg = ModelConfig.load()

    train_ds, train_labels = load_split_dataset('train', data_cfg, shuffle=True, augment=True)
    val_ds, val_labels = load_split_dataset('val', data_cfg)

    weights = class_weights(train_labels)
    print(f'class_weight = {weights}  (train: {int(train_labels.sum())} PNEUMONIA / '
          f'{len(train_labels) - int(train_labels.sum())} NORMAL)')

    steps_per_epoch = max(1, math.ceil(len(train_labels) / data_cfg.batch_size))

    # Stage 1 gets its own warmup-cosine schedule over its own epoch budget;
    # stage 2 gets a separate one inside finetune(). A single schedule across
    # both stages would decay the LR to ~0 exactly when the backbone unfreezes.
    stage1_lr = WarmupCosine(
        base_lr=model_cfg.learning_rate,
        total_steps=steps_per_epoch * max(model_cfg.epochs, 1),
        warmup_steps=int(steps_per_epoch * model_cfg.warmup_epochs),
        final_lr=model_cfg.learning_rate * 0.05,
    )
    model = build_model(model_cfg, learning_rate=stage1_lr)

    arch = describe(model)
    print('\n' + json.dumps(arch, indent=2, default=str))
    print(f'CAM grid at {data_cfg.img_size[0]}x{data_cfg.img_size[1]}: '
          f'{arch["feature_map"][0]}x{arch["feature_map"][1]}\n')

    run_dir = tracking.start_run('train_baseline', {
        'data': dataclasses.asdict(data_cfg) if dataclasses.is_dataclass(data_cfg) else vars(data_cfg),
        'model': {**vars(model_cfg), 'architecture': arch},
    })

    print('=' * 68)
    print(f'Stage 1/2 -- head warmup ({model_cfg.epochs} epochs, backbone frozen)')
    print('=' * 68)
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=model_cfg.epochs,
                         class_weight=weights, callbacks=_callbacks(model_cfg, model_cfg.epochs),
                         verbose=1)

    history2 = None
    if model_cfg.finetune_epochs > 0:
        n_trainable = finetune(model, model_cfg, steps_per_epoch=steps_per_epoch,
                               epochs=model_cfg.finetune_epochs)
        print('\n' + '=' * 68)
        print(f'Stage 2/2 -- fine-tune ({model_cfg.finetune_epochs} epochs, '
              f'{n_trainable} backbone layers trainable, BN frozen='
              f'{model_cfg.freeze_batchnorm})')
        print('=' * 68)
        # Hand stage 1's best val_auc to stage 2's checkpoint so a worse
        # fine-tuned model can never overwrite a better warmed-up one.
        best_stage1 = max(history1.history[MONITOR]) if history1 is not None else None
        print(f'stage 1 best {MONITOR} = {best_stage1:.5f} (stage 2 must beat this to overwrite '
              'the checkpoint)')
        history2 = model.fit(train_ds, validation_data=val_ds, epochs=model_cfg.finetune_epochs,
                             class_weight=weights,
                             callbacks=_callbacks(model_cfg, model_cfg.early_stopping_patience,
                                                  best_so_far=best_stage1),
                             verbose=1)
    else:
        print('\nfinetune_epochs: 0 -- skipping stage 2 (frozen-backbone model).')

    # Reload the best checkpoint so the threshold below matches what is
    # actually on disk. EarlyStopping's restore_best_weights only covers the
    # stage it ran in; the checkpoint file spans both.
    from xai_cxr.models.baseline import load_trained
    model = load_trained(model_cfg.model_path)

    val_scores = model.predict(val_ds, verbose=0).reshape(-1)
    if model_cfg.threshold_policy == 'min_sensitivity':
        threshold = tune_threshold_at_sensitivity(val_labels, val_scores, model_cfg.min_sensitivity)
        policy_desc = f'lowest FPR at sensitivity >= {model_cfg.min_sensitivity:.2f}'
    else:
        threshold = tune_threshold(val_labels, val_scores)
        policy_desc = "Youden's J"
    save_threshold(threshold)
    print(f'\nTuned decision threshold ({policy_desc}, on val): {threshold:.4f}')

    history = _merge_history(history1, history2)
    boundary = len(history1.history['loss'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in ((ax1, 'auc', 'AUC'), (ax2, 'loss', 'Loss')):
        ax.plot(history[key], label='Train')
        ax.plot(history[f'val_{key}'], label='Val')
        if history2 is not None:
            ax.axvline(boundary - 0.5, color='#888', ls='--', lw=1, label='fine-tune starts')
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.legend()
    fig.suptitle(f'{arch["backbone_description"]} -- two-stage training')
    fig.tight_layout()
    plot_path = os.path.join(MODELS_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=120)

    tracking.log_metrics(run_dir, {
        'backbone': model_cfg.backbone,
        'best_val_auc': float(max(history['val_auc'])),
        'best_val_loss': float(min(history['val_loss'])),
        'best_val_auprc': float(max(history.get('val_auprc', [float('nan')]))),
        'decision_threshold': threshold,
        'threshold_policy': model_cfg.threshold_policy,
        'epochs_run_stage1': boundary,
        'epochs_run_stage2': len(history['loss']) - boundary,
        'total_params': arch['total_params'],
    })

    print(f'\nModel saved  -> {model_cfg.model_path}')
    print(f'Plot saved   -> {plot_path}')
    print(f'Run logged   -> {run_dir}')
