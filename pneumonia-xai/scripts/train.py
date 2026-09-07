"""CLI: train the M-1 baseline with every fix applied (M-1).

Usage: python scripts/train.py
"""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from xai_cxr.config import DataConfig, ModelConfig, MODELS_DIR
from xai_cxr.data import load_split_dataset, class_weights
from xai_cxr.models.baseline import build_model, tune_threshold, save_threshold
from xai_cxr import tracking

if __name__ == '__main__':
    data_cfg = DataConfig.load()
    model_cfg = ModelConfig.load()

    train_ds, train_labels = load_split_dataset('train', data_cfg, shuffle=True, augment=True)
    val_ds, val_labels = load_split_dataset('val', data_cfg)

    weights = class_weights(train_labels)
    print(f'class_weight = {weights}  (train: {int(train_labels.sum())} PNEUMONIA / '
          f'{len(train_labels) - int(train_labels.sum())} NORMAL)')

    model = build_model(model_cfg)
    model.summary()

    run_dir = tracking.start_run('train_baseline', {
        'data': dataclasses.asdict(data_cfg) if dataclasses.is_dataclass(data_cfg) else vars(data_cfg),
        'model': vars(model_cfg),
    })

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_cfg.model_path, save_best_only=True,
                                            monitor='val_auc', mode='max', verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=model_cfg.early_stopping_patience,
                                          monitor='val_auc', mode='max',
                                          restore_best_weights=True, verbose=1),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=model_cfg.epochs,
                         class_weight=weights, callbacks=callbacks)

    # Load best checkpoint (ModelCheckpoint already wrote it) so the threshold
    # tuning below matches what's actually saved to disk.
    model = tf.keras.models.load_model(model_cfg.model_path)

    val_scores = model.predict(val_ds, verbose=0).reshape(-1)
    threshold = tune_threshold(val_labels, val_scores)
    save_threshold(threshold)
    print(f'\nTuned decision threshold (Youden J on val): {threshold:.4f}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['auc'], label='Train')
    ax1.plot(history.history['val_auc'], label='Val')
    ax1.set_title('AUC'); ax1.legend()
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.legend()
    plot_path = os.path.join(MODELS_DIR, 'training_history.png')
    plt.savefig(plot_path)

    tracking.log_metrics(run_dir, {
        'final_val_auc': float(max(history.history['val_auc'])),
        'final_val_loss': float(min(history.history['val_loss'])),
        'decision_threshold': threshold,
        'epochs_run': len(history.history['loss']),
    })

    print(f'\nModel saved  -> {model_cfg.model_path}')
    print(f'Plot saved   -> {plot_path}')
    print(f'Run logged   -> {run_dir}')
