"""CLI: render every registered explanation method on one NORMAL and one
PNEUMONIA sample (replaces the old src/gradcam.py + src/shap_explain.py,
which each hardcoded a single method with duplicated preprocessing).

Usage: python scripts/explain.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xai_cxr.config import DataConfig, ModelConfig, MODELS_DIR
from xai_cxr.data import read_manifest, load_image
from xai_cxr.models.baseline import XAIModel, load_trained
from xai_cxr.explain.registry import METHODS, METHOD_INFO, explain as registry_explain
from xai_cxr.explain._common import overlay

if __name__ == '__main__':
    data_cfg = DataConfig.load()
    model_cfg = ModelConfig.load()

    model = load_trained(model_cfg.model_path)
    xai_model = XAIModel(model, model_cfg.last_conv_layer)

    rows = read_manifest('test')
    sample = {
        'NORMAL': next(r for r, l in rows if l == 'NORMAL'),
        'PNEUMONIA': next(r for r, l in rows if l == 'PNEUMONIA'),
    }

    method_names = list(METHODS)
    fig, axes = plt.subplots(2, len(method_names) + 1, figsize=(3 * (len(method_names) + 1), 6))

    for row, (label, relpath) in enumerate(sample.items()):
        img = load_image(os.path.join(data_cfg.dataset_dir, relpath), data_cfg.img_size)
        proba = float(xai_model.proba(img[None])[0])
        pred = 'PNEUMONIA' if proba > 0.5 else 'NORMAL'

        axes[row, 0].imshow(img.astype('uint8'))
        axes[row, 0].set_title(f'{label}\npred={pred} ({proba:.2f})', fontsize=9)
        axes[row, 0].axis('off')

        for col, method in enumerate(method_names, start=1):
            heatmap = registry_explain(method, xai_model, img, img_size=data_cfg.img_size)
            blended = overlay(img, heatmap)
            axes[row, col].imshow(blended)
            axes[row, col].set_title(METHOD_INFO[method]['label'], fontsize=9)
            axes[row, col].axis('off')

    plt.tight_layout()
    out_path = os.path.join(MODELS_DIR, 'explanations_grid.png')
    plt.savefig(out_path)
    print(f'Saved: {out_path}')
