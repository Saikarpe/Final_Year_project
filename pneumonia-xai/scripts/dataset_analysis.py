"""Dataset analysis plots for /dataset -- class distribution, pie chart, and
a sample-image grid. Counts come from configs/splits/*.txt (the patient-
grouped manifests, xai_cxr.data.read_manifest), not a hardcoded dict -- those
numbers changed the moment the splits were re-derived patient-level (D-7),
and hardcoding them here would silently drift from what /dataset's live
table shows.
"""
import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xai_cxr.config import DataConfig
from xai_cxr.data import read_manifest

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'chest_xray')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

SPLIT_NAMES = ['train', 'val', 'calibration', 'test']

cfg = DataConfig.load()
splits = {}
for name in SPLIT_NAMES:
    rows = read_manifest(name)
    splits[name.capitalize()] = {
        'NORMAL': sum(1 for _, l in rows if l == 'NORMAL'),
        'PNEUMONIA': sum(1 for _, l in rows if l == 'PNEUMONIA'),
    }

# ── Class Distribution Bar Chart ───────────────────────────────────────────────
labels = list(splits.keys())
normal = [splits[s]['NORMAL'] for s in labels]
pneumonia = [splits[s]['PNEUMONIA'] for s in labels]
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(7, 4))
bars1 = ax.bar(x - 0.2, normal, 0.4, label='NORMAL', color='#2e7d32')
bars2 = ax.bar(x + 0.2, pneumonia, 0.4, label='PNEUMONIA', color='#c62828')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Number of Images')
ax.set_title('Class Distribution per Split (patient-grouped, D-7)')
ax.legend()
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'class_distribution.png'))
plt.clf()
print("Saved: class_distribution.png")

# ── Pie Chart (Train split) ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    [splits['Train']['NORMAL'], splits['Train']['PNEUMONIA']],
    labels=['NORMAL', 'PNEUMONIA'],
    colors=['#2e7d32', '#c62828'],
    autopct='%1.1f%%',
    startangle=90
)
ax.set_title('Training Set Class Distribution')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'class_pie.png'))
plt.clf()
print("Saved: class_pie.png")

# ── Sample Images Grid ─────────────────────────────────────────────────────────
train_rows = read_manifest('train')
normal_paths = [r for r, l in train_rows if l == 'NORMAL']
pneumonia_paths = [r for r, l in train_rows if l == 'PNEUMONIA']

normal_samples = random.sample(normal_paths, 4)
pneumonia_samples = random.sample(pneumonia_paths, 4)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, relpath in enumerate(normal_samples):
    img = Image.open(os.path.join(cfg.dataset_dir, relpath)).resize((224, 224))
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title('NORMAL', fontsize=9, color='#2e7d32')
    axes[0, i].axis('off')

for i, relpath in enumerate(pneumonia_samples):
    img = Image.open(os.path.join(cfg.dataset_dir, relpath)).resize((224, 224))
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title('PNEUMONIA', fontsize=9, color='#c62828')
    axes[1, i].axis('off')

plt.suptitle('Sample X-Ray Images', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'sample_images.png'))
plt.clf()
print("Saved: sample_images.png")

print("All dataset analysis plots saved.")
