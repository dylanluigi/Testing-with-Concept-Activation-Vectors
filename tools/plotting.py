from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

def plot_tcav_bars(scores_by_layer: Dict[str, Dict[str, float]], *, sort: bool = True, title: Optional[str] = None):
    for layer, d in scores_by_layer.items():
        labels = list(d.keys())
        vals = np.array([d[k] for k in labels], dtype=float)
        if sort:
            idx = np.argsort(-vals)
            labels = [labels[i] for i in idx]
            vals = vals[idx]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, vals)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("TCAV score")
        ttl = title or "TCAV scores"
        ax.set_title(f"{ttl} — {layer}")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.show()

def plot_tcav_heatmap(scores_by_layer: Dict[str, Dict[str, float]], *, title: Optional[str] = None):
    layers = list(scores_by_layer.keys())
    concepts = sorted({c for layer in layers for c in scores_by_layer[layer].keys()})
    mat = np.zeros((len(layers), len(concepts)), dtype=float)
    for i, layer in enumerate(layers):
        for j, c in enumerate(concepts):
            mat[i, j] = scores_by_layer[layer].get(c, np.nan)
    fig, ax = plt.subplots(figsize=(max(6, len(concepts) * 0.6), max(4, len(layers) * 0.5)))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(concepts))); ax.set_xticklabels(concepts, rotation=30)
    ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
    ax.set_title(title or "TCAV score heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
