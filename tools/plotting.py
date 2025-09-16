from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional, Mapping, Any

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# ---- Helpers ---------------------------------------------------------------

def _scores_to_df(scores: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """
    Accepts either:
      scores[layer][concept] = float
    or
      scores[layer][concept] = {"mean": float, "std": float, "n": int}
    Returns a tidy DataFrame: columns = [layer, concept, score, std, n]
    """
    rows: List[Dict[str, Any]] = []
    for layer, cd in scores.items():
        for concept, val in cd.items():
            if isinstance(val, dict):
                rows.append({
                    "layer": layer,
                    "concept": concept,
                    "score": float(val.get("mean", np.nan)),
                    "std": float(val.get("std", np.nan)),
                    "n": int(val.get("n", 0)),
                })
            else:
                rows.append({
                    "layer": layer,
                    "concept": concept,
                    "score": float(val),
                    "std": np.nan,
                    "n": 0,
                })
    return pd.DataFrame(rows)

def _concept_order(df: pd.DataFrame, top_k: int | None = None) -> List[str]:
    order = (
        df.groupby("concept")["score"]
          .mean()
          .sort_values(ascending=False)
          .index
          .tolist()
    )
    return order if top_k is None else order[:top_k]

# ---- Plots -----------------------------------------------------------------

def plot_tcav_bars_multi(
    scores: Mapping[str, Mapping[str, Any]],
    title: str | None = None,
    save_dir: str | None = None,
    top_k: int | None = None,
):

    df = _scores_to_df(scores)
    if df.empty:
        print("No data to plot.")
        return

    order = _concept_order(df, top_k=top_k)
    df["concept"] = pd.Categorical(df["concept"], categories=order, ordered=True)

    layers = df["layer"].unique().tolist()
    for layer in layers:
        sub = df[df["layer"] == layer].sort_values("concept")
        x = np.arange(len(sub))
        y = sub["score"].to_numpy()
        yerr = sub["std"].to_numpy() if "std" in sub and not sub["std"].isna().all() else None

        # Figure size scaled by number of concepts for readability
        plt.figure(figsize=(max(8, 0.55 * len(sub)), 4.5))
        plt.bar(x, y)
        if yerr is not None:
            plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, linewidth=1)

        plt.xticks(x, sub["concept"].astype(str), rotation=60, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("TCAV score")
        plt.title(f"{title or 'TCAV scores'} — {layer}")
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"tcav_{layer}.png"), dpi=200)
        plt.show()

def plot_tcav_heatmap_multi(
    scores: Mapping[str, Mapping[str, Any]],
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Heatmap of layers (rows) × concepts (cols) using the mean/score.
    """
    df = _scores_to_df(scores)
    if df.empty:
        print("No data to plot.")
        return

    
    order = _concept_order(df)
    pivot = (
        df.pivot(index="layer", columns="concept", values="score")
          .reindex(columns=order)
          .sort_index()
    )

    plt.figure(figsize=(max(8, 0.5 * pivot.shape[1]), 0.6 * pivot.shape[0] + 1.5))
    im = plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=60, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    cbar = plt.colorbar(im)
    cbar.set_label("TCAV score", rotation=90)
    plt.title(title or "TCAV scores (layer × concept)")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()

