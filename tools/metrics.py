from __future__ import annotations
from typing import Dict, List
import numpy as np

def aggregate_tcav_runs(runs: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    TODO comment
    """
    out: Dict[str, Dict[str, float]] = {}
    if not runs: return out
    layers = runs[0].keys()
    for layer in layers:
        all_concepts = runs[0][layer].keys()
        out[layer] = {}
        for c in all_concepts:
            vals = [r[layer][c] for r in runs if c in r[layer]]
            out[layer][c] = float(np.nanmean(vals)) if vals else float("nan")
    return out
