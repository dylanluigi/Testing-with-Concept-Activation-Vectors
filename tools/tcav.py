from __future__ import annotations
from typing import Dict, List, Sequence, Optional, Tuple

import torch

try:
    from captum.concept import TCAV, Concept, Classifier  # type: ignore
except Exception:
    TCAV = None  # type: ignore
    Concept = None  # type: ignore
    Classifier = object  # type: ignore

class HookedClassifier(Classifier): 
    """Minimal PyTorch classifier wrapper for Captum TCAV.
    Replace with your own wrapper if you already have one."""
    def __init__(self, model: torch.nn.Module, layers: Sequence[torch.nn.Module], device: torch.device):
        self.model = model
        self.layers = list(layers)
        self.device = device

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs.to(self.device))

    def layer_activations(self, layer: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        acts = None
        def hook(_m, _in, out):
            nonlocal acts
            acts = out.detach()
        h = layer.register_forward_hook(hook)
        _ = self.forward(inputs)
        h.remove()
        if acts is None:
            raise RuntimeError("Failed to capture activations")
        return acts

def make_concepts(concepts_root: str, concept_names: Sequence[str]) -> List["Concept"]:
    if Concept is None:
        raise ImportError("captum.concept not available. Install captum to run TCAV.")
    cons = []
    for i, name in enumerate(concept_names):
        cons.append(Concept(i, name, concepts_root))
    return cons

def run_tcav_for_sets(classifier: "Classifier", experimental_sets: Sequence[Sequence[str]], layers: Sequence[torch.nn.Module], *, random_state: int = 0, processes: int = 0) -> Dict[str, Dict[str, float]]:
    """Execute TCAV across experimental sets and return a compact dict:
        {layer_name: {concept_name: mean_tcav_score, ...}, ...}
    NOTE: Thin wrapper; adapt for your own post-processing.
    """
    if TCAV is None:
        raise ImportError("captum.concept not available. Install captum to run TCAV.")

    tcav = TCAV(classifier, layers)
    raw = tcav.interpret(experimental_sets, random_state=random_state, processes=processes)

    results: Dict[str, Dict[str, float]] = {}
    for exp_key, per_layer in raw.items():
        for layer_name, stats in per_layer.items():
            # Captum typically returns tensors under 'sign_count' and 'magnitude'.
            sign = stats.get("sign_count", None)
            if sign is None:
                continue
            try:
                # Treat sign[0] as "positive count"
                score = float(sign[0].item() if hasattr(sign[0], "item") else sign[0])
            except Exception:
                try:
                    score = float(sign.item())
                except Exception:
                    score = float("nan")
            results.setdefault(layer_name, {})[exp_key] = score
    return results
