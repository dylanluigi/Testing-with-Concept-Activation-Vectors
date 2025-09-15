from __future__ import annotations
from typing import Dict, List, Sequence, Optional, Tuple

import torch

try:
    from captum.concept import TCAV, Concept, Classifier 
except Exception:
    TCAV = None  
    Concept = None  
    Classifier = object  

class HookedClassifier(Classifier): 
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

def run_tcav_for_sets(
    classifier: "Classifier",
    experimental_sets: Sequence[Sequence["Concept"]],
    layers: Sequence[torch.nn.Module],
    *,
    random_state: int = 0,
    processes: int = 0,
) -> Dict[str, Dict[Tuple[str, str], Dict[str, float]]]:
    """
    Returns:
      {
        layer_name: {
          (concept_name, random_name): {
            "pos": <int>,
            "neg": <int>,
            "rate": <float in [0,1]>,   # pos / (pos+neg)
          }, ...
        }, ...
      }
    """
    if TCAV is None:
        raise ImportError("captum.concept not available. Install captum to run TCAV.")

    tcav = TCAV(classifier, layers)
    raw = tcav.interpret(experimental_sets, random_state=random_state, processes=processes)

    results: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {}
    for exp_idx, per_layer in raw.items():
        # Captum uses keys like "0-1" (conceptIndex-randomIndex)
        c_idx_str, r_idx_str = str(exp_idx).split("-")
        c_idx, r_idx = int(c_idx_str), int(r_idx_str)
        concept_name = experimental_sets[c_idx][0].name
        random_name  = experimental_sets[r_idx][1].name

        for layer_name, stats in per_layer.items():
            sign = stats.get("sign_count", None)
            if sign is None or getattr(sign, "numel", lambda: 0)() < 2:
                continue
            pos = float(sign[0].item()) if hasattr(sign[0], "item") else float(sign[0])
            neg = float(sign[1].item()) if hasattr(sign[1], "item") else float(sign[1])
            total = max(pos + neg, 1.0)
            rate = pos / total

            results.setdefault(layer_name, {})[(concept_name, random_name)] = {
                "pos": pos, "neg": neg, "rate": rate
            }
    return results

