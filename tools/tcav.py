from __future__ import annotations
from typing import Dict, List, Sequence, Optional, Tuple, Any, Union
from PIL import Image
import torch
import numpy as np

# These are your own modules, so the relative imports are correct
from . import data as data_mod
from .utils import select_device

try:
    from captum.concept import TCAV, Concept, Classifier
except Exception:
    TCAV = None
    Concept = None
    Classifier = object

class TorchLinearClassifier(Classifier):
    def __init__(self, device: Union[str, torch.device] = "cpu",
                 epochs: int = 200, lr: float = 0.1, weight_decay: float = 0.0):
        self.device = torch.device(device)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self._linear: torch.nn.Linear | None = None
        self._classes: List[int] = []

    def train_and_eval(self, dataloader, **kwargs: Any) -> Dict[str, Any] | None:
        Xs, Ys = [], []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                feats, labels = batch
            else:
                raise RuntimeError("Expected (features, labels) batches")
            # feats: [B,C,H,W] or [B,F] or [C,H,W]
            if feats.ndim == 3:  # [C,H,W] -> add batch dim
                feats = feats.unsqueeze(0)
            Xs.append(feats)
            Ys.append(labels)

        # concat & move to classifier device
        X = torch.cat(Xs, 0).to(self.device)         # [N,...]
        y_raw = torch.cat(Ys, 0).to(self.device).view(-1)  # raw concept IDs (e.g., 29,41)

        # map raw IDs -> [0..C-1]
        classes = torch.unique(y_raw).sort()[0]      # tensor([...])
        self._classes = classes.detach().cpu().tolist()
        y = torch.searchsorted(classes, y_raw).long()  # [N] in 0..C-1

        # flatten features
        X = X.view(X.size(0), -1)                    # [N,F]
        C, F = int(classes.numel()), int(X.size(1))
        if self._linear is None or self._linear.in_features != F or self._linear.out_features != C:
            self._linear = torch.nn.Linear(F, C, bias=False).to(self.device)

        
        opt = torch.optim.SGD(self._linear.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        self._linear.train()
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            logits = self._linear(X)                 # [N,C]
            loss = loss_fn(logits, y)                # y ∈ [0..C-1]
            loss.backward()
            opt.step()

        with torch.no_grad():
            acc = (self._linear(X).argmax(1) == y).float().mean().item()
        return {"accs": torch.tensor(acc)}

    def weights(self) -> torch.Tensor:
        assert self._linear is not None, "Call train_and_eval() first."
        return self._linear.weight.detach().cpu()

    def classes(self) -> List[int]:
        return self._classes
def make_concepts_auto(
    concepts_root: str,
    concept_names: Sequence[str],
    *,
    img_size: int = 128,
    grayscale: bool = True,
    device: Optional[torch.device | str] = None,
    limit_per_concept: int | None = None,
) -> List["Concept"]:
    """
    Builds Captum Concept objects.
    - On CPU: use path-based Concepts (lazy, memory-light).
    - On CUDA/MPS: preload tensors on that device to avoid CPU/GPU mismatch.
    """
    if Concept is None:
        raise ImportError("captum.concept not available. Install captum to run TCAV.")

    dev = torch.device(device) if device is not None else select_device("cuda")
    tfm = data_mod.default_transform(img_size, grayscale=grayscale)

    concepts: List[Concept] = []
    print(f"Building concepts for device: {dev.type}")

    if dev.type == "cpu":
        # Simple, lazy path-based Concepts (low memory usage)
        for i, name in enumerate(concept_names):
            concepts.append(Concept(i, name, concepts_root))
        return concepts

    # GPU/MPS path: preload tensors onto the chosen device
    for i, name in enumerate(concept_names):
        print(f"  - Loading concept: {name}")
        paths = data_mod.concept_image_paths(concepts_root, name)
        if limit_per_concept is not None:
            paths = paths[:limit_per_concept]
        
        exs: List[torch.Tensor] = []
        for p in paths:
            img = Image.open(p).convert("L") if grayscale else Image.open(p).convert("RGB")
            t = tfm(img).to(dev).float()
            exs.append(t)
        
        # Build Concept from the list of tensors already on the device
        concepts.append(Concept(i, name, exs))
    return concepts

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

