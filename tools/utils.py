from __future__ import annotations
import json, os, random, time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Dict, Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int = 0) -> None:
    """Setter de llavor Python/NumPy/PyTorch RNG per reproduibilitat."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def select_device(prefer: str = "cuda") -> torch.device:
    """
    Seleccio de dispositu.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@contextmanager
def timeit(label: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[timeit] {label}: {dt:.3f}s")


def ensure_dir(p: os.PathLike | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Any, path: os.PathLike | str, *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)

def load_json(path: os.PathLike | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def list_named_layers(model: nn.Module, *, include_types: Sequence[type] | None = None) -> Dict[str, nn.Module]:
    """
    Reotrna una llista de les capes del model.
    """
    out: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        # leaf module has no children
        if len(list(module.children())) == 0:
            if include_types is None or isinstance(module, tuple(include_types)):
                out[name] = module
    return out

def resolve_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Solucionador de layers.
    """
    cur: nn.Module = model
    for part in layer_name.split("."):
        cur = getattr(cur, part)
    if not isinstance(cur, nn.Module):
        raise ValueError(f"{layer_name} did not resolve to nn.Module")
    return cur

def chunked(seq: Sequence, n: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

def flatten(nested: Iterable[Iterable[Any]]) -> List[Any]:
    return [x for xs in nested for x in xs]
