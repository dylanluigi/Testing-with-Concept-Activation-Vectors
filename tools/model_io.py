from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Any, Optional, Type

import torch
import torch.nn as nn

from .utils import select_device

def load_module_from_file(py_file: str | Path):
    """
    Carrega dinamicament un modul de Python de un file path.
    """
    py_file = str(py_file)
    spec = importlib.util.spec_from_file_location("user_model_module", py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {py_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod) 
    return mod

def instantiate_model(py_file: str | Path, class_name: str = "Net", *args, **kwargs) -> nn.Module:
    """
    Importa una class model de un .py i instancia.
    """
    mod = load_module_from_file(py_file)
    if not hasattr(mod, class_name):
        raise AttributeError(f"Class {class_name} not found in {py_file}")
    cls = getattr(mod, class_name)
    model = cls(*args, **kwargs)
    return model

def load_checkpoint_state(ckpt_path: str | Path) -> dict:
    """
    Carrera checkpoint de Pythorch.
    """
    ckpt_path = Path(ckpt_path)
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    return state

def load_model_weights(model: nn.Module, ckpt_path: str | Path, strict: bool = False) -> nn.Module:
    """
    Carrega els pesos.
    """
    obj = load_checkpoint_state(ckpt_path)
    state_dict = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print("[load_model_weights] Missing keys:", missing)
    if unexpected:
        print("[load_model_weights] Unexpected keys:", unexpected)
    return model

def build_model_from_py(py_file: str | Path, class_name: str, ckpt_path: str | Path | None, device: Optional[torch.device] = None, **kwargs) -> nn.Module:
    """
    Carregar model i carregar pesos.
    """
    if device is None:
        device = select_device()
    model = instantiate_model(py_file, class_name, **kwargs)
    if ckpt_path is not None:
        load_model_weights(model, ckpt_path)
    model.to(device)
    model.eval()
    return model
