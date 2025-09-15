
from __future__ import annotations
import os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImagePathDataset(Dataset):

    """
    Crea un dataset per una llista de 
    """


    def __init__(self, paths: Sequence[Path], transform=None):
        # Normalitza qualsevol "paths" de string a objectes Path i els guarda 
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self):
        # Numero de imatges disponibles
        return len(self.paths)
    
    def __getitem__(self, idx: int):
        # Carrega l'imatge del disk ("L" per escala de grisos)
        img = Image.open(self.paths[idx]).convert("L")
        
        # Aplicam les transformacions indicades per l'usuari
        if self.transform: 
            img = self.transform(img)
        
        # Retornam tensor + path original com a string
        return img, str(self.paths[idx])
    
def default_transform(img_size: int, *, grayscale: bool = True):
    """
    Construim un transform de torchvision
        - Posibilitat de convertir a escala de grisos
        - Canvia de dimensions a (img_size, img_size) 
        - Conversio a tensor de PyTorch en [0, 1]
    """

    t = []
    if grayscale: t.append(transforms.Grayscale())
    t.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return transforms.Compose(t)

def list_images(root: str | Path) -> List[Path]:
    """
    Recoleccio de imatges recursivament.

    Retorna una llista de objectes Path.
    """
    root = Path(root)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def list_concepts(concepts_root: str | Path) -> List[str]:
    """
    Llistar conceptes de la carpeta de conceptes.
    La carpeta ha de tenir una forma tal que:
        conceptes/
            cercle/
            quadrat/
            triangle/
            etc/
    """
    root = Path(concepts_root)
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def concept_image_paths(concepts_root: str | Path, concept: str) -> List[Path]:
    """
    Getter de totes les imatges d'un concepte
    """
    return list_images(Path(concepts_root) / concept)

def build_loader(
    image_paths: Sequence[Path], *,
    img_size: int = 128,
    batch_size: int = 256,
    num_workers: int = 0,
    grayscale: bool = True,
    shuffle: bool = False
) -> DataLoader:
    
    """
    Construccio d'un DataLoader a partir de una llista de paths de imatge.

    Args:
        image_paths: Llista de fitxer a carregar
        img_size: Redimensionar les imatges a dimensions especificades.
        batch_size: Total de instancies per al batch.
        num_workers: Workers de Pytorch Dataloader.
        grayscale: Bool per saber si greyscale o no.
        shuffle: Shuffle les mostres cada epoca/episodi
    
    Returns:
        torch.utils.data.DataLoader que proporciona (tensor_image, path_str).

    """

    ds = ImagePathDataset(
        image_paths,
        transform=default_transform(img_size, grayscale=grayscale)
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    


