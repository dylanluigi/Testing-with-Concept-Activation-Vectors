
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
    

# ---------------------------------------------------------------------
# Generacio de pools randoms synthetics
# ---------------------------------------------------------------------

def _rand_color() -> Tuple[int, int, int]:
    """
    Crea una tupla RGB aleatoria.
    """
    return tuple(np.random.randint(0, 256, size=3).tolist())

def _draw_random_shape(draw: ImageDraw.ImageDraw, w: int, h: int):
    """
    Dibuixa una primitiva simple,
    """
    import numpy as _np
    shape = _np.random.choice(["ellipse", "rect", "line"]) 
    xy = [
        int(_np.random.randint(0, w//2)),
        int(_np.random.randint(0, h//2)),
        int(_np.random.randint(w//2, w)),
        int(_np.random.randint(h//2, h)),
    ]
    if shape == "ellipse":
        draw.ellipse(xy, outline=_rand_color(), width=int(_np.random.randint(1, 6)))
    elif shape == "rect":
        draw.rectangle(xy, outline=_rand_color(), width=int(_np.random.randint(1, 6)))
    else:
        draw.line(xy, fill=_rand_color(), width=int(_np.random.randint(1, 6)))

def synth_random_image(size: int = 128) -> Image.Image:
    """
    Crea un imatge random sintetica.
    """
    
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    base = Image.fromarray(arr)
    draw = ImageDraw.Draw(base)
    
    for _ in range(int(np.random.randint(1, 4))):
        _draw_random_shape(draw, size, size)
    return base

def build_random_pool(
    out_dir: str | Path, *,
    count: int = 2000,
    size: int = 128,
    prefix: str = "rand"
) -> List[Path]:
    """
    Generacio de fitxers amb imatges randomitzades sintetiques.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)  
    paths: List[Path] = []
    for i in range(count):
        img = synth_random_image(size)          
        p = out / f"{prefix}_{i:06d}.png"       
        img.save(p)                              
        paths.append(p)
    return paths

# ---------------------------------------------------------------------
# Sets experimentals per a TCAV
# ---------------------------------------------------------------------

def build_experimental_sets(
    concepts: Sequence[str],
    n_random: int,
    resamples: int,
    *,
    random_pool_names: Sequence[str]
) -> List[List[str]]:
    """

    Cear una llista de sets experimentals (parells de) per a TCAV, tal que:
    [nom_conepte, pool_random_nom]

    Exemples:
      concepts        = ['circle', 'square']
      random_pool_names = ['random_pool_0', 'random_pool_1']
      n_random        = 2
      resamples       = 3

    Notes:
      - We shuffle `random_pool_names` before each resample to vary pairings.
      - Only the first `n_random` random pools after shuffling are used each round.
    """
    exps: List[List[str]] = []
    rp = list(random_pool_names)  
    for _ in range(resamples):
        random.shuffle(rp)  
        for c in concepts:
            for r in rp[:n_random]:
                exps.append([c, r])
    return exps
