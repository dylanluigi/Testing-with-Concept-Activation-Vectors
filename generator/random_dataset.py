#!/usr/bin/env python3
# make_random_pool_combiners.py
# Build a highly heterogeneous random_pool by *combining* your own photos.
import argparse, random, math
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------- IO & preprocessing ----------------------
def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def open_rgb_square(p: Path, size: int) -> Image.Image:
    img = Image.open(p).convert("RGB")
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((int(round(w*scale)), int(round(h*scale))), Image.BICUBIC)
    w, h = img.size
    x0 = (w - size) // 2
    y0 = (h - size) // 2
    return img.crop((x0, y0, x0 + size, y0 + size))

def basic_aug(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5: img = ImageOps.mirror(img)
    if rng.random() < 0.3: img = ImageOps.flip(img)
    # slight random crop & resize back (jitter 0–6%)
    if rng.random() < 0.5:
        w, h = img.size
        pad = int(0.06 * min(w, h) * rng.random())
        if pad > 0:
            img = img.crop((pad, pad, w - pad, h - pad)).resize((w, h), Image.BICUBIC)
    # small photometric jitter
    if rng.random() < 0.5: img = ImageEnhance.Brightness(img).enhance(0.8 + 0.4 * rng.random())
    if rng.random() < 0.5: img = ImageEnhance.Contrast(img).enhance(0.8 + 0.4 * rng.random())
    if rng.random() < 0.5: img = ImageEnhance.Color(img).enhance(0.8 + 0.6 * rng.random())
    if rng.random() < 0.2: img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 1.0)))
    return img

def npimg(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)

def pilimg(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

# ---------------------- combiners ----------------------
def mosaic2x2(imgs: List[Image.Image], rng: random.Random, size: int) -> Image.Image:
    assert len(imgs) >= 4
    # random split points (avoid tiny tiles)
    x_split = rng.randint(int(0.35*size), int(0.65*size))
    y_split = rng.randint(int(0.35*size), int(0.65*size))
    out = Image.new("RGB", (size, size))
    tiles = [
        (0,           0,            x_split,      y_split),      # TL
        (x_split,     0,            size,         y_split),      # TR
        (0,           y_split,      x_split,      size),         # BL
        (x_split,     y_split,      size,         size),         # BR
    ]
    rng.shuffle(imgs)
    for tile, src in zip(tiles, imgs[:4]):
        x0,y0,x1,y1 = tile
        w,h = x1-x0, y1-y0
        patch = src.resize((w,h), Image.BICUBIC)
        out.paste(patch, (x0,y0))
    return out

def mixup(a: Image.Image, b: Image.Image, rng: random.Random) -> Image.Image:
    # alpha in [0.3,0.7] to avoid near-duplicates
    alpha = 0.3 + 0.4 * rng.random()
    A, B = npimg(a), npimg(b)
    C = alpha * A + (1 - alpha) * B
    return pilimg(C)

def cutmix(a: Image.Image, b: Image.Image, rng: random.Random, n_patches: int = None) -> Image.Image:
    # paste 1–5 random rectangles from b onto a
    if n_patches is None:
        n_patches = rng.randint(1, 5)
    out = a.copy()
    W, H = out.size
    for _ in range(n_patches):
        pw = rng.randint(W//8, W//2)
        ph = rng.randint(H//8, H//2)
        px = rng.randint(0, W - pw)
        py = rng.randint(0, H - ph)
        bx = rng.randint(0, W - pw)
        by = rng.randint(0, H - ph)
        patch = b.crop((bx, by, bx + pw, by + ph))
        out.paste(patch, (px, py))
    return out

def patchwork(imgs: List[Image.Image], rng: random.Random, size: int) -> Image.Image:
    # start from one image; paste many tiny random patches from others
    base = imgs[0].copy()
    W, H = base.size
    n = rng.randint(10, 40)
    for _ in range(n):
        src = rng.choice(imgs)
        pw = rng.randint(W//16, W//6)
        ph = rng.randint(H//16, H//6)
        bx = rng.randint(0, W - pw)
        by = rng.randint(0, H - ph)
        px = rng.randint(0, W - pw)
        py = rng.randint(0, H - ph)
        patch = src.crop((bx, by, bx + pw, by + ph))
        # optional light alpha to soften seams
        if rng.random() < 0.5:
            alpha = int(128 + 127 * rng.random())  # 128..255
            mask = Image.new("L", (pw, ph), alpha)
            base.paste(patch, (px, py), mask)
        else:
            base.paste(patch, (px, py))
    return base

# ---------------------- driver ----------------------
def main():
    ap = argparse.ArgumentParser("Compose personal photos into a diverse TCAV random_pool")
    ap.add_argument("--src", required=True, help="Folder with photos (recursively scanned)")
    ap.add_argument("--dst", required=True, help="Output folder (e.g., .../TCAV_data/random_pool_combo)")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-std", type=float, default=5.0, help="Reject near-solid outputs (pixel std threshold)")
    ap.add_argument("--weights", default="mosaic:0.3,mixup:0.25,cutmix:0.25,patchwork:0.2",
                    help="Mode weights, comma-separated, e.g. mosaic:0.4,mixup:0.3,cutmix:0.2,patchwork:0.1")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    paths = list_images(src)
    if not paths:
        raise SystemExit(f"No images found under {src}")

    # parse weights
    weights = {}
    for tok in args.weights.split(","):
        name, val = tok.split(":")
        weights[name.strip()] = float(val)
    modes = list(weights.keys())
    probs = np.array([weights[m] for m in modes], dtype=np.float64)
    probs = probs / probs.sum()

    def sample_imgs(k: int) -> List[Image.Image]:
        picks = rng.sample(paths, k) if len(paths) >= k else [rng.choice(paths) for _ in range(k)]
        imgs = [basic_aug(open_rgb_square(p, args.size), rng) for p in picks]
        return imgs

    # writer
    filelist = (dst / "filelist.txt").open("w")
    written = 0
    i = 0
    try:
        while written < args.n:
            mode = rng.choices(modes, weights=probs, k=1)[0]
            if mode == "mosaic":
                imgs = sample_imgs(4)
                out = mosaic2x2(imgs, rng, args.size)
            elif mode == "mixup":
                a, b = sample_imgs(2)
                out = mixup(a, b, rng)
            elif mode == "cutmix":
                a, b = sample_imgs(2)
                out = cutmix(a, b, rng)
            elif mode == "patchwork":
                imgs = sample_imgs(rng.randint(4, 10))
                out = patchwork(imgs, rng, args.size)
            else:
                # fallback: just load one and augment
                out = sample_imgs(1)[0]

            # reject overly uniform outputs
            std = np.array(out).astype(np.float32).std()
            if std < args.min_std:
                continue

            name = f"rand_combo_{i:06d}.png"
            out.save(dst / name)
            filelist.write(f"{name}\n")
            i += 1
            written += 1
    finally:
        filelist.close()

    print(f"Wrote {written} images to {dst}")
    print(f"File list: {dst/'filelist.txt'}")

if __name__ == "__main__":
    main()
