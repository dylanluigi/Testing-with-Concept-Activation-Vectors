#!/usr/bin/env python3
"""
AIXI-style TCAV concept synthesizer (shapes + fractions).

Generates black-background (0) grayscale images with low-intensity shapes
(default 1..10) matching AIXI dataset style:

Concepts:
  Circles: circle_full, circle_half_{top,bottom,left,right}, circle_quarter_{tl,tr,bl,br}
  Squares: square_full, square_half_{top,bottom,left,right}, square_corner_{tl,tr,bl,br}, square_edge_{top,bottom,left,right}
  Crosses: cross_full, cross_arm_{horizontal,vertical,top,bottom,left,right}
  Random:  random_pool (backgrounds)

Notes
- Positioning uses a jittered 3x3 grid so concepts aren’t always centered.
- No cropping → patches naturally include background (no “all-white” issues).
- Intensities default to 1..10; add --white-only for pure 255 foreground.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, List, Tuple

# ------------------------ utils ------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_intensity_spec(spec: str) -> Tuple[int, int]:
    # "1-10" or "3-8"; fallback to 1..10
    if "-" in spec:
        a, b = spec.split("-", 1)
        return max(0, int(a)), min(255, int(b))
    return 1, 10

def bg(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)

def grid_position(size: int) -> Tuple[int, int, int]:
    """Return (cx, cy, cell_size) for a jittered 3x3 grid."""
    cell = size // 3
    gx = np.random.randint(0, 3)
    gy = np.random.randint(0, 3)
    cx = gx * cell + cell // 2 + np.random.randint(-cell//6, cell//6 + 1)
    cy = gy * cell + cell // 2 + np.random.randint(-cell//6, cell//6 + 1)
    m = cell // 3
    cx = int(np.clip(cx, m, size - m))
    cy = int(np.clip(cy, m, size - m))
    return cx, cy, cell

def intensity(minmax: Tuple[int, int], white_only: bool) -> int:
    return 255 if white_only else int(np.random.randint(minmax[0], minmax[1] + 1))

def save_concepts(concepts: Dict[str, List[np.ndarray]], out_root: Path):
    for name, imgs in concepts.items():
        d = out_root / name
        ensure_dir(d)
        for i, im in enumerate(imgs):
            cv2.imwrite(str(d / f"{name}_{i:05d}.png"), im)
        print(f"  ✓ {name}: {len(imgs)}")

# ------------------------ generators ------------------------

def gen_circle_set(size: int, n: int, inten: Tuple[int, int], white_only: bool) -> Dict[str, List[np.ndarray]]:
    result = {"circle_full": [], "circle_half_top": [], "circle_half_bottom": [],
              "circle_half_left": [], "circle_half_right": [],
              "circle_quarter_tl": [], "circle_quarter_tr": [],
              "circle_quarter_bl": [], "circle_quarter_br": []}

    for _ in range(n):
        img = bg(size)
        cx, cy, cell = grid_position(size)
        r = int(cell * 0.3 * np.random.uniform(0.85, 1.25))
        r = int(np.clip(r, 5, size // 3))
        val = intensity(inten, white_only)

        # full
        im_full = img.copy()
        cv2.circle(im_full, (cx, cy), r, val, -1)
        result["circle_full"].append(im_full)

        # masks
        H = W = size
        yy, xx = np.indices((H, W))
        circle_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), r, 255, -1)

        halves = {
            "top": (yy < cy),
            "bottom": (yy >= cy),
            "left": (xx < cx),
            "right": (xx >= cx),
        }
        for name, cond in halves.items():
            m = (cond.astype(np.uint8) * 255) & circle_mask
            im = img.copy()
            im[m > 0] = val
            result[f"circle_half_{name}"].append(im)

        quarters = {
            "tl": (yy < cy) & (xx < cx),
            "tr": (yy < cy) & (xx >= cx),
            "bl": (yy >= cy) & (xx < cx),
            "br": (yy >= cy) & (xx >= cx),
        }
        for name, cond in quarters.items():
            m = (cond.astype(np.uint8) * 255) & circle_mask
            im = img.copy()
            im[m > 0] = val
            result[f"circle_quarter_{name}"].append(im)

    return result

def gen_square_set(size: int, n: int, inten: Tuple[int, int], white_only: bool) -> Dict[str, List[np.ndarray]]:
    result = {"square_full": [], "square_half_top": [], "square_half_bottom": [],
              "square_half_left": [], "square_half_right": [],
              "square_corner_tl": [], "square_corner_tr": [],
              "square_corner_bl": [], "square_corner_br": [],
              "square_edge_top": [], "square_edge_bottom": [],
              "square_edge_left": [], "square_edge_right": []}

    for _ in range(n):
        img = bg(size)
        cx, cy, cell = grid_position(size)
        s = int(cell * 0.6 * np.random.uniform(0.85, 1.25))
        s = int(np.clip(s, 10, size // 2))
        x = int(np.clip(cx - s // 2, 0, size - s))
        y = int(np.clip(cy - s // 2, 0, size - s))
        val = intensity(inten, white_only)

        # full
        im_full = img.copy()
        cv2.rectangle(im_full, (x, y), (x + s, y + s), val, -1)
        result["square_full"].append(im_full)

        # halves
        halves = {
            "top":    (slice(y, y + s // 2), slice(x, x + s)),
            "bottom": (slice(y + s // 2, y + s), slice(x, x + s)),
            "left":   (slice(y, y + s), slice(x, x + s // 2)),
            "right":  (slice(y, y + s), slice(x + s // 2, x + s)),
        }
        for name, (ys, xs) in halves.items():
            im = img.copy()
            im[ys, xs] = val
            result[f"square_half_{name}"].append(im)

        # corners
        c = int(max(4, round(s * 0.40)))
        corners = {
            "tl": (slice(y, y + c), slice(x, x + c)),
            "tr": (slice(y, y + c), slice(x + s - c, x + s)),
            "bl": (slice(y + s - c, y + s), slice(x, x + c)),
            "br": (slice(y + s - c, y + s), slice(x + s - c, x + s)),
        }
        for name, (ys, xs) in corners.items():
            im = img.copy()
            im[ys, xs] = val
            result[f"square_corner_{name}"].append(im)

        # edges
        t = max(2, int(round(s * 0.15)))
        edges = {
            "top":    ((y, x), (x + s, y + t)),
            "bottom": ((y + s - t, x), (x + s, y + s)),
            "left":   ((y, x), (x + t, y + s)),
            "right":  ((y, x + s - t), (x + s, y + s)),
        }
        for name, ((y1, x1), (x2, y2)) in edges.items():
            im = img.copy()
            cv2.rectangle(im, (x1, y1), (x2, y2), val, -1)
            result[f"square_edge_{name}"].append(im)

    return result

def gen_cross_set(size: int, n: int, inten: Tuple[int, int], white_only: bool) -> Dict[str, List[np.ndarray]]:
    result = {"cross_full": [], "cross_arm_horizontal": [], "cross_arm_vertical": [],
              "cross_arm_top": [], "cross_arm_bottom": [], "cross_arm_left": [], "cross_arm_right": []}

    for _ in range(n):
        img = bg(size)
        cx, cy, cell = grid_position(size)
        span = int(cell * 0.55 * np.random.uniform(0.85, 1.2))
        span = int(np.clip(span, size // 8, size // 2))
        thick = max(2, span // 5)
        val = intensity(inten, white_only)

        # full cross
        im_full = img.copy()
        cv2.rectangle(im_full, (cx - span//2, cy - thick//2), (cx + span//2, cy + thick//2), val, -1)
        cv2.rectangle(im_full, (cx - thick//2, cy - span//2), (cx + thick//2, cy + span//2), val, -1)
        result["cross_full"].append(im_full)

        # individual arms
        def horiz(dst, left, right, y1, y2): cv2.rectangle(dst, (left, y1), (right, y2), val, -1)
        def vert(dst, top, bottom, x1, x2): cv2.rectangle(dst, (x1, top), (x2, bottom), val, -1)

        # horizontal (full width at y=cy)
        im = img.copy(); horiz(im, cx - span//2, cx + span//2, cy - thick//2, cy + thick//2)
        result["cross_arm_horizontal"].append(im)
        # vertical (full height at x=cx)
        im = img.copy(); vert(im, cy - span//2, cy + span//2, cx - thick//2, cx + thick//2)
        result["cross_arm_vertical"].append(im)

        # top/bottom/left/right halves
        im = img.copy(); vert(im, cy - span//2, cy, cx - thick//2, cx + thick//2)
        result["cross_arm_top"].append(im)
        im = img.copy(); vert(im, cy, cy + span//2, cx - thick//2, cx + thick//2)
        result["cross_arm_bottom"].append(im)
        im = img.copy(); horiz(im, cx - span//2, cx, cy - thick//2, cy + thick//2)
        result["cross_arm_left"].append(im)
        im = img.copy(); horiz(im, cx, cx + span//2, cy - thick//2, cy + thick//2)
        result["cross_arm_right"].append(im)

    return result

def gen_random_pool(size: int, count: int) -> Dict[str, List[np.ndarray]]:
    imgs: List[np.ndarray] = []
    for _ in range(count):
        if np.random.rand() < 0.8:
            imgs.append(np.zeros((size, size), dtype=np.uint8))
        else:
            # sparse salt noise at low levels, still "backgroundish"
            im = np.zeros((size, size), dtype=np.uint8)
            mask = np.random.rand(size, size) < 0.03
            vals = np.random.randint(1, 3, size=(size, size), dtype=np.uint8)
            im[mask] = vals[mask]
            imgs.append(im)
    return {"random_pool": imgs}

# ------------------------ CLI ------------------------

def main():
    ap = argparse.ArgumentParser(description="Synthesize AIXI-style TCAV concepts (shapes + fractions).")
    ap.add_argument("--out", required=True, help="Output root directory for concept folders")
    ap.add_argument("--size", type=int, default=128, help="Image width/height (square)")
    ap.add_argument("--per-concept", type=int, default=200, help="Images per concept (for shapes/fractions)")
    ap.add_argument("--intensities", default="1-10", help="Intensity range like '1-10'")
    ap.add_argument("--white-only", action="store_true", help="Use pure white (255) foreground instead of 1..10")
    ap.add_argument("--concepts", nargs="+",
                    choices=["circles", "squares", "crosses", "all"], default=["all"],
                    help="Which concept families to generate")
    ap.add_argument("--random-pool", action="store_true", help="Also generate random_pool backgrounds")
    ap.add_argument("--random-pool-count", type=int, default=0, help="How many random_pool images")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_root = Path(args.out); ensure_dir(out_root)
    inten = parse_intensity_spec(args.intensities)

    print("=" * 60)
    print("AIXI-STYLE TCAV CONCEPT SYNTHESIZER")
    print("=" * 60)
    print(f"out                : {out_root}")
    print(f"size               : {args.size}")
    print(f"per concept        : {args.per_concept}")
    print(f"intensities        : {'255 only' if args.white_only else f'{inten[0]}..{inten[1]}'}")
    print(f"concept groups     : {args.concepts}")
    print(f"random_pool        : {args.random_pool}  (count={args.random_pool_count})")
    print(f"seed               : {args.seed}")
    print("=" * 60)

    concepts: Dict[str, List[np.ndarray]] = {}

    if "all" in args.concepts or "circles" in args.concepts:
        print("\nGenerating circle concepts…")
        concepts.update(gen_circle_set(args.size, args.per_concept, inten, args.white_only))

    if "all" in args.concepts or "squares" in args.concepts:
        print("\nGenerating square concepts…")
        concepts.update(gen_square_set(args.size, args.per_concept, inten, args.white_only))

    if "all" in args.concepts or "crosses" in args.concepts:
        print("\nGenerating cross concepts…")
        concepts.update(gen_cross_set(args.size, args.per_concept, inten, args.white_only))

    print("\nSaving concept folders…")
    save_concepts(concepts, out_root)

    if args.random_pool and args.random_pool_count > 0:
        print("\nGenerating random_pool…")
        save_concepts(gen_random_pool(args.size, args.random_pool_count), out_root)

    total = sum(len(v) for v in concepts.values()) + (args.random_pool_count if args.random_pool else 0)
    print("\nDone.")
    print(f"Concept folders: {len(concepts) + (1 if args.random_pool and args.random_pool_count>0 else 0)}")
    print(f"Total images   : {total}")

if __name__ == "__main__":
    main()
