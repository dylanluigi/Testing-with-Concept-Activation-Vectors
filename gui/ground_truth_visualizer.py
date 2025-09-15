# view_labels_big.py
# Crisp, non-distorting viewer for small images + GT labels.

import argparse, json, random
from pathlib import Path
import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}

def gather_images(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in EXTS])

def resize_preserve_aspect(img, max_w, max_h, scale=None, integer=True, interp=cv2.INTER_NEAREST):
    """
    If scale is None, compute a scale that fits the image inside (max_w, max_h),
    allowing UPSCALING. If integer=True, snap scale to the nearest integer >= 1.
    Returns resized image and the scale used.
    """
    h, w = img.shape[:2]
    if scale is None:
        s_fit = min(max_w / max(w, 1), max_h / max(h, 1))
    else:
        s_fit = float(scale)

    if integer:
        s = max(1, int(round(s_fit)))
    else:
        s = max(1e-6, s_fit)

    new_w, new_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return resized, s

def pad_to_canvas(img, canvas_w, canvas_h, bg=16):
    """Center the image on a canvas (no distortion)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    canvas = np.full((canvas_h, canvas_w, 3), bg, np.uint8)
    y = (canvas_h - h) // 2
    x = (canvas_w - w) // 2
    canvas[y:y+h, x:x+w] = img
    return canvas

def overlay_label(img, text, color=(0, 220, 0)):
    pad, font, scale, thick = 8, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    box = img.copy()
    cv2.rectangle(box, (8, 8), (16 + tw + 2*pad, 16 + th + 2*pad), (0, 0, 0), -1)
    out = cv2.addWeighted(box, 0.35, img, 0.65, 0)
    cv2.putText(out, text, (16 + pad, 16 + th + pad), font, scale, color, thick, cv2.LINE_AA)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5000, help="Use first N images (default 5000)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--autoplay", type=int, default=0, help="ms per image; 0 disables")
    ap.add_argument("--winw", type=int, default=1280)
    ap.add_argument("--winh", type=int, default=900)
    args = ap.parse_args()

    paths = gather_images(args.images)
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)

    n = min(len(paths), len(labels))
    if n == 0:
        raise SystemExit("No images/labels to show.")
    if len(paths) != len(labels):
        print(f"[info] Using first {n} pairs (images={len(paths)} labels={len(labels)})")

    pairs = list(zip(paths[:n], labels[:n]))
    if args.shuffle:
        random.shuffle(pairs)

    # Window + viewing state
    win = "GT Viewer  [←/a prev, →/d next, +/- zoom, i integer, f fullscreen, space pause, r reshuffle, q quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.winw, args.winh)

    idx = 0
    paused = (args.autoplay == 0)
    delay = max(1, args.autoplay) if not paused else 0
    zoom = 1.0            # extra multiplier on top of fit scale
    integer_snap = True   # crisp integer scaling by default
    fullscreen = False

    while True:
        path, y = pairs[idx]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.full((128, 128, 3), 30, np.uint8)
            cv2.putText(img, "Failed to read", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # 1) Fit (with upscaling), 2) apply zoom, 3) integer snap if enabled
        base_fit, s_fit = resize_preserve_aspect(img, args.winw, args.winh,
                                                 scale=None, integer=False, interp=cv2.INTER_NEAREST)
        # recompute with zoom & snap
        target_scale = s_fit * zoom
        vis, _ = resize_preserve_aspect(img, args.winw, args.winh,
                                        scale=target_scale, integer=integer_snap, interp=cv2.INTER_NEAREST)

        canvas = pad_to_canvas(vis, args.winw, args.winh, bg=16)
        label_txt = f"[{idx+1}/{len(pairs)}] {path.name}   GT={int(y)}   scale={'int' if integer_snap else 'free'}×{target_scale:.2f}"
        color = (0, 220, 0) if float(y) >= 0.5 else (0, 170, 255)
        canvas = overlay_label(canvas, label_txt, color=color)
        cv2.imshow(win, canvas)

        k = cv2.waitKeyEx(delay if not paused else 0)

        if k in (ord('q'), 27):
            break
        elif k in (ord('d'), 2555904):   # right
            idx = (idx + 1) % len(pairs)
        elif k in (ord('a'), 2424832):   # left
            idx = (idx - 1) % len(pairs)
        elif k == ord(' '):              # pause/play
            paused = not paused
        elif k in (ord('r'), ord('R')):  # reshuffle
            random.shuffle(pairs); idx = 0
        elif k in (ord('+'), ord('=')):  # zoom in
            zoom *= 1.25
        elif k in (ord('-'), ord('_')):  # zoom out
            zoom = max(0.1, zoom / 1.25)
        elif k in (ord('i'), ord('I')):  # toggle integer scaling
            integer_snap = not integer_snap
        elif k in (ord('f'), ord('F')):  # toggle fullscreen
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, prop)
        else:
            if not paused:
                idx = (idx + 1) % len(pairs)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
