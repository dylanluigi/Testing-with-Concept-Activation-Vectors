"""Progressive complexity search: FlexibleTNet for binary circle detection.

Trains stages in order of increasing complexity, stopping at the FIRST
architecture that achieves >= TARGET_ACC on the validation set.

Complexity ladder (each stage is a fresh model):
  Stage 1 │ conv=[8]                    fc=[32]            tiny
  Stage 2 │ conv=[16, 32]               fc=[64]            small
  Stage 3 │ conv=[16, 32, 64]           fc=[128, 64]       medium-small
  Stage 4 │ conv=[25, 35, 50]           fc=[256, 128]      medium
  Stage 5 │ conv=[25, 35, 50, 75]       fc=[500, 250]      large
  Stage 6 │ conv=[25, 35, 50, 75, 125]  fc=[500, 250, 50]  full TNet

Label (from train.py): circle_only(c, s, cr) = 1.0 if c > 0 else 0.0
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model.model_dylan import FlexibleTNet

# ── Config ────────────────────────────────────────────────────────────────
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE         = 128
BATCH_SIZE       = 64
LR               = 1e-3
TARGET_ACC       = 0.99
EPOCHS_PER_STAGE = 30   # max epochs given to each stage
PATIENCE         = 8    # early-stop within a stage if acc stops improving
DROPOUT_P        = 0.2
SAVE_PATH        = "weights/flex_circle_only.pt"
TRAIN_CSV        = "data/aixi_shape/train/dades.csv"
TRAIN_DIR        = "data/aixi_shape/train"
VAL_CSV          = "data/aixi_shape/val/dades.csv"
VAL_DIR          = "data/aixi_shape/val"

# Ordered from simplest to most complex
STAGES = [
    {"conv": [8],                   "fc": [32],           "name": "Stage 1 — tiny          (1 conv,  1 FC)"},
    {"conv": [16, 32],              "fc": [64],           "name": "Stage 2 — small         (2 conv,  1 FC)"},
    {"conv": [16, 32, 64],          "fc": [128, 64],      "name": "Stage 3 — medium-small  (3 conv,  2 FC)"},
    {"conv": [25, 35, 50],          "fc": [256, 128],     "name": "Stage 4 — medium        (3 conv,  2 FC)"},
    {"conv": [25, 35, 50, 75],      "fc": [500, 250],     "name": "Stage 5 — large         (4 conv,  2 FC)"},
    {"conv": [25, 35, 50, 75, 125], "fc": [500, 250, 50], "name": "Stage 6 — full TNet     (5 conv,  3 FC)"},
]

# ── Label function (mirrors train.py style) ───────────────────────────────
def circle_only(x1=0, x2=0, x3=0):
    """1.0 if circles present (x1 > 0), 0.0 otherwise."""
    return np.array([float(x1 > 0)], dtype=np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class ShapeDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str):
        self.img_dir = Path(img_dir)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f, delimiter=";"))
        self.samples = [
            (int(r[""]), circle_only(int(r["c"]), int(r["s"]), int(r["cr"]))[0])
            for r in rows
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, label = self.samples[idx]
        img = Image.open(self.img_dir / f"{img_idx:05d}.png").convert("L")
        x = _transform(img)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y


# ── Train / eval ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        total_loss += criterion(pred, y).item() * len(x)
        correct += ((pred > 0.5).float() == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Stage runner ──────────────────────────────────────────────────────────
def run_stage(stage, train_loader, val_loader):
    """Train one stage. Returns (best_acc, model_state_dict | None).

    Returns the saved state dict if TARGET_ACC was reached, else None.
    """
    model = FlexibleTNet(
        num_channels=1,
        num_classes=1,
        conv_channels=stage["conv"],
        fc_layers=stage["fc"],
        dropout_p=DROPOUT_P,
        do_sigmoid=True,
        size_img=IMG_SIZE,
    ).to(DEVICE)

    n_params = count_params(model)
    print(f"\n{'='*60}")
    print(f"  {stage['name']}")
    print(f"  conv={stage['conv']}  fc={stage['fc']}  params={n_params:,}")
    print(f"{'='*60}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_acc      = 0.0
    best_state    = None
    epochs_no_imp = 0

    for epoch in range(1, EPOCHS_PER_STAGE + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)

        flag = ""
        if val_acc > best_acc:
            best_acc      = val_acc
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_imp = 0
            flag          = " ◀ best"
        else:
            epochs_no_imp += 1

        print(f"  ep {epoch:02d}/{EPOCHS_PER_STAGE} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | acc={val_acc:.4f}{flag}")

        if val_acc >= TARGET_ACC:
            print(f"  ✓ Target {TARGET_ACC:.0%} reached!")
            return best_acc, best_state

        if epochs_no_imp >= PATIENCE:
            print(f"  ✗ No improvement for {PATIENCE} epochs — moving to next stage.")
            break

    return best_acc, None   # target not reached


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print(f"Device     : {DEVICE}")
    print(f"Target acc : {TARGET_ACC:.0%}")
    print(f"Max epochs / stage : {EPOCHS_PER_STAGE}  (patience={PATIENCE})")
    print(f"Stages     : {len(STAGES)}")

    train_ds = ShapeDataset(TRAIN_CSV, TRAIN_DIR)
    val_ds   = ShapeDataset(VAL_CSV,   VAL_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    for i, stage in enumerate(STAGES, 1):
        best_acc, state = run_stage(stage, train_loader, val_loader)

        if state is not None:
            torch.save(state, SAVE_PATH)
            print(f"\n{'='*60}")
            print(f"  DONE — winning architecture: {stage['name']}")
            print(f"  Val acc : {best_acc:.4f}")
            print(f"  Saved   : {SAVE_PATH}")
            print(f"{'='*60}")
            return

        print(f"  Stage {i} best acc: {best_acc:.4f} — insufficient, growing network.")

    print(f"\nAll stages exhausted without reaching {TARGET_ACC:.0%}. "
          f"Consider adding more stages or tuning hyperparameters.")


if __name__ == "__main__":
    main()
