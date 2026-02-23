"""Train TNet for binary circle detection.

Label function from train.py:
    circle_only(c, s, cr) = 1.0 if c > 0 else 0.0

Saves best model (by val loss) to weights/circle_only.pt
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model.model_sq import TNet

# Label function (mirrors train.py style: same signature as ssin/ssum/discrete)
# circle_only(x1, x2, x3) -> [1.0] if x1 > 0 else [0.0]
import numpy as _np


def circle_only(x1=0, x2=0, x3=0):
    return _np.array([float(x1 > 0)], dtype=_np.float32)

# ── Config ────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 128
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3
SAVE_PATH  = "weights/circle_only.pt"
TRAIN_CSV  = "data/aixi_shape/train/dades.csv"
TRAIN_DIR  = "data/aixi_shape/train"
VAL_CSV    = "data/aixi_shape/val/dades.csv"
VAL_DIR    = "data/aixi_shape/val"

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


# ── Train / eval loops ────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Label: circle_only  (1 if circles > 0, 0 otherwise)")

    train_ds = ShapeDataset(TRAIN_CSV, TRAIN_DIR)
    val_ds   = ShapeDataset(VAL_CSV,   VAL_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = TNet(numChannels=1, classes=1, size_img=IMG_SIZE).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{EPOCHS} | train={train_loss:.4f} | "
              f"val={val_loss:.4f} | acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Saved ({SAVE_PATH})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
