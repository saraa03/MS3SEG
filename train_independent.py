from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MS3SEGDataset
from model import UNet


# ---------------- Configuration ----------------

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "outputs" / "independent_split.json"

OUT_DIR = ROOT / "outputs" / "independent_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUT_DIR / "best_unet.pt"
HISTORY_PATH = OUT_DIR / "training_history.csv"
CONFIG_PATH = OUT_DIR / "config.json"

SEED = 20261029
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
THRESHOLD = 0.5

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)


# ---------------- Safety checks ----------------

if not SPLIT_PATH.exists():
    raise FileNotFoundError(f"Split file not found: {SPLIT_PATH}")

if (
    BEST_MODEL_PATH.exists()
    or HISTORY_PATH.exists()
    or CONFIG_PATH.exists()
):
    raise FileExistsError(
        f"Experiment outputs already exist in {OUT_DIR}. "
        "Rename or back up that experiment before running again."
    )

with open(SPLIT_PATH, "r", encoding="utf-8-sig") as f:
    split = json.load(f)

train_ids = [str(pid) for pid in split["train_ids"]]
val_ids = [str(pid) for pid in split["val_ids"]]
test_ids = [str(pid) for pid in split["test_ids"]]

assert len(train_ids) == 70, f"Expected 70 train patients, got {len(train_ids)}"
assert len(val_ids) == 15, f"Expected 15 validation patients, got {len(val_ids)}"
assert len(test_ids) == 15, f"Expected 15 test patients, got {len(test_ids)}"

assert len(set(train_ids)) == 70
assert len(set(val_ids)) == 15
assert len(set(test_ids)) == 15

assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)

all_split_ids = set(train_ids) | set(val_ids) | set(test_ids)
assert len(all_split_ids) == 100, "Expected 100 unique patients."

# Confirm the split covers the available patient folders.
image_root = (
    ROOT / "MS_100_patient_preprocessed"
    / "MS_100_patient_preprocessed"
)
mask_root = (
    ROOT / "MS_100_patient_masks"
    / "MS_100_patient_masks"
    / "abWMH_Masks"
)

available_ids = {
    p.name for p in image_root.iterdir()
    if p.is_dir()
}
mask_ids = {
    p.name for p in mask_root.iterdir()
    if p.is_dir()
}

if available_ids != mask_ids or available_ids != all_split_ids:
    raise RuntimeError(
        "Patient folders do not exactly match the saved split. "
        "Training stopped before loading data."
    )

print("Verified patient split: 70 train / 15 validation / 15 test")
print("Test patients will not be loaded during training.")


# ---------------- Datasets and loaders ----------------

train_ds = MS3SEGDataset(ROOT, train_ids)
val_ds = MS3SEGDataset(ROOT, val_ids)

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    generator=generator,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

print("Training slices:", len(train_ds))
print("Validation slices:", len(val_ds))


# ---------------- BCE + Dice loss ----------------

def dice_bce_loss(logits, targets, smooth=1.0):
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(logits, targets)

    probabilities = torch.sigmoid(logits).flatten(start_dim=1)
    targets_flat = targets.flatten(start_dim=1)

    intersection = (probabilities * targets_flat).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


# ---------------- Validation ----------------

@torch.no_grad()
def validate(model, loader):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_intersection = 0
    total_predicted = 0
    total_target = 0

    for x, y, _, _ in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        logits = model(x)
        loss = dice_bce_loss(logits, y)

        pred = torch.sigmoid(logits) >= THRESHOLD
        target = y >= 0.5

        total_intersection += (pred & target).sum().item()
        total_predicted += pred.sum().item()
        total_target += target.sum().item()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    if total_samples == 0:
        raise RuntimeError("Validation loader is empty.")

    val_dice = (
        2.0 * total_intersection + 1e-6
    ) / (
        total_predicted + total_target + 1e-6
    )

    return total_loss / total_samples, val_dice


# ---------------- Model and optimizer ----------------

# Train a new model from scratch.
# Do not load or overwrite the baseline checkpoint.
model = UNet(in_channels=1, base=16).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

best_dice = -1.0
history = []

config = {
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "threshold_for_model_selection": THRESHOLD,
    "loss": "0.5 * BCE + 0.5 * soft Dice loss",
    "architecture": "UNet(in_channels=1, base=16)",
    "train_patients": len(train_ids),
    "validation_patients": len(val_ids),
    "test_patients": len(test_ids),
    "split_file": str(SPLIT_PATH.relative_to(ROOT)),
    "test_used_during_training": False,
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)


# ---------------- Training loop ----------------

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    total_samples = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
    )

    for x, y, _, _ in progress:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = dice_bce_loss(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0:
        raise RuntimeError("Training loader is empty.")

    train_loss = running_loss / total_samples
    val_loss, val_dice = validate(model, val_loader)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_dice": val_dice,
    })

    if val_dice > best_dice:
        best_dice = val_dice

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_dice": val_dice,
                "threshold": THRESHOLD,
                "seed": SEED,
                "loss": "BCE + Dice",
                "architecture": "UNet(in_channels=1, base=16)",
            },
            BEST_MODEL_PATH,
        )

        print("Saved best new model:", BEST_MODEL_PATH)

    with open(
        HISTORY_PATH, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "train_loss", "val_loss", "val_dice"
            ],
        )
        writer.writeheader()
        writer.writerows(history)


print("\nTraining finished.")
print("Best validation Dice:", round(best_dice, 4))
print("New model:", BEST_MODEL_PATH)
print("Training history:", HISTORY_PATH)
print("Config:", CONFIG_PATH)
print("Original baseline model and original split were not modified.")
