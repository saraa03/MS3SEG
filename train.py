
from pathlib import Path
import random
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import MS3SEGDataset
from model import UNet


# ---------------- Configuration ----------------

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
THRESHOLD = 0.5

# Save the improved experiment separately.
BEST_MODEL_PATH = OUTPUT_DIR / "dice_bce_unet.pt"
SPLIT_PATH = OUTPUT_DIR / "patient_split.json"
HISTORY_PATH = OUTPUT_DIR / "dice_bce_training_history.csv"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", device)


# ---------------- Find valid patients ----------------

image_root = (
    DATA_ROOT
    / "MS_100_patient_preprocessed"
    / "MS_100_patient_preprocessed"
)

mask_root = (
    DATA_ROOT
    / "MS_100_patient_masks"
    / "MS_100_patient_masks"
    / "abWMH_Masks"
)

print("Image root:", image_root)
print("Image folder exists:", image_root.exists())
print("Mask root:", mask_root)
print("Mask folder exists:", mask_root.exists())

patient_ids = []

for image_path in sorted(image_root.glob("*/")):
    pid = image_path.name

    flair = image_path / f"{pid}_FLAIR.nii.gz"
    mask = mask_root / pid / f"{pid}_abWMH_Mask.nii.gz"

    if flair.exists() and mask.exists():
        patient_ids.append(pid)

print("Matched patients:", len(patient_ids))

if len(patient_ids) < 5:
    raise RuntimeError(
        "Too few matched patients. Check DATA_ROOT and folder names."
    )


# ---------------- Patient split ----------------

train_ids, val_ids = train_test_split(
    patient_ids,
    test_size=0.2,
    random_state=SEED
)

print("Training patients:", len(train_ids))
print("Validation patients:", len(val_ids))

assert set(train_ids).isdisjoint(val_ids)

# Preserve the split from the original experiment if it exists.
# This keeps the validation patients identical across experiments.
if SPLIT_PATH.exists():
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        saved_split = json.load(f)

    saved_train_ids = saved_split.get("train_ids", [])
    saved_val_ids = saved_split.get("val_ids", [])

    if (
        set(saved_train_ids).isdisjoint(saved_val_ids)
        and set(saved_train_ids) | set(saved_val_ids) == set(patient_ids)
        and len(saved_train_ids) > 0
        and len(saved_val_ids) > 0
    ):
        train_ids = saved_train_ids
        val_ids = saved_val_ids
        print("Using existing patient split.")
    else:
        print("Existing split does not match current data; using new split.")
else:
    print("No saved split found; using the generated split.")

assert set(train_ids).isdisjoint(val_ids)

with open(SPLIT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "train_ids": train_ids,
            "val_ids": val_ids
        },
        f,
        indent=2
    )

print("Patient split saved:", SPLIT_PATH)


# ---------------- Datasets and loaders ----------------

train_ds = MS3SEGDataset(DATA_ROOT, train_ids)
val_ds = MS3SEGDataset(DATA_ROOT, val_ids)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print("Training slices:", len(train_ds))
print("Validation slices:", len(val_ds))


# ---------------- Combined BCE + Dice loss ----------------

def dice_bce_loss(logits, targets, smooth=1.0):
    """
    Combined Binary Cross-Entropy and soft Dice loss.

    logits: raw model outputs, before sigmoid
    targets: binary masks with shape [B, 1, H, W]
    """
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(
        logits, targets
    )

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(start_dim=1)
    targets_flat = targets.flatten(start_dim=1)

    intersection = (probabilities * targets_flat).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        + targets_flat.sum(dim=1)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    soft_dice_loss = 1.0 - dice.mean()

    return 0.5 * bce + 0.5 * soft_dice_loss


# ---------------- Model and optimizer ----------------

model = UNet(in_channels=1, base=16).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


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

        total_intersection += (
            pred & target
        ).sum().item()

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

    val_loss = total_loss / total_samples

    return val_loss, val_dice


# ---------------- Training ----------------

best_dice = -1.0
history = []

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    total_samples = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
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

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

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
        "val_dice": val_dice
    })

    # Save only when validation Dice improves.
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
            },
            BEST_MODEL_PATH
        )

        print("Saved best new model:", BEST_MODEL_PATH)

    # Save progress after every epoch.
    with open(
        HISTORY_PATH, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_dice"
            ]
        )
        writer.writeheader()
        writer.writerows(history)


print("\nTraining finished.")
print("Best validation Dice:", round(best_dice, 4))
print("New model:", BEST_MODEL_PATH)
print("Training history:", HISTORY_PATH)
print("Original baseline model was not overwritten.")
