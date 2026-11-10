
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import UNet
from dataset import MS3SEGDataset


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_PATH = OUTPUT_DIR / "baseline" / "best_unet.pt"
SPLIT_PATH = OUTPUT_DIR / "patient_split.json"

RESULTS_DIR = OUTPUT_DIR / "patient_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "patient_metrics.csv"

THRESHOLD = 0.5
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. Metrics
# --------------------------------------------------

def calculate_metrics(pred, target):
    """
    pred and target are binary NumPy arrays.
    """

    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    dice_denominator = 2 * tp + fp + fn

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / dice_denominator

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    return dice, precision, recall


# --------------------------------------------------
# 3. Load validation patients
# --------------------------------------------------

with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split = json.load(f)

val_ids = set(str(pid) for pid in split["val_ids"])

print(f"Device: {DEVICE}")
print(f"Validation patients: {len(val_ids)}")


# --------------------------------------------------
# 4. Load the existing dataset
# --------------------------------------------------

# This uses the same dataset class as the existing project.
# If your dataset constructor differs, reuse the exact
# initialization line from compare_predictions.py.

# Use only the validation patients
val_patient_ids = sorted(val_ids)

dataset = MS3SEGDataset(
    root=ROOT_DIR,
    patient_ids=val_patient_ids
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# 5. Load the baseline U-Net
# --------------------------------------------------

model = UNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# --------------------------------------------------
# 6. Evaluate each validation patient
# --------------------------------------------------

patient_data = {}

with torch.no_grad():
    for batch in loader:

        images, masks, patient_ids, slice_indices = batch

        images = images.to(DEVICE).float()
        masks = masks.to(DEVICE).float()

        logits = model(images)

        # The model produces logits, so convert them
        # into probabilities before applying a threshold.
        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).cpu().numpy()

        masks = (masks >= 0.5).cpu().numpy()

        for i, patient_id in enumerate(patient_ids):

            patient_id = str(patient_id)

            if patient_id not in val_ids:
                continue

            if patient_id not in patient_data:
                patient_data[patient_id] = {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "predicted_pixels": 0,
                    "true_pixels": 0,
                    "slices": 0,
                }

            pred = predictions[i].astype(bool)
            target = masks[i].astype(bool)

            stats = patient_data[patient_id]

            stats["tp"] += int(
                np.logical_and(pred, target).sum()
            )
            stats["fp"] += int(
                np.logical_and(pred, ~target).sum()
            )
            stats["fn"] += int(
                np.logical_and(~pred, target).sum()
            )

            stats["predicted_pixels"] += int(pred.sum())
            stats["true_pixels"] += int(target.sum())
            stats["slices"] += 1


# --------------------------------------------------
# 7. Save patient-level results
# --------------------------------------------------

rows = []

for patient_id in sorted(patient_data):

    stats = patient_data[patient_id]

    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]

    dice_denominator = 2 * tp + fp + fn

    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    rows.append({
        "patient_id": patient_id,
        "slices": stats["slices"],
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "predicted_pixels": stats["predicted_pixels"],
        "true_pixels": stats["true_pixels"],
    })


if not rows:
    raise RuntimeError(
        "No validation patients were evaluated. "
        "Check the dataset paths, patient IDs, and dataset initialization."
    )


with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "patient_id",
            "slices",
            "dice",
            "precision",
            "recall",
            "predicted_pixels",
            "true_pixels",
        ],
    )

    writer.writeheader()

    for row in rows:
        writer.writerow({
            "patient_id": row["patient_id"],
            "slices": row["slices"],
            "dice": f'{row["dice"]:.4f}',
            "precision": f'{row["precision"]:.4f}',
            "recall": f'{row["recall"]:.4f}',
            "predicted_pixels": row["predicted_pixels"],
            "true_pixels": row["true_pixels"],
        })


# --------------------------------------------------
# 8. Print summary
# --------------------------------------------------

print("\nPatient-level results")
print("-" * 75)
print(
    f'{"Patient":<12}'
    f'{"Slices":<10}'
    f'{"Dice":<12}'
    f'{"Precision":<14}'
    f'{"Recall":<12}'
)

for row in rows:
    print(
        f'{row["patient_id"]:<12}'
        f'{row["slices"]:<10}'
        f'{row["dice"]:<12.4f}'
        f'{row["precision"]:<14.4f}'
        f'{row["recall"]:<12.4f}'
    )

print("-" * 75)

print(f"Evaluated patients: {len(rows)}")
print(f"Mean patient Dice: {np.mean([r['dice'] for r in rows]):.4f}")
print(f"Mean patient Precision: {np.mean([r['precision'] for r in rows]):.4f}")
print(f"Mean patient Recall: {np.mean([r['recall'] for r in rows]):.4f}")

print(f"\nResults saved to: {CSV_PATH}")
