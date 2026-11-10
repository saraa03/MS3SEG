
from pathlib import Path
import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import UNet
from dataset import MS3SEGDataset


# =========================
# Configuration
# =========================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

MODEL_PATH = OUTPUT_DIR / "best_unet.pt"
SPLIT_PATH = OUTPUT_DIR / "patient_split.json"

DATA_ROOT = ROOT
BATCH_SIZE = 4
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def calculate_metrics(pred, target):
    """Calculate Dice, precision, and recall for binary masks."""

    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return float(dice), float(precision), float(recall)


def main():

    # 1. Check required files
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Patient split file not found: {SPLIT_PATH}\n"
            "This file must contain the original training/validation split."
        )

    # 2. Load the original patient split
    with SPLIT_PATH.open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_ids = split_data["train_ids"]
    val_ids = split_data["val_ids"]

    if set(train_ids) & set(val_ids):
        raise ValueError(
            "Some patients appear in both training and validation sets."
        )

    print("Device:", DEVICE)
    print("Training patients in saved split:", len(train_ids))
    print("Validation patients in saved split:", len(val_ids))

    # 3. Load validation data only
    dataset = MS3SEGDataset(DATA_ROOT, val_ids)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    if len(dataset) == 0:
        raise RuntimeError("The validation dataset is empty.")

    # 4. Load trained model
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = UNet().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded model from:", MODEL_PATH)
    print("Evaluating validation slices...")

    # 5. Accumulate pixel-level metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0

    rows = []

    # Also calculate a Dice score for each complete patient
    patient_counts = {}

    with torch.no_grad():
        for x, y, patient_batch, slice_batch in loader:

            x = x.to(
                device=DEVICE,
                dtype=torch.float32
            )
            y = y.to(
                device=DEVICE,
                dtype=torch.float32
            )

            probabilities = torch.sigmoid(model(x))
            predictions = probabilities >= THRESHOLD
            targets = y >= 0.5

            for i in range(x.size(0)):

                pred = predictions[i, 0].cpu().numpy()
                target = targets[i, 0].cpu().numpy()

                dice, precision, recall = calculate_metrics(
                    pred, target
                )

                tp = int(np.logical_and(pred, target).sum())
                fp = int(np.logical_and(pred, ~target).sum())
                fn = int(np.logical_and(~pred, target).sum())

                total_tp += tp
                total_fp += fp
                total_fn += fn

                patient_id = str(patient_batch[i])
                slice_index = int(slice_batch[i])

                if patient_id not in patient_counts:
                    patient_counts[patient_id] = {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                    }

                patient_counts[patient_id]["tp"] += tp
                patient_counts[patient_id]["fp"] += fp
                patient_counts[patient_id]["fn"] += fn

                rows.append({
                    "patient_id": patient_id,
                    "slice": slice_index,
                    "dice": dice,
                    "precision": precision,
                    "recall": recall,
                    "true_lesion_pixels": int(target.sum()),
                    "predicted_lesion_pixels": int(pred.sum()),
                })

    # 6. Overall pixel-level metrics
    overall_dice = (
        2 * total_tp
        / max(2 * total_tp + total_fp + total_fn, 1)
    )

    overall_precision = (
        total_tp / max(total_tp + total_fp, 1)
    )

    overall_recall = (
        total_tp / max(total_tp + total_fn, 1)
    )

    # 7. Patient-level Dice scores
    patient_dice_scores = []

    for patient_id, counts in patient_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        denominator = 2 * tp + fp + fn

        # If both reference and prediction are empty,
        # define Dice as 1 for that patient.
        if denominator == 0:
            dice = 1.0
        else:
            dice = (2 * tp) / denominator

        patient_dice_scores.append(dice)

    mean_patient_dice = float(
        np.mean(patient_dice_scores)
    )

    # 8. Save slice-level results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "validation_metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fieldnames = [
            "patient_id",
            "slice",
            "dice",
            "precision",
            "recall",
            "true_lesion_pixels",
            "predicted_lesion_pixels",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    # 9. Save patient-level results
    patient_csv_path = (
        OUTPUT_DIR / "patient_level_metrics.csv"
    )

    with patient_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "dice",
                "tp",
                "fp",
                "fn",
            ]
        )

        writer.writeheader()

        for patient_id, counts in sorted(
            patient_counts.items()
        ):
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]

            denominator = 2 * tp + fp + fn
            dice = (
                (2 * tp / denominator)
                if denominator > 0
                else 1.0
            )

            writer.writerow({
                "patient_id": patient_id,
                "dice": dice,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

    # 10. Print results
    print("\n========== VALIDATION RESULTS ==========")
    print("Validation patients evaluated:", len(patient_counts))
    print("Validation slices evaluated:", len(rows))

    print("\nPixel-level metrics:")
    print(f"Dice:      {overall_dice:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall:    {overall_recall:.4f}")

    print("\nPatient-level metric:")
    print(f"Mean patient Dice: {mean_patient_dice:.4f}")

    print("\nSaved results:")
    print("Slice-level:", csv_path)
    print("Patient-level:", patient_csv_path)


if __name__ == "__main__":
    main()
