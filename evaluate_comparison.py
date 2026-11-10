
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
DATA_ROOT = ROOT

SPLIT_PATH = OUTPUT_DIR / "patient_split.json"

MODELS = {
    "baseline": OUTPUT_DIR / "baseline" / "best_unet.pt",
    "dice_bce": OUTPUT_DIR / "dice_bce_unet.pt",
}

BATCH_SIZE = 4
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def calculate_counts(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())

    return tp, fp, fn


def metrics_from_counts(tp, fp, fn):
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return dice, precision, recall


def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    # Support both checkpoint files and plain state dictionaries.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = UNet().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def evaluate_model(model_name, model_path, loader):
    print(f"\nEvaluating: {model_name}")
    print("Checkpoint:", model_path)

    model = load_model(model_path)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    patient_counts = {}
    slice_rows = []

    with torch.no_grad():
        for x, y, patient_batch, slice_batch in loader:
            x = x.to(device=DEVICE, dtype=torch.float32)
            y = y.to(device=DEVICE, dtype=torch.float32)

            probabilities = torch.sigmoid(model(x))
            predictions = probabilities >= THRESHOLD
            targets = y >= 0.5

            for i in range(x.size(0)):
                pred = predictions[i, 0].cpu().numpy()
                target = targets[i, 0].cpu().numpy()

                tp, fp, fn = calculate_counts(pred, target)
                dice, precision, recall = metrics_from_counts(
                    tp, fp, fn
                )

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

                slice_rows.append({
                    "model": model_name,
                    "patient_id": patient_id,
                    "slice": slice_index,
                    "dice": dice,
                    "precision": precision,
                    "recall": recall,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                })

    overall_dice, overall_precision, overall_recall = (
        metrics_from_counts(total_tp, total_fp, total_fn)
    )

    patient_rows = []

    for patient_id, counts in sorted(patient_counts.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        dice, precision, recall = metrics_from_counts(
            tp, fp, fn
        )

        patient_rows.append({
            "model": model_name,
            "patient_id": patient_id,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    mean_patient_dice = float(
        np.mean([row["dice"] for row in patient_rows])
    )

    summary = {
        "model": model_name,
        "pixel_dice": overall_dice,
        "pixel_precision": overall_precision,
        "pixel_recall": overall_recall,
        "mean_patient_dice": mean_patient_dice,
        "patients_evaluated": len(patient_rows),
        "slices_evaluated": len(slice_rows),
    }

    print(f"Pixel Dice:         {overall_dice:.4f}")
    print(f"Pixel Precision:    {overall_precision:.4f}")
    print(f"Pixel Recall:       {overall_recall:.4f}")
    print(f"Mean patient Dice:  {mean_patient_dice:.4f}")
    print(f"Patients:           {len(patient_rows)}")
    print(f"Slices:             {len(slice_rows)}")

    return summary, slice_rows, patient_rows


def save_csv(path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Patient split not found: {SPLIT_PATH}"
        )

    with SPLIT_PATH.open("r", encoding="utf-8") as f:
        split = json.load(f)

    train_ids = split["train_ids"]
    val_ids = split["val_ids"]

    if set(train_ids) & set(val_ids):
        raise ValueError(
            "Training and validation patients overlap."
        )

    print("Device:", DEVICE)
    print("Validation patients:", len(val_ids))

    # Construct validation data only once, for both models.
    val_dataset = MS3SEGDataset(DATA_ROOT, val_ids)

    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    summaries = []
    all_slice_rows = []
    all_patient_rows = []

    for model_name, model_path in MODELS.items():
        summary, slice_rows, patient_rows = evaluate_model(
            model_name,
            model_path,
            val_loader,
        )

        summaries.append(summary)
        all_slice_rows.extend(slice_rows)
        all_patient_rows.extend(patient_rows)

    # Save all comparison results separately from the old evaluation.
    summary_path = OUTPUT_DIR / "model_comparison_summary.csv"
    slices_path = OUTPUT_DIR / "model_comparison_slices.csv"
    patients_path = OUTPUT_DIR / "model_comparison_patients.csv"

    save_csv(
        summary_path,
        summaries,
        [
            "model",
            "pixel_dice",
            "pixel_precision",
            "pixel_recall",
            "mean_patient_dice",
            "patients_evaluated",
            "slices_evaluated",
        ],
    )

    save_csv(
        slices_path,
        all_slice_rows,
        [
            "model",
            "patient_id",
            "slice",
            "dice",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
        ],
    )

    save_csv(
        patients_path,
        all_patient_rows,
        [
            "model",
            "patient_id",
            "dice",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
        ],
    )

    print("\n========== MODEL COMPARISON ==========")
    for row in summaries:
        print(f"\nModel: {row['model']}")
        print(f"Pixel Dice: {row['pixel_dice']:.4f}")
        print(f"Precision: {row['pixel_precision']:.4f}")
        print(f"Recall: {row['pixel_recall']:.4f}")
        print(f"Mean patient Dice: {row['mean_patient_dice']:.4f}")

    print("\nComparison files saved:")
    print(summary_path)
    print(slices_path)
    print(patients_path)


if __name__ == "__main__":
    main()
