
from pathlib import Path
import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import UNet
from dataset import MS3SEGDataset


# ==========================================
# Configuration
# ==========================================
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

MODEL_PATH = OUTPUT_DIR / "baseline" / "best_unet.pt"
SPLIT_PATH = OUTPUT_DIR / "patient_split.json"

PATIENT_ID = "011"
SLICE_INDEX = 10
THRESHOLDS = [0.5, 0.9]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = UNet()
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def calculate_metrics(tp, fp, fn):
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

    return dice, precision, recall


def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Patient: {PATIENT_ID}")
    print(f"Slice to inspect: {SLICE_INDEX}")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_PATH}"
        )

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    val_ids = [
        str(pid).zfill(3)
        for pid in split["val_ids"]
    ]

    if PATIENT_ID not in val_ids:
        raise ValueError(
            f"Patient {PATIENT_ID} is not in validation."
        )

    # Reuse the existing dataset and validation split.
    dataset = MS3SEGDataset(ROOT, val_ids)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = load_model()

    # Separate aggregate counts for each threshold.
    totals = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in THRESHOLDS
    }

    slice_rows = []
    all_slice_rows = []
    selected_slice_found = False

    with torch.no_grad():
        for batch in loader:
            images, masks, patient_ids, slice_indices = batch

            pid = str(patient_ids[0]).zfill(3)
            slice_idx = int(slice_indices[0])

            if pid != PATIENT_ID:
                continue

            images = images.to(DEVICE).float()
            masks = masks.to(DEVICE).float()

            probabilities = torch.sigmoid(model(images))

            probability = probabilities[0, 0].cpu().numpy()
            target = masks[0, 0].cpu().numpy() >= 0.5

            if slice_idx == SLICE_INDEX:
                selected_slice_found = True

                print("\n--- Selected slice ---")
                print(f"Slice index: {slice_idx}")
                print(f"Image shape: {tuple(images.shape)}")
                print(f"Ground-truth positive pixels: {int(target.sum())}")
                print(
                    "Probability min/max: "
                    f"{probability.min():.6f} / "
                    f"{probability.max():.6f}"
                )
                print(
                    "Probability >= 0.5: "
                    f"{int((probability >= 0.5).sum())}"
                )
                print(
                    "Probability >= 0.9: "
                    f"{int((probability >= 0.9).sum())}"
                )

            for threshold in THRESHOLDS:
                pred = probability >= threshold

                tp = int(np.logical_and(pred, target).sum())
                fp = int(np.logical_and(pred, ~target).sum())
                fn = int(np.logical_and(~pred, target).sum())

                totals[threshold]["tp"] += tp
                totals[threshold]["fp"] += fp
                totals[threshold]["fn"] += fn

                all_slice_rows.append({
                "patient_id": pid,
                "slice_index": slice_idx,
                "threshold": threshold,
                "gt_pixels": int(target.sum()),
                "predicted_pixels": int(pred.sum()),
                "TP": tp,
                "FP": fp,
                "FN": fn,
            })

                if slice_idx == SLICE_INDEX:
                    dice, precision, recall = calculate_metrics(
                        tp, fp, fn
                    )

                    slice_rows.append({
                        "patient_id": pid,
                        "slice_index": slice_idx,
                        "threshold": threshold,
                        "gt_pixels": int(target.sum()),
                        "predicted_pixels": int(pred.sum()),
                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                        "dice": dice,
                        "precision": precision,
                        "recall": recall,
                    })

    if not totals[0.5]["tp"] and not slice_rows:
        raise RuntimeError(
            f"No slices found for patient {PATIENT_ID}. "
            "Check the patient IDs returned by the dataset."
        )

    if not selected_slice_found:
        raise RuntimeError(
            f"Patient {PATIENT_ID}, slice {SLICE_INDEX} "
            "was not found in the dataset."
        )

    print("\n--- Patient-level aggregate results ---")

    for threshold in THRESHOLDS:
        stats = totals[threshold]

        dice, precision, recall = calculate_metrics(
            stats["tp"], stats["fp"], stats["fn"]
        )

        print(f"\nThreshold: {threshold}")
        print(f"TP: {stats['tp']}")
        print(f"FP: {stats['fp']}")
        print(f"FN: {stats['fn']}")
        print(f"Dice: {dice:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

    results_dir = OUTPUT_DIR / "patient_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_slices_csv = (
    results_dir / "patient_011_all_slices_diagnostic.csv"
)

    with open(
        all_slices_csv,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        fields = [
            "patient_id",
            "slice_index",
            "threshold",
            "gt_pixels",
            "predicted_pixels",
            "TP",
            "FP",
            "FN",
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_slice_rows)

    print(f"All-slices report saved to:\n{all_slices_csv}")

    # Show the slices with the most false positives at threshold 0.5.
    ranked = sorted(
        [
            row for row in all_slice_rows
            if row["threshold"] == 0.5
        ],
        key=lambda row: row["FP"],
        reverse=True,
    )

    print("\nTop 10 slices by false-positive pixels (threshold 0.5)")
    print("Slice | GT pixels | Predicted | TP | FP | FN")

    for row in ranked[:10]:
        print(
            f'{row["slice_index"]:5} | '
            f'{row["gt_pixels"]:9} | '
            f'{row["predicted_pixels"]:9} | '
            f'{row["TP"]:2} | '
            f'{row["FP"]:3} | '
            f'{row["FN"]:2}'
        )



    print(f"\nSlice diagnostic saved to:\n{all_slices_csv}")
    print("\nNo model weights or existing files were modified.")


if __name__ == "__main__":
    main()
