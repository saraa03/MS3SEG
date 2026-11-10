
from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import UNet
from dataset import MS3SEGDataset


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "baseline" / "best_unet.pt"
SPLIT_PATH = ROOT / "outputs" / "patient_split.json"
OUTPUT_DIR = ROOT / "outputs" / "threshold_patient_impact"

THRESHOLDS = [0.5, 0.9]
BATCH_SIZE = 8


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, device):
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_patient_ids(split):
    if "val_ids" in split:
        return split["val_ids"]

    if "val" in split:
        return split["val"]

    if "validation" in split:
        return split["validation"]

    raise KeyError(
        "Could not find validation IDs in patient_split.json. "
        f"Available keys: {list(split.keys())}"
    )


def main():
    import json

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_PATH}")

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    patient_ids = get_patient_ids(split)

    device = get_device()
    print(f"Device: {device}")
    print(f"Validation patients: {len(patient_ids)}")

    dataset = MS3SEGDataset(ROOT, patient_ids)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNet().to(device)
    model = load_checkpoint(model, device)

    # One set of counts per patient and threshold.
    counts = {
        threshold: {}
        for threshold in THRESHOLDS
    }

    with torch.no_grad():
        for batch in loader:
            # This assumes the dataset returns image, mask, patient_id.
            # If your dataset returns a different structure, see the note below.
            images, masks, batch_patient_ids ,_ = batch

            images = images.to(device).float()
            masks = masks.to(device).float()

            logits = model(images)

            # Support models returning probabilities instead of logits.
            if torch.any(logits < 0) or torch.any(logits > 1):
                probabilities = torch.sigmoid(logits)
            else:
                probabilities = logits

            probabilities = probabilities.cpu().numpy()
            masks = masks.cpu().numpy()

            for i, patient_id in enumerate(batch_patient_ids):
                patient_id = str(patient_id)

                gt = masks[i] >= 0.5
                prob = probabilities[i]

                for threshold in THRESHOLDS:
                    pred = prob >= threshold

                    tp = int(np.logical_and(pred, gt).sum())
                    fp = int(np.logical_and(pred, ~gt).sum())
                    fn = int(np.logical_and(~pred, gt).sum())

                    patient_counts = counts[threshold].setdefault(
                        patient_id,
                        {"TP": 0, "FP": 0, "FN": 0, "slices": 0},
                    )

                    patient_counts["TP"] += tp
                    patient_counts["FP"] += fp
                    patient_counts["FN"] += fn
                    patient_counts["slices"] += 1

    def metrics(c):
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        denominator = 2 * tp + fp + fn

        dice = 2 * tp / denominator if denominator else 1.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        return dice, precision, recall

    rows = []

    for patient_id in sorted(counts[0.5]):
        c05 = counts[0.5][patient_id]
        c09 = counts[0.9][patient_id]

        dice05, precision05, recall05 = metrics(c05)
        dice09, precision09, recall09 = metrics(c09)

        rows.append({
            "patient_id": patient_id,
            "slices": c05["slices"],
            "dice_0.5": dice05,
            "dice_0.9": dice09,
            "dice_change": dice09 - dice05,
            "precision_0.5": precision05,
            "precision_0.9": precision09,
            "recall_0.5": recall05,
            "recall_0.9": recall09,
            "FP_0.5": c05["FP"],
            "FP_0.9": c09["FP"],
            "FP_reduction": c05["FP"] - c09["FP"],
            "FN_0.5": c05["FN"],
            "FN_0.9": c09["FN"],
            "FN_increase": c09["FN"] - c05["FN"],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / "patient_threshold_impact.csv"

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in row.items()
            })

    improved = sum(row["dice_change"] > 0.0001 for row in rows)
    worsened = sum(row["dice_change"] < -0.0001 for row in rows)
    unchanged = len(rows) - improved - worsened

    print("\nPatient-level threshold impact")
    print("-" * 88)
    print(
        f"{'Patient':<12} {'Dice@0.5':>10} {'Dice@0.9':>10} "
        f"{'Change':>10} {'FP reduced':>12} {'FN increase':>12}"
    )
    print("-" * 88)

    for row in sorted(rows, key=lambda r: r["dice_change"]):
        print(
            f"{row['patient_id']:<12} "
            f"{row['dice_0.5']:>10.4f} "
            f"{row['dice_0.9']:>10.4f} "
            f"{row['dice_change']:>+10.4f} "
            f"{row['FP_reduction']:>12} "
            f"{row['FN_increase']:>12}"
        )

    print("-" * 88)
    print(f"Patients improved: {improved}")
    print(f"Patients worsened: {worsened}")
    print(f"Patients unchanged: {unchanged}")
    print(f"\nCSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
