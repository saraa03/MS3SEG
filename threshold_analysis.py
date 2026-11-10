
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import UNet
from dataset import MS3SEGDataset


# -----------------------------
# Paths and configuration
# -----------------------------
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

DATA_ROOT = ROOT

MODELS = {
    "baseline": OUTPUT_DIR / "baseline" / "best_unet.pt",
    "dice_bce": OUTPUT_DIR / "dice_bce_unet.pt",
}

SPLIT_FILE = OUTPUT_DIR / "patient_split.json"

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]
BATCH_SIZE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def calculate_metrics(tp, fp, fn):
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return dice, precision, recall


def main():
    print(f"Device: {DEVICE}")

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_FILE}\n"
            "Please locate the patient_split.json used in evaluation."
        )

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split = json.load(f)

    val_ids = split["val_ids"]

    dataset = MS3SEGDataset(DATA_ROOT, val_ids)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_results = []

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"\nSkipping {model_name}: checkpoint not found:")
            print(model_path)
            continue

        print(f"\nAnalyzing {model_name}...")
        model = load_checkpoint(UNet(), model_path)

        # Collect probabilities and ground-truth masks once.
        patient_data = {}

        with torch.no_grad():
            for images, masks, patient_ids, slice_indices in loader:
                images = images.to(DEVICE)
                probabilities = torch.sigmoid(model(images)).cpu().numpy()
                masks = masks.cpu().numpy()

                for i, patient_id in enumerate(patient_ids):
                    pid = str(patient_id).zfill(3)

                    if pid not in patient_data:
                        patient_data[pid] = {
                            "probabilities": [],
                            "masks": [],
                        }

                    patient_data[pid]["probabilities"].append(
                        probabilities[i, 0]
                    )
                    patient_data[pid]["masks"].append(masks[i, 0])

        for threshold in THRESHOLDS:
            total_tp = total_fp = total_fn = 0
            patient_dices = []

            for pid, data in patient_data.items():
                probs = np.stack(data["probabilities"])
                gt = np.stack(data["masks"]) > 0.5
                pred = probs >= threshold

                tp = int(np.logical_and(pred, gt).sum())
                fp = int(np.logical_and(pred, ~gt).sum())
                fn = int(np.logical_and(~pred, gt).sum())

                dice, precision, recall = calculate_metrics(tp, fp, fn)

                total_tp += tp
                total_fp += fp
                total_fn += fn

                patient_dices.append({
                    "patient_id": pid,
                    "dice": dice,
                    "precision": precision,
                    "recall": recall,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                })

            pixel_dice, pixel_precision, pixel_recall = calculate_metrics(
                total_tp, total_fp, total_fn
            )
            mean_patient_dice = float(
                np.mean([p["dice"] for p in patient_dices])
            )

            all_results.append({
                "model": model_name,
                "threshold": threshold,
                "pixel_dice": pixel_dice,
                "precision": pixel_precision,
                "recall": pixel_recall,
                "mean_patient_dice": mean_patient_dice,
            })

            print(
                f"Threshold={threshold:.1f} | "
                f"Pixel Dice={pixel_dice:.4f} | "
                f"Precision={pixel_precision:.4f} | "
                f"Recall={pixel_recall:.4f} | "
                f"Mean patient Dice={mean_patient_dice:.4f}"
            )

    if not all_results:
        raise RuntimeError(
            "No models were evaluated. Check the checkpoint paths."
        )

    output_csv = OUTPUT_DIR / "threshold_analysis.csv"

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print("\n========== BEST THRESHOLDS ==========")

    for model_name in MODELS:
        results = [
            r for r in all_results if r["model"] == model_name
        ]
        if not results:
            continue

        best = max(results, key=lambda r: r["mean_patient_dice"])

        print(
            f"{model_name}: threshold={best['threshold']:.1f}, "
            f"mean patient Dice={best['mean_patient_dice']:.4f}, "
            f"pixel Dice={best['pixel_dice']:.4f}, "
            f"precision={best['precision']:.4f}, "
            f"recall={best['recall']:.4f}"
        )

    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main()
