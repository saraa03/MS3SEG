
from pathlib import Path
import csv
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
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

REPORT_DIR = OUTPUT_DIR / "threshold_evaluation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

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

    model = UNet().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def metrics_from_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn

    dice = (
        2 * tp / denominator
        if denominator > 0
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
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(SPLIT_PATH)

    with open(SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)

    val_ids = sorted({
        str(pid).zfill(3)
        for pid in split["val_ids"]
    })

    print(f"Device: {DEVICE}")
    print(f"Validation patients: {len(val_ids)}")
    print(f"Thresholds: {THRESHOLDS}")

    # Reuse the existing validation split and dataset.
    dataset = MS3SEGDataset(ROOT, val_ids)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = load_model()

    # Counts are accumulated over every slice of each patient.
    patient_counts = {
        threshold: {
            pid: {"tp": 0, "fp": 0, "fn": 0, "slices": 0}
            for pid in val_ids
        }
        for threshold in THRESHOLDS
    }

    slice_counts = {
        threshold: {
            pid: {"tp": 0, "fp": 0, "fn": 0, "slices": 0}
            for pid in val_ids
        }
        for threshold in THRESHOLDS
    }

    total_slices = 0

    with torch.no_grad():
        for batch in loader:
            images, masks, patient_ids, slice_indices = batch

            images = images.to(DEVICE).float()
            probabilities = torch.sigmoid(model(images))

            predictions = {
                threshold: (
                    probabilities >= threshold
                ).cpu().numpy().astype(bool)
                for threshold in THRESHOLDS
            }

            targets = (
                masks.cpu().numpy() >= 0.5
            ).astype(bool)

            batch_size = images.shape[0]

            for i in range(batch_size):
                pid = str(patient_ids[i]).zfill(3)

                if pid not in patient_counts[THRESHOLDS[0]]:
                    continue

                total_slices += 1
                target = targets[i]

                for threshold in THRESHOLDS:
                    pred = predictions[threshold][i]

                    tp = int(np.logical_and(pred, target).sum())
                    fp = int(np.logical_and(pred, ~target).sum())
                    fn = int(np.logical_and(~pred, target).sum())

                    for container in (
                        patient_counts[threshold][pid],
                        slice_counts[threshold][pid],
                    ):
                        container["tp"] += tp
                        container["fp"] += fp
                        container["fn"] += fn
                        container["slices"] += 1

    if total_slices == 0:
        raise RuntimeError(
            "No validation slices were evaluated. "
            "Check the dataset and validation IDs."
        )

    # ------------------------------------------
    # Patient-level results
    # ------------------------------------------
    patient_rows = []

    for threshold in THRESHOLDS:
        for pid in val_ids:
            counts = patient_counts[threshold][pid]

            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]

            dice, precision, recall = metrics_from_counts(
                tp, fp, fn
            )

            patient_rows.append({
                "patient_id": pid,
                "threshold": threshold,
                "slices": counts["slices"],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "dice": dice,
                "precision": precision,
                "recall": recall,
            })

    patient_csv = REPORT_DIR / "patient_metrics_by_threshold.csv"

    with open(
        patient_csv,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(patient_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(patient_rows)

    # ------------------------------------------
    # Threshold-level summary
    # ------------------------------------------
    summary_rows = []

    for threshold in THRESHOLDS:
        all_tp = sum(
            patient_counts[threshold][pid]["tp"]
            for pid in val_ids
        )
        all_fp = sum(
            patient_counts[threshold][pid]["fp"]
            for pid in val_ids
        )
        all_fn = sum(
            patient_counts[threshold][pid]["fn"]
            for pid in val_ids
        )

        global_dice, global_precision, global_recall = (
            metrics_from_counts(all_tp, all_fp, all_fn)
        )

        rows = [
            row for row in patient_rows
            if row["threshold"] == threshold
        ]

        summary_rows.append({
            "threshold": threshold,
            "global_TP": all_tp,
            "global_FP": all_fp,
            "global_FN": all_fn,
            "global_dice": global_dice,
            "global_precision": global_precision,
            "global_recall": global_recall,
            "mean_patient_dice": np.mean(
                [r["dice"] for r in rows]
            ),
            "mean_patient_precision": np.mean(
                [r["precision"] for r in rows]
            ),
            "mean_patient_recall": np.mean(
                [r["recall"] for r in rows]
            ),
        })

    summary_csv = REPORT_DIR / "threshold_summary.csv"

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # ------------------------------------------
    # Print summary
    # ------------------------------------------
    print(f"\nEvaluated slices: {total_slices}")
    print("\nThreshold comparison")
    print("-" * 105)

    print(
        f'{"Threshold":<12}'
        f'{"Global Dice":<14}'
        f'{"Global Prec.":<15}'
        f'{"Global Recall":<16}'
        f'{"Mean Pt Dice":<15}'
        f'{"Global FP":<12}'
    )

    for row in summary_rows:
        print(
            f'{row["threshold"]:<12.1f}'
            f'{row["global_dice"]:<14.4f}'
            f'{row["global_precision"]:<15.4f}'
            f'{row["global_recall"]:<16.4f}'
            f'{row["mean_patient_dice"]:<15.4f}'
            f'{row["global_FP"]:<12}'
        )

    print("-" * 105)

    # ------------------------------------------
    # Plot metric curves
    # ------------------------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        constrained_layout=True,
    )

    thresholds = [row["threshold"] for row in summary_rows]

    axes[0].plot(
        thresholds,
        [row["global_dice"] for row in summary_rows],
        marker="o",
        label="Global Dice",
    )
    axes[0].plot(
        thresholds,
        [row["global_precision"] for row in summary_rows],
        marker="o",
        label="Global Precision",
    )
    axes[0].plot(
        thresholds,
        [row["global_recall"] for row in summary_rows],
        marker="o",
        label="Global Recall",
    )
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Global metrics by threshold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        thresholds,
        [row["global_FP"] for row in summary_rows],
        marker="o",
        label="False-positive pixels",
    )
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Pixel count")
    axes[1].set_title("False positives by threshold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure_path = REPORT_DIR / "threshold_comparison.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.show()

    print(f"\nPatient report: {patient_csv}")
    print(f"Threshold summary: {summary_csv}")
    print(f"Comparison plot: {figure_path}")
    print("\nBaseline model weights were not modified.")


if __name__ == "__main__":
    main()
