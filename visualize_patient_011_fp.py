
from pathlib import Path
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import UNet
from dataset import MS3SEGDataset


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"

MODEL_PATH = OUTPUT / "baseline" / "best_unet.pt"
SPLIT_PATH = OUTPUT / "patient_split.json"

PATIENT_ID = "011"
SLICE_INDICES = [5, 8, 9, 12, 11]
THRESHOLDS = [0.5, 0.9]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
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


def calculate_metrics(prediction, target):
    """Calculate pixel-level metrics for one slice."""
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())

    pred_count = int(prediction.sum())
    gt_count = int(target.sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    dice_denominator = 2 * tp + fp + fn
    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    return {
        "GT": gt_count,
        "Pred": pred_count,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "Dice": dice,
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Patient split file not found: {SPLIT_PATH}"
        )

    with open(SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)

    val_ids = [
        str(pid).zfill(3)
        for pid in split["val_ids"]
    ]

    if PATIENT_ID not in val_ids:
        raise ValueError(
            f"Patient {PATIENT_ID} is not in validation."
        )

    dataset = MS3SEGDataset(ROOT, val_ids)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = load_model()
    results = {}

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Patient: {PATIENT_ID}")
    print(f"Thresholds: {THRESHOLDS}")
    print("\nCollecting selected slices...")

    with torch.no_grad():
        for batch in loader:
            images, masks, patient_ids, slice_indices = batch

            pid = str(patient_ids[0]).zfill(3)
            idx = int(slice_indices[0])

            if pid != PATIENT_ID or idx not in SLICE_INDICES:
                continue

            images = images.to(DEVICE).float()
            probabilities = torch.sigmoid(model(images))

            image = images[0, 0].cpu().numpy()
            target = masks[0, 0].cpu().numpy() >= 0.5
            probability = probabilities[0, 0].cpu().numpy()

            results[idx] = (image, target, probability)

    missing = sorted(set(SLICE_INDICES) - set(results))

    if missing:
        raise RuntimeError(
            f"Could not find slices {missing} "
            f"for patient {PATIENT_ID}."
        )

    # Display numerical results for both thresholds.
    print("\nPatient-level slice investigation")
    print("=" * 100)

    header = (
        f"{'Slice':<8}"
        f"{'Threshold':<12}"
        f"{'GT':<8}"
        f"{'Pred':<8}"
        f"{'TP':<8}"
        f"{'FP':<8}"
        f"{'FN':<8}"
        f"{'Precision':<13}"
        f"{'Recall':<11}"
        f"{'Dice':<10}"
        f"{'MaxProb':<10}"
    )
    print(header)
    print("-" * 100)

    for idx in SLICE_INDICES:
        image, target, probability = results[idx]

        for threshold in THRESHOLDS:
            prediction = probability >= threshold
            metrics = calculate_metrics(prediction, target)

            print(
                f"{idx:<8}"
                f"{threshold:<12.1f}"
                f"{metrics['GT']:<8}"
                f"{metrics['Pred']:<8}"
                f"{metrics['TP']:<8}"
                f"{metrics['FP']:<8}"
                f"{metrics['FN']:<8}"
                f"{metrics['Precision']:<13.4f}"
                f"{metrics['Recall']:<11.4f}"
                f"{metrics['Dice']:<10.4f}"
                f"{probability.max():<10.4f}"
            )

    print("=" * 100)

    # Plot the original five-panel comparison.
    fig, axes = plt.subplots(
        len(SLICE_INDICES),
        5,
        figsize=(17, 4 * len(SLICE_INDICES)),
        constrained_layout=True,
    )

    for row, idx in enumerate(SLICE_INDICES):
        image, target, probability = results[idx]

        pred05 = probability >= 0.5
        pred09 = probability >= 0.9

        panels = [
            ("FLAIR", image),
            ("Ground Truth", target),
            ("Probability", probability),
            ("Prediction 0.5", pred05),
            ("Prediction 0.9", pred09),
        ]

        for col, (title, data) in enumerate(panels):
            ax = axes[row, col]

            # Keep the original FLAIR image as the background.
            ax.imshow(image, cmap="gray")

            if title == "Ground Truth":
                ax.imshow(
                    target,
                    cmap="Reds",
                    alpha=0.65,
                    vmin=0,
                    vmax=1,
                )

            elif title == "Probability":
                ax.imshow(
                    probability,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                )

            elif title.startswith("Prediction"):
                ax.imshow(
                    data,
                    cmap="Blues",
                    alpha=0.7,
                    vmin=0,
                    vmax=1,
                )

            if row == 0:
                ax.set_title(title)

            if col == 0:
                ax.set_ylabel(
                    f"Slice {idx}",
                    fontsize=12,
                )

            ax.axis("off")

    fig.suptitle(
        f"Patient {PATIENT_ID} | False-positive investigation",
        fontsize=15,
    )

    output_path = (
        OUTPUT
        / "patient_analysis"
        / "patient_011_fp_visualization.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    print(f"\nFigure saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
