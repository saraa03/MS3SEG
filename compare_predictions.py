
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import UNet
from dataset import MS3SEGDataset
from torch.utils.data import DataLoader


# ==========================================
# Configuration
# ==========================================
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

DATA_ROOT = ROOT
MODEL_PATH = OUTPUT_DIR / "baseline" / "best_unet.pt"
SPLIT_PATH = OUTPUT_DIR / "patient_split.json"

THRESHOLDS = [0.5, 0.9]

# Change these to inspect another validation patient/slice.
PATIENT_ID = "011"
SLICE_INDEX = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}\n"
            "Check that baseline/best_unet.pt is the original model."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = UNet()
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def main():
    import json

    print(f"Device: {DEVICE}")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Patient split file not found:\n{SPLIT_PATH}"
        )

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    val_ids = [str(pid).zfill(3) for pid in split["val_ids"]]

    patient_id = str(PATIENT_ID).zfill(3)

    if patient_id not in val_ids:
        raise ValueError(
            f"Patient {patient_id} is not in the validation set.\n"
            f"Available patients: {val_ids}"
        )

    dataset = MS3SEGDataset(DATA_ROOT, val_ids)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = load_model()

    selected_image = None
    selected_mask = None
    selected_probability = None
    selected_slice = None

    with torch.no_grad():
        for batch in loader:
            # Dataset returns four values:
            # image, mask, patient_id, slice_index
            images, masks, patient_ids, slice_indices = batch

            pid = str(patient_ids[0]).zfill(3)
            idx = int(slice_indices[0])

            if pid != patient_id or idx != SLICE_INDEX:
                continue

            images = images.to(DEVICE)
            logits = model(images)
            probabilities = torch.sigmoid(logits)

            selected_image = images[0, 0].cpu().numpy()
            selected_mask = masks[0, 0].cpu().numpy()
            selected_probability = probabilities[0, 0].cpu().numpy()
            selected_slice = idx
            break

    if selected_image is None:
        raise ValueError(
            f"Could not find patient {patient_id}, "
            f"slice index {SLICE_INDEX}."
        )

    ground_truth = selected_mask > 0.5

    # One row for FLAIR/GT and one row per threshold.
    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 9),
        constrained_layout=True,
    )

    # Row 1: original image, ground truth, probability map
    axes[0, 0].imshow(selected_image, cmap="gray")
    axes[0, 0].set_title("FLAIR")

    axes[0, 1].imshow(selected_image, cmap="gray")
    axes[0, 1].imshow(
        ground_truth,
        cmap="Reds",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_title("Ground Truth (red overlay)")

    probability_plot = axes[0, 2].imshow(
        selected_probability,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[0, 2].set_title("Predicted probability")
    fig.colorbar(
        probability_plot,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.04,
    )

    # Row 2: same slice at thresholds 0.5 and 0.9,
    # with the third panel showing their difference.
    predictions = {}

    for threshold in THRESHOLDS:
        predictions[threshold] = selected_probability >= threshold

    axes[1, 0].imshow(selected_image, cmap="gray")
    axes[1, 0].imshow(
        predictions[0.5],
        cmap="Blues",
        alpha=0.6,
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title("Prediction: threshold 0.5")

    axes[1, 1].imshow(selected_image, cmap="gray")
    axes[1, 1].imshow(
        predictions[0.9],
        cmap="Blues",
        alpha=0.6,
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title("Prediction: threshold 0.9")

    # Blue = predicted lesion at 0.5 but not at 0.9.
    # Orange = predicted lesion at both thresholds.
    difference = np.zeros((*selected_image.shape, 3), dtype=np.float32)
    only_low = predictions[0.5] & ~predictions[0.9]
    both = predictions[0.5] & predictions[0.9]

    difference[only_low] = [0.1, 0.5, 1.0]
    difference[both] = [1.0, 0.45, 0.05]

    axes[1, 2].imshow(selected_image, cmap="gray")
    axes[1, 2].imshow(difference, alpha=0.85)
    axes[1, 2].set_title(
        "Threshold difference\nBlue: only at 0.5 | Orange: both"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"Patient {patient_id} | Slice {selected_slice} | "
        "Baseline U-Net",
        fontsize=14,
    )

    plt.show()


if __name__ == "__main__":
    main()
