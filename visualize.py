
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import UNet


# Project paths
ROOT = Path(__file__).resolve().parent

IMAGE_ROOT = (
    ROOT / "MS_100_patient_preprocessed"
    / "MS_100_patient_preprocessed"
)

MASK_ROOT = (
    ROOT / "MS_100_patient_masks"
    / "MS_100_patient_masks"
    / "abWMH_Masks"
)

MODEL_PATH = ROOT / "outputs" / "best_unet.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5


def load_image(patient_id):
    image_path = (
        IMAGE_ROOT / patient_id / f"{patient_id}_FLAIR.nii.gz"
    )
    mask_path = (
        MASK_ROOT / patient_id / f"{patient_id}_abWMH_Mask.nii.gz"
    )

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    image = np.asarray(
        nib.load(str(image_path)).dataobj,
        dtype=np.float32,
    )
    mask = np.asarray(
        nib.load(str(mask_path)).dataobj,
        dtype=np.float32,
    )

    if image.shape != mask.shape:
        raise ValueError("Image and mask shapes do not match.")

    # Match the normalization used in dataset.py
    brain = image > 0
    if brain.any():
        lo, hi = np.percentile(image[brain], [1, 99])
        image = np.clip(
            (image - lo) / max(float(hi - lo), 1e-8),
            0,
            1,
        )
        image[~brain] = 0

    mask = (mask > 0).astype(np.uint8)

    return image, mask


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = UNet().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    patient_id = input(
        "Patient ID (e.g. 081): "
    ).strip().zfill(3)

    image, mask = load_image(patient_id)

    # Choose a slice that contains lesions if possible
    lesion_counts = mask.sum(axis=(0, 1))
    if lesion_counts.max() > 0:
        slice_index = int(np.argmax(lesion_counts))
    else:
        slice_index = image.shape[2] // 2

    x = torch.from_numpy(
        image[:, :, slice_index].copy()
    )[None, None, :, :].float().to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prediction = probabilities >= THRESHOLD

    ground_truth = mask[:, :, slice_index]
    dice = (
        2 * np.logical_and(prediction, ground_truth).sum()
        / max(prediction.sum() + ground_truth.sum(), 1)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image[:, :, slice_index].T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR MRI")

    axes[1].imshow(image[:, :, slice_index].T, cmap="gray", origin="lower")
    axes[1].imshow(
        np.ma.masked_where(ground_truth.T == 0, ground_truth.T),
        cmap="Greens",
        alpha=0.7,
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Ground Truth (Green)")

    axes[2].imshow(image[:, :, slice_index].T, cmap="gray", origin="lower")
    axes[2].imshow(
        np.ma.masked_where(~prediction.T, prediction.T),
        cmap="Reds",
        alpha=0.7,
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("U-Net Prediction (Red)")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Patient {patient_id} | Slice {slice_index} | "
        f"Slice Dice: {dice:.4f}"
    )
    plt.tight_layout()
    plt.show()

    print(f"Patient: {patient_id}")
    print(f"Slice: {slice_index}")
    print(f"Slice Dice: {dice:.4f}")
    print(f"Ground-truth lesion pixels: {int(ground_truth.sum())}")
    print(f"Predicted lesion pixels: {int(prediction.sum())}")


if __name__ == "__main__":
    main()
