
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from model import UNet
from dataset import MS3SEGDataset


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
ANALYSIS_DIR = OUTPUT / "patient_analysis"

MODEL_PATH = OUTPUT / "baseline" / "best_unet.pt"
SPLIT_PATH = OUTPUT / "patient_split.json"

PATIENT_ID = "011"
THRESHOLD = 0.5
MARGIN = 12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (slice index, y_min, y_max, x_min, x_max)
REGIONS = [
    (5, 101, 111, 59, 68),
    (8, 100, 105, 194, 203),
    (8, 149, 153, 198, 208),
    (9, 110, 115, 61, 67),
    (9, 155, 160, 66, 71),
    (9, 98, 101, 106, 110),
]


def load_checkpoint(path):
    try:
        return torch.load(
            path, map_location=DEVICE, weights_only=False
        )
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def normalize_for_display(image):
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1, 99])

    if high <= low:
        return np.zeros_like(image)

    return np.clip((image - low) / (high - low), 0, 1)


def crop_bounds(y0, y1, x0, x1, height, width, margin):
    return (
        max(0, y0 - margin),
        min(height, y1 + margin + 1),
        max(0, x0 - margin),
        min(width, x1 + margin + 1),
    )


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_PATH}")

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    val_ids = split_data.get("val_ids")
    if val_ids is None:
        raise KeyError("Missing 'val_ids' in patient_split.json")

    # Use the existing validation split and dataset.
    dataset = MS3SEGDataset(ROOT, val_ids)

    # dataset.samples stores (image, mask, patient_id, slice_index).
    # Build the lookup using the actual patient ID and z index.
    sample_lookup = {}

    for index, sample in enumerate(dataset.samples):
        _, _, pid, z = sample
        pid = str(pid).zfill(3)
        z = int(z)

        if pid == PATIENT_ID:
            sample_lookup[z] = index

    print(f"Device: {DEVICE}")
    print(f"Patient: {PATIENT_ID}")
    print(f"Samples found for patient: {len(sample_lookup)}")
    print(f"Threshold: {THRESHOLD}")

    if not sample_lookup:
        raise RuntimeError(
            f"Patient {PATIENT_ID} was not found in dataset.samples. "
            "Check patient IDs in patient_split.json."
        )

    checkpoint = load_checkpoint(MODEL_PATH)
    model = UNet().to(DEVICE)
    model.load_state_dict(get_state_dict(checkpoint))
    model.eval()

    # Predict only the slices required for this visual inspection.
    slice_cache = {}

    for slice_idx in sorted({region[0] for region in REGIONS}):
        if slice_idx not in sample_lookup:
            print(f"WARNING: slice {slice_idx} not found; skipping.")
            continue

        sample_index = sample_lookup[slice_idx]
        image_tensor, mask_tensor, pid, z = dataset[sample_index]

        image = image_tensor[0].cpu().numpy()
        mask = mask_tensor[0].cpu().numpy() > 0.5

        with torch.no_grad():
            inputs = image_tensor.unsqueeze(0).to(DEVICE)
            logits = model(inputs)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

        if probability.shape != image.shape:
            raise ValueError(
                f"Shape mismatch at slice {slice_idx}: "
                f"image={image.shape}, probability={probability.shape}"
            )

        slice_cache[slice_idx] = {
            "image": normalize_for_display(image),
            "mask": mask,
            "probability": probability,
        }

        print(
            f"Loaded patient {pid}, slice {z}: "
            f"GT pixels={int(mask.sum())}, "
            f"predicted pixels={int((probability >= THRESHOLD).sum())}"
        )

    valid_regions = [
        r for r in REGIONS if r[0] in slice_cache
    ]

    if not valid_regions:
        raise RuntimeError("None of the requested slices were available.")

    fig, axes = plt.subplots(
        len(valid_regions),
        4,
        figsize=(15, 4 * len(valid_regions)),
        squeeze=False,
    )

    for row, (slice_idx, y0, y1, x0, x1) in enumerate(valid_regions):
        data = slice_cache[slice_idx]

        image = data["image"]
        gt = data["mask"]
        probability = data["probability"]
        prediction = probability >= THRESHOLD

        height, width = image.shape
        cy0, cy1, cx0, cx1 = crop_bounds(
            y0, y1, x0, x1, height, width, MARGIN
        )

        image_crop = image[cy0:cy1, cx0:cx1]
        gt_crop = gt[cy0:cy1, cx0:cx1]
        prob_crop = probability[cy0:cy1, cx0:cx1]
        pred_crop = prediction[cy0:cy1, cx0:cx1]

        axes[row, 0].imshow(image_crop, cmap="gray")
        axes[row, 0].set_title(
            f"Slice {slice_idx} — FLAIR\n"
            f"Region y={y0}-{y1}, x={x0}-{x1}"
        )

        axes[row, 1].imshow(image_crop, cmap="gray")
        if gt_crop.any():
            axes[row, 1].imshow(
                np.ma.masked_where(~gt_crop, gt_crop),
                cmap="Greens",
                alpha=0.8,
                vmin=0,
                vmax=1,
            )
        axes[row, 1].set_title("Ground truth overlay")

        axes[row, 2].imshow(image_crop, cmap="gray")
        axes[row, 2].imshow(
            prob_crop, cmap="magma", alpha=0.65, vmin=0, vmax=1
        )
        axes[row, 2].set_title(
            f"Probability map — max={prob_crop.max():.4f}"
        )

        false_positive = pred_crop & ~gt_crop
        axes[row, 3].imshow(image_crop, cmap="gray")

        if false_positive.any():
            axes[row, 3].imshow(
                np.ma.masked_where(~false_positive, false_positive),
                cmap="Reds",
                alpha=0.85,
                vmin=0,
                vmax=1,
            )

        axes[row, 3].set_title(
            f"False positives — {int(false_positive.sum())} pixels"
        )

        # Mark the original region in cyan in all four panels.
        rect_x = x0 - cx0
        rect_y = y0 - cy0
        rect_w = x1 - x0 + 1
        rect_h = y1 - y0 + 1

        for col in range(4):
            axes[row, col].add_patch(
                Rectangle(
                    (rect_x, rect_y),
                    rect_w,
                    rect_h,
                    linewidth=1.2,
                    edgecolor="cyan",
                    facecolor="none",
                )
            )
            axes[row, col].axis("off")

    fig.suptitle(
        f"Patient {PATIENT_ID}: False-positive region inspection",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_path = (
        ANALYSIS_DIR
        / f"patient_{PATIENT_ID}_false_positive_crops.png"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nVisualization saved to:\n{output_path}")
    print("Original dataset files and model weights were not modified.")


if __name__ == "__main__":
    main()
