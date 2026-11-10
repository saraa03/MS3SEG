
from pathlib import Path
import json
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
from torch.utils.data import DataLoader

from model import UNet
from dataset import MS3SEGDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"

MODEL_PATH = OUTPUT / "baseline" / "best_unet.pt"
SPLIT_PATH = OUTPUT / "patient_split.json"

PATIENT_ID = "011"

# The three slices with no reference lesions.
SLICE_INDICES = [5, 8, 9]

THRESHOLDS = [0.5, 0.9]

# Components smaller than this are still reported.
# This is NOT a filtering threshold.
MIN_COMPONENT_SIZE = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ANALYSIS_DIR = OUTPUT / "patient_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Model
# --------------------------------------------------

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


# --------------------------------------------------
# Connected-component analysis
# --------------------------------------------------

def analyze_components(prediction, target, probability):
    """
    Find 8-connected components in a 2D prediction.

    For each predicted component, measure:
    - Area in pixels
    - Number of pixels overlapping the reference mask
    - Number of false-positive pixels
    - Mean and maximum model probability
    - Bounding box

    Components are measured separately at each threshold.
    """

    structure = np.ones((3, 3), dtype=np.uint8)

    labels, component_count = ndimage.label(
        prediction,
        structure=structure,
    )

    components = []

    for component_id in range(1, component_count + 1):
        component = labels == component_id
        area = int(component.sum())

        if area < MIN_COMPONENT_SIZE:
            continue

        true_positive_pixels = int(
            np.logical_and(component, target).sum()
        )

        false_positive_pixels = int(
            np.logical_and(component, ~target).sum()
        )

        rows, cols = np.where(component)

        components.append({
            "component_id": component_id,
            "area": area,
            "TP_pixels": true_positive_pixels,
            "FP_pixels": false_positive_pixels,
            "mean_probability": float(
                probability[component].mean()
            ),
            "max_probability": float(
                probability[component].max()
            ),
            "y_min": int(rows.min()),
            "y_max": int(rows.max()),
            "x_min": int(cols.min()),
            "x_max": int(cols.max()),
        })

    components.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    false_positive_mask = np.logical_and(
        prediction,
        ~target,
    )

    fp_labels, fp_component_count = ndimage.label(
        false_positive_mask,
        structure=structure,
    )

    fp_components = []

    for component_id in range(1, fp_component_count + 1):
        component = fp_labels == component_id
        area = int(component.sum())

        if area < MIN_COMPONENT_SIZE:
            continue

        rows, cols = np.where(component)

        fp_components.append({
            "area": area,
            "mean_probability": float(
                probability[component].mean()
            ),
            "max_probability": float(
                probability[component].max()
            ),
            "y_min": int(rows.min()),
            "y_max": int(rows.max()),
            "x_min": int(cols.min()),
            "x_max": int(cols.max()),
        })

    fp_components.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    return components, fp_components


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_PATH}"
        )

    with open(SPLIT_PATH, encoding="utf-8") as file:
        split = json.load(file)

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
    print(f"Selected slices: {SLICE_INDICES}")
    print("\nLoading selected slices...")

    with torch.no_grad():
        for batch in loader:
            images, masks, patient_ids, slice_indices = batch

            pid = str(patient_ids[0]).zfill(3)
            idx = int(slice_indices[0])

            if pid != PATIENT_ID:
                continue

            if idx not in SLICE_INDICES:
                continue

            images = images.to(DEVICE).float()

            probabilities = torch.sigmoid(model(images))

            image = images[0, 0].cpu().numpy()
            target = masks[0, 0].cpu().numpy() >= 0.5
            probability = probabilities[0, 0].cpu().numpy()

            results[idx] = (
                image,
                target,
                probability,
            )

    missing = sorted(set(SLICE_INDICES) - set(results))

    if missing:
        raise RuntimeError(
            f"Could not find slices: {missing}"
        )

    csv_path = (
        ANALYSIS_DIR
        / "patient_011_components.csv"
    )

    csv_rows = []

    # One row per predicted component.
    for idx in SLICE_INDICES:
        image, target, probability = results[idx]

        print("\n" + "=" * 95)
        print(
            f"Patient {PATIENT_ID} | Slice {idx} | "
            f"Reference pixels: {int(target.sum())}"
        )

        for threshold in THRESHOLDS:
            prediction = probability >= threshold

            components, fp_components = analyze_components(
                prediction,
                target,
                probability,
            )

            predicted_pixels = int(prediction.sum())
            fp_pixels = int(
                np.logical_and(prediction, ~target).sum()
            )
            tp_pixels = int(
                np.logical_and(prediction, target).sum()
            )

            print(f"\nThreshold: {threshold:.1f}")
            print(f"Predicted pixels: {predicted_pixels}")
            print(f"TP pixels: {tp_pixels}")
            print(f"FP pixels: {fp_pixels}")
            print(
                f"Predicted connected components: "
                f"{len(components)}"
            )
            print(
                f"False-positive connected components: "
                f"{len(fp_components)}"
            )

            print("\nLargest false-positive components:")

            if not fp_components:
                print("No false-positive components found.")

            for rank, component in enumerate(
                fp_components[:10],
                start=1,
            ):
                print(
                    f"  #{rank}: "
                    f"area={component['area']} px, "
                    f"mean_prob="
                    f"{component['mean_probability']:.4f}, "
                    f"max_prob="
                    f"{component['max_probability']:.4f}, "
                    f"bbox=(y:{component['y_min']}-"
                    f"{component['y_max']}, "
                    f"x:{component['x_min']}-"
                    f"{component['x_max']})"
                )

            # Save every predicted component, including
            # components that overlap the reference mask.
            for component in components:
                csv_rows.append({
                    "patient_id": PATIENT_ID,
                    "slice": idx,
                    "threshold": threshold,
                    **component,
                })

    # Export component measurements without pandas.
    fieldnames = [
        "patient_id",
        "slice",
        "threshold",
        "component_id",
        "area",
        "TP_pixels",
        "FP_pixels",
        "mean_probability",
        "max_probability",
        "y_min",
        "y_max",
        "x_min",
        "x_max",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # --------------------------------------------------
    # Visualization: show FP components by size
    # --------------------------------------------------

    fig, axes = plt.subplots(
        len(SLICE_INDICES),
        len(THRESHOLDS),
        figsize=(12, 4 * len(SLICE_INDICES)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, idx in enumerate(SLICE_INDICES):
        image, target, probability = results[idx]

        for col, threshold in enumerate(THRESHOLDS):
            prediction = probability >= threshold

            false_positive_mask = np.logical_and(
                prediction,
                ~target,
            )

            labels, count = ndimage.label(
                false_positive_mask,
                structure=np.ones((3, 3), dtype=np.uint8),
            )

            ax = axes[row, col]
            ax.imshow(image, cmap="gray")

            # Overlay only false-positive pixels.
            ax.imshow(
                np.ma.masked_where(
                    ~false_positive_mask,
                    false_positive_mask,
                ),
                cmap="autumn",
                alpha=0.85,
                vmin=0,
                vmax=1,
            )

            ax.set_title(
                f"Slice {idx} | Threshold {threshold:.1f}\n"
                f"FP pixels: {int(false_positive_mask.sum())} | "
                f"FP components: {count}"
            )

            ax.axis("off")

    fig.suptitle(
        f"Patient {PATIENT_ID} | False-positive components",
        fontsize=15,
    )

    figure_path = (
        ANALYSIS_DIR
        / "patient_011_component_analysis.png"
    )

    fig.savefig(
        figure_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()

    print("\n" + "=" * 95)
    print("Analysis completed.")
    print(f"Component table: {csv_path}")
    print(f"Visualization: {figure_path}")
    print(
        "\nNo model weights were changed. "
        "No training was performed."
    )


if __name__ == "__main__":
    main()
