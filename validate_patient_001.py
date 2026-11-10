
from pathlib import Path
import json
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F

from model import UNet

ROOT = Path(__file__).resolve().parent

IMAGE_PATH = (
    ROOT / "MS_100_patient_preprocessed"
    / "MS_100_patient_preprocessed"
    / "001" / "001_FLAIR.nii.gz"
)

MASK_PATH = (
    ROOT / "MS_100_patient_masks"
    / "MS_100_patient_masks"
    / "abWMH_Masks"
    / "001" / "001_abWMH_Mask.nii.gz"
)

MODEL_PATH = ROOT / "outputs" / "baseline" / "best_unet.pt"
SPLIT_PATH = ROOT / "outputs" / "baseline" / "patient_split.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_volume(volume):
    """Use the same percentile normalization as dataset.py."""
    volume = np.asarray(volume, dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRI, got shape {volume.shape}")

    if not np.isfinite(volume).all():
        raise ValueError("MRI contains NaN or infinite values.")

    brain = volume > 0

    if not brain.any():
        raise ValueError("No positive voxels found in the MRI.")

    lo, hi = np.percentile(volume[brain], [1, 99])
    normalized = np.clip(
        (volume - lo) / max(float(hi - lo), 1e-8),
        0, 1
    )
    normalized[~brain] = 0

    return normalized



def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # The checkpoint stores weights inside "model_state_dict".
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Model checkpoint loaded successfully.")
    if isinstance(checkpoint, dict):
        print("Checkpoint epoch:", checkpoint.get("epoch", "not recorded"))
        print("Checkpoint validation Dice:", checkpoint.get("val_dice", "not recorded"))
        print("Checkpoint threshold:", checkpoint.get("threshold", "not recorded"))

    return model



def predict_slice(model, image_slice):
    """Predict one 2D slice, padding dimensions to multiples of 16."""
    height, width = image_slice.shape
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16

    x = torch.from_numpy(image_slice).float()[None, None].to(DEVICE)

    x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

    with torch.no_grad():
        output = model(x)

        # The U-Net is expected to return logits.
        probability = torch.sigmoid(output)[0, 0]
        probability = probability[:height, :width]

    return probability.cpu().numpy()


def calculate_metrics(reference, prediction):
    reference = reference.astype(bool)
    prediction = prediction.astype(bool)

    tp = np.logical_and(reference, prediction).sum()
    fp = np.logical_and(~reference, prediction).sum()
    fn = np.logical_and(reference, ~prediction).sum()

    dice_denominator = 2 * tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    dice = (
        2 * tp / dice_denominator
        if dice_denominator else 1.0
    )
    precision = (
        tp / precision_denominator
        if precision_denominator else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator else 0.0
    )

    return {
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Reference voxels": int(reference.sum()),
        "Predicted voxels": int(prediction.sum()),
    }


def main():
    for path in (IMAGE_PATH, MASK_PATH, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    image_nii = nib.load(str(IMAGE_PATH))
    mask_nii = nib.load(str(MASK_PATH))

    image_raw = np.asarray(image_nii.dataobj, dtype=np.float32)
    mask_raw = np.asarray(mask_nii.dataobj)

    print("\n=== FILE AND ALIGNMENT CHECK ===")
    print("MRI shape: ", image_raw.shape)
    print("Mask shape:", mask_raw.shape)
    print("MRI affine:\n", image_nii.affine)
    print("Mask affine:\n", mask_nii.affine)

    if image_raw.shape != mask_raw.shape:
        raise ValueError(
            "MRI and mask shapes differ. Stop: do not calculate metrics "
            "until their spatial alignment is resolved."
        )

    if not np.allclose(
        image_nii.affine, mask_nii.affine, rtol=1e-5, atol=1e-4
    ):
        raise ValueError(
            "MRI and mask affine matrices differ. "
            "Spatial alignment must be checked before evaluation."
        )

    image = normalize_volume(image_raw)
    reference = mask_raw > 0

    print("\nReference positive voxels:", int(reference.sum()))
    print("Number of slices:", image.shape[2])

    model = load_model()

    # Predict every slice, then aggregate metrics across the full volume.
    probabilities = np.zeros(image.shape, dtype=np.float32)

    for z in range(image.shape[2]):
        probabilities[:, :, z] = predict_slice(model, image[:, :, z])

        if (z + 1) % 10 == 0 or z == image.shape[2] - 1:
            print(f"Processed slice {z + 1}/{image.shape[2]}")

    rows = []

    for threshold in (0.3, 0.5, 0.7, 0.9):
        prediction = probabilities >= threshold
        metrics = calculate_metrics(reference, prediction)
        rows.append({"Threshold": threshold, **metrics})

    results = pd.DataFrame(rows)

    print("\n=== PATIENT 001: FULL-VOLUME METRICS ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_dir = ROOT / "outputs" / "patient_001_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "patient_001_metrics.csv", index=False
    )

    # Save slice-by-slice results for further inspection.
    slice_rows = []
    for z in range(image.shape[2]):
        for threshold in (0.5, 0.9):
            m = calculate_metrics(
                reference[:, :, z],
                probabilities[:, :, z] >= threshold
            )
            slice_rows.append({
                "Slice": z,
                "Threshold": threshold,
                **m
            })

    pd.DataFrame(slice_rows).to_csv(
        output_dir / "patient_001_slice_metrics.csv", index=False
    )

    print("\nSaved reports to:", output_dir)
    print(
        "\nIMPORTANT: Check patient_split.json to determine whether "
        "patient 001 belongs to training, validation, or test data. "
        "If this patient was used for training, these results are not "
        "an independent test of generalization."
    )


if __name__ == "__main__":
    main()
