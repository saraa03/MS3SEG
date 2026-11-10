import tempfile
import nibabel as nib

from pathlib import Path
import json
import csv
import io

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from model import UNet
from dataset import MS3SEGDataset

from PIL import Image
# =========================================================
# Configuration
# =========================================================
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "baseline" / "best_unet.pt"
SPLIT_PATH = ROOT / "outputs" / "baseline" / "patient_split.json"
REPORT_DIR = ROOT / "outputs" / "streamlit_reports"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="MS3SEG | MRI Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Styling
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(100, 116, 139, 0.08);
        border: 1px solid rgba(100, 116, 139, 0.20);
        padding: 16px;
        border-radius: 12px;
    }
    .small-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Model and data loading
# =========================================================
@st.cache_resource(show_spinner="Loading U-Net model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = UNet().to(DEVICE)

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_validation_ids():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_PATH}")

    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split = json.load(file)

    for key in ("val_ids", "val", "validation"):
        if key in split:
            return [str(pid) for pid in split[key]]

    raise KeyError(
        f"Validation IDs not found. Available keys: {list(split.keys())}"
    )


@st.cache_resource(show_spinner="Loading validation MRI slices...")
def load_dataset():
    patient_ids = load_validation_ids()
    return MS3SEGDataset(ROOT, patient_ids)


def get_sigmoid_probabilities(model, images):
    logits = model(images)
    return torch.sigmoid(logits)


def calculate_metrics(gt, pred):
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())

    denom = 2 * tp + fp + fn
    dice = (2 * tp / denom) if denom else 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }

def calculate_metrics_from_counts(counts):
    tp = int(counts.get("TP", 0))
    fp = int(counts.get("FP", 0))
    fn = int(counts.get("FN", 0))

    denom = 2 * tp + fp + fn
    dice = 2 * tp / denom if denom else 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
    }

@st.cache_data(show_spinner="Evaluating validation predictions...")
def evaluate_validation(_model, _dataset, batch_size=8):
    # The underscore-prefixed parameters are intentionally excluded
    # from Streamlit's cache hashing. Clear this cache after model changes.
    loader = DataLoader(
        _dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    results = []

    with torch.no_grad():
        for images, masks, patient_ids, slice_indices in loader:
            images = images.to(DEVICE).float()
            probabilities = get_sigmoid_probabilities(
                _model, images
            ).cpu().numpy()

            masks = masks.numpy()

            for i in range(len(patient_ids)):
                results.append({
                    "patient_id": str(patient_ids[i]),
                    "slice_index": int(slice_indices[i]),
                    "image": images[i, 0].cpu().numpy().astype(np.float32),
                    "mask": masks[i, 0].astype(np.uint8),
                    "probability": probabilities[i, 0].astype(np.float32),
                })

    return results


# =========================================================
# Header and sidebar
# =========================================================
st.title("🧠 MS3SEG")
st.subheader("Multiple Sclerosis Lesion Segmentation Research Dashboard")
st.caption(
    "U-Net · MRI FLAIR · Patient-level analysis · Threshold evaluation"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Patient Explorer",
        "New MRI Analysis",
        "Threshold Analysis",
        "Error Analysis",
        "Export Reports",
    ],
)

st.sidebar.divider()
st.sidebar.write(f"**Device:** `{DEVICE}`")
st.sidebar.write("**Model:** Baseline U-Net")
st.sidebar.write("**Input:** 2D FLAIR slices")
st.sidebar.caption("Research use only — not for clinical diagnosis.")

try:
    model = load_model()
    dataset = load_dataset()
except Exception as exc:
    st.error(f"Could not initialize the project: {exc}")
    st.info(
        "Check the model path, patient_split.json, extracted data folders, "
        "and installed dependencies."
    )
    st.stop()


# =========================================================
# Patient Explorer: identify samples by patient and slice
# =========================================================
if page == "Patient Explorer":
    st.header("Patient Explorer")
    st.write(
        "Inspect an MRI slice, its reference lesion mask, "
        "and the model's predicted lesion probability."
    )

    patient_ids = sorted(set(str(s[2]) for s in dataset.samples))
    selected_patient = st.selectbox("Patient ID", patient_ids)

    patient_sample_indices = [
        i for i, sample in enumerate(dataset.samples)
        if str(sample[2]) == selected_patient
    ]

    selected_index = st.select_slider(
        "Slice index",
        options=list(range(len(patient_sample_indices))),
        value=0,
        format_func=lambda x: str(dataset.samples[patient_sample_indices[x]][3]),
    )

    sample_index = patient_sample_indices[selected_index]
    image_tensor, mask_tensor, patient_id, slice_index = dataset[sample_index]

    with torch.no_grad():
        probability = get_sigmoid_probabilities(
            model, image_tensor.unsqueeze(0).to(DEVICE)
        )[0, 0].cpu().numpy()

    threshold = st.slider(
        "Prediction threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="explorer_threshold",
    )

    image = image_tensor[0].numpy()
    gt = mask_tensor[0].numpy() >= 0.5
    prediction = probability >= threshold

    metrics = calculate_metrics(gt, prediction)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dice", f"{metrics['Dice']:.4f}")
    c2.metric("Precision", f"{metrics['Precision']:.4f}")
    c3.metric("Recall", f"{metrics['Recall']:.4f}")
    c4.metric("False Positives", f"{metrics['FP']:,}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("FLAIR MRI")

    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reference mask")

    axes[2].imshow(probability, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Predicted probability")

    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(
        prediction,
        cmap="autumn",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[3].set_title(f"Prediction ≥ {threshold:.2f}")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"Patient {patient_id} · Slice {slice_index} · "
        f"Ground-truth pixels: {int(gt.sum()):,} · "
        f"Predicted pixels: {int(prediction.sum()):,}"
    )


# =========================================================
# New MRI Analysis
# =========================================================

elif page == "New MRI Analysis":
    st.header("New MRI Analysis")
    st.write(
        "Upload a 2D MRI image (PNG/JPG) or a 3D FLAIR "
        "MRI volume (NIfTI) to inspect the U-Net prediction."
    )

    st.info(
        "Research prototype only. Predictions are not a medical "
        "diagnosis. Use authorized data only."
    )

    uploaded_file = st.file_uploader(
        "Upload MRI: PNG, JPG, JPEG, NII, or NII.GZ",
        type=["png", "jpg", "jpeg", "nii", "gz"],
        key="new_mri_upload",
    )

    if uploaded_file is not None:
        filename = uploaded_file.name.lower()

        try:
            # -----------------------------------------------
            # Load PNG / JPG as a single 2D slice
            # -----------------------------------------------
            if filename.endswith((".png", ".jpg", ".jpeg")):
                with Image.open(
                    io.BytesIO(uploaded_file.getvalue())
                ) as pil_image:
                    image_2d = np.asarray(
                        pil_image.convert("L"),
                        dtype=np.float32,
                    ).copy()

                if image_2d.ndim != 2 or image_2d.size == 0:
                    st.error("The uploaded image is empty or invalid.")
                    st.stop()

                if not np.isfinite(image_2d).all():
                    st.error("The image contains invalid pixel values.")
                    st.stop()

                # Normalize image intensities robustly.
                positive = image_2d > 0
                if not positive.any():
                    st.error("The uploaded image is completely black.")
                    st.stop()

                lo, hi = np.percentile(image_2d[positive], [1, 99])
                if hi <= lo:
                    st.error("The image has insufficient intensity variation.")
                    st.stop()

                image_2d = np.clip(
                    (image_2d - lo) / max(float(hi - lo), 1e-8),
                    0,
                    1,
                )
                image_2d[~positive] = 0

                volume = image_2d[:, :, None]
                st.caption(
                    "2D image mode: one slice. PNG/JPG files do not "
                    "contain NIfTI spatial metadata."
                )

            # -----------------------------------------------
            # Load NIfTI volume
            # -----------------------------------------------
            elif filename.endswith((".nii", ".nii.gz")):
                suffix = (
                    ".nii.gz"
                    if filename.endswith(".nii.gz")
                    else ".nii"
                )

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:
                    nii = nib.load(temp_path)
                    volume = np.asarray(
                        nii.dataobj,
                        dtype=np.float32,
                    ).copy()
                finally:
                    Path(temp_path).unlink(missing_ok=True)

                if volume.ndim != 3:
                    st.error(
                        f"Expected a 3D NIfTI volume; got {volume.ndim}D."
                    )
                    st.stop()

                if not np.isfinite(volume).all():
                    st.error("The MRI volume contains invalid values.")
                    st.stop()

                brain = volume > 0
                if not brain.any():
                    st.error(
                        "No positive-valued voxels found. Check the "
                        "uploaded preprocessed FLAIR volume."
                    )
                    st.stop()

                lo, hi = np.percentile(volume[brain], [1, 99])
                if hi <= lo:
                    st.error("The volume has insufficient intensity variation.")
                    st.stop()

                volume = np.clip(
                    (volume - lo) / max(float(hi - lo), 1e-8),
                    0,
                    1,
                )
                volume[~brain] = 0

            else:
                st.error(
                    "Unsupported format. Upload PNG, JPG, JPEG, "
                    ".nii, or .nii.gz."
                )
                st.stop()

            # -----------------------------------------------
            # Select slice and run inference
            # -----------------------------------------------
            depth = volume.shape[2]

            st.success(
                f"Image loaded: {volume.shape[0]} × "
                f"{volume.shape[1]} × {depth}"
            )

            
            if depth == 1:
                slice_index = 0
                st.caption("Single 2D image: one slice available.")
            else:
                slice_index = st.slider(
                    "MRI slice index",
                    min_value=0,
                    max_value=depth - 1,
                    value=depth // 2,
                    key="new_mri_slice",
                )


            threshold = st.slider(
                "Prediction threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                key="new_mri_threshold",
            )

            image = volume[:, :, slice_index]
            height, width = image.shape

            # Pad to dimensions compatible with U-Net.
            pad_h = (16 - height % 16) % 16
            pad_w = (16 - width % 16) % 16

            padded_image = np.pad(
                image,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
            )

            input_tensor = torch.from_numpy(
                padded_image[None, None, :, :].copy()
            ).float().to(DEVICE)

            model = load_model()

            with torch.no_grad():
                probability = get_sigmoid_probabilities(
                    model, input_tensor
                )[0, 0].cpu().numpy()

            probability = probability[:height, :width]
            prediction = probability >= threshold

            c1, c2, c3 = st.columns(3)
            c1.metric("Slice", f"{slice_index} / {depth - 1}")
            c2.metric("Predicted pixels", f"{int(prediction.sum()):,}")
            c3.metric(
                "Predicted area fraction",
                f"{100 * prediction.mean():.3f}%",
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image.T, cmap="gray", origin="lower")
            axes[0].set_title("MRI image")

            axes[1].imshow(
                probability.T,
                cmap="magma",
                vmin=0,
                vmax=1,
                origin="lower",
            )
            axes[1].set_title("Predicted probability")

            axes[2].imshow(image.T, cmap="gray", origin="lower")
            axes[2].imshow(
                prediction.T,
                cmap="autumn",
                alpha=0.55,
                vmin=0,
                vmax=1,
                origin="lower",
            )
            axes[2].set_title(f"Prediction ≥ {threshold:.2f}")

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Experimental segmentation only. Predictions may contain "
                "false positives or miss lesions. No ground-truth mask "
                "was supplied, so Dice, precision, and recall are unavailable."
            )

        except Exception as exc:
            st.error(f"Could not analyze this MRI file: {exc}")

# =========================================================
# Overview
# =========================================================
elif page == "Overview":
    st.header("Project Overview")

    st.markdown(
        """
        This dashboard explores a baseline U-Net segmentation model
        on the project's validation MRI slices. It supports visual review,
        threshold comparisons, and patient-level error analysis.
        """
    )

    with st.spinner("Preparing validation summary..."):
        results = evaluate_validation(model, dataset)

    patient_count = len(set(r["patient_id"] for r in results))
    slice_count = len(results)
    lesion_slices = sum(int(r["mask"].sum() > 0) for r in results)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Validation patients", f"{patient_count}")
    col2.metric("MRI slices", f"{slice_count}")
    col3.metric("Slices with lesions", f"{lesion_slices}")
    col4.metric("Device", str(DEVICE).upper())

    rows = []
    for threshold in (0.5, 0.7, 0.9):
        totals = {"TP": 0, "FP": 0, "FN": 0}
        patient_metrics = {}

        for r in results:
            m = calculate_metrics(r["mask"], r["probability"] >= threshold)
            for key in totals:
                totals[key] += m[key]

            p = patient_metrics.setdefault(
                r["patient_id"], {"TP": 0, "FP": 0, "FN": 0}
            )
            for key in p:
                p[key] += m[key]

        global_metrics = calculate_metrics_from_counts(totals)
        patient_dices = [
            calculate_metrics_from_counts(counts)["Dice"]
            for counts in patient_metrics.values()
        ]

        rows.append({
            "Threshold": threshold,
            "Global Dice": global_metrics["Dice"],
            "Global Precision": global_metrics["Precision"],
            "Global Recall": global_metrics["Recall"],
            "Mean Patient Dice": float(np.mean(patient_dices)),
            "Global FP": totals["FP"],
        })

    st.subheader("Baseline threshold summary")
    overview_df = pd.DataFrame(rows)
    st.dataframe(
        overview_df.style.format({
            "Threshold": "{:.1f}",
            "Global Dice": "{:.4f}",
            "Global Precision": "{:.4f}",
            "Global Recall": "{:.4f}",
            "Mean Patient Dice": "{:.4f}",
            "Global FP": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Thresholds are evaluated on the validation set. Keep the test set "
        "independent for the final evaluation."
    )


# =========================================================
# Metric helper for pooled counts
# =========================================================
# Defined before use at runtime by Python after module execution.
# The pages are rendered only after all top-level definitions have run.


# =========================================================
# Threshold Analysis
# =========================================================
elif page == "Threshold Analysis":
    st.header("Threshold Analysis")
    st.write(
        "Compare probability thresholds across all validation slices "
        "and patients."
    )

    thresholds = st.multiselect(
        "Thresholds to compare",
        options=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        default=[0.5, 0.7, 0.9],
    )

    if st.button("Evaluate selected thresholds", type="primary"):
        if not thresholds:
            st.warning("Select at least one threshold.")
        else:
            results = evaluate_validation(model, dataset)
            rows = []

            for threshold in thresholds:
                total_counts = {"TP": 0, "FP": 0, "FN": 0}
                by_patient = {}

                for r in results:
                    m = calculate_metrics(
                        r["mask"], r["probability"] >= threshold
                    )
                    for key in total_counts:
                        total_counts[key] += m[key]

                    p = by_patient.setdefault(
                        r["patient_id"],
                        {"TP": 0, "FP": 0, "FN": 0},
                    )
                    for key in p:
                        p[key] += m[key]

                gm = calculate_metrics_from_counts(total_counts)
                mean_patient_dice = np.mean([
                    calculate_metrics_from_counts(c)["Dice"]
                    for c in by_patient.values()
                ])

                rows.append({
                    "Threshold": threshold,
                    "Global Dice": gm["Dice"],
                    "Global Precision": gm["Precision"],
                    "Global Recall": gm["Recall"],
                    "Mean Patient Dice": mean_patient_dice,
                    "Global FP": total_counts["FP"],
                    "Global FN": total_counts["FN"],
                })

            comparison = pd.DataFrame(rows).sort_values("Threshold")
            st.session_state["threshold_comparison"] = comparison
            st.dataframe(
                comparison.style.format({
                    "Threshold": "{:.2f}",
                    "Global Dice": "{:.4f}",
                    "Global Precision": "{:.4f}",
                    "Global Recall": "{:.4f}",
                    "Mean Patient Dice": "{:.4f}",
                    "Global FP": "{:,.0f}",
                    "Global FN": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(
                comparison["Threshold"],
                comparison["Global Dice"],
                marker="o",
                label="Global Dice",
            )
            ax.plot(
                comparison["Threshold"],
                comparison["Global Precision"],
                marker="o",
                label="Global Precision",
            )
            ax.plot(
                comparison["Threshold"],
                comparison["Global Recall"],
                marker="o",
                label="Global Recall",
            )
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if "threshold_comparison" in st.session_state:
        st.download_button(
            "Download threshold comparison CSV",
            data=st.session_state["threshold_comparison"].to_csv(index=False),
            file_name="threshold_comparison.csv",
            mime="text/csv",
        )


# =========================================================
# Error Analysis
# =========================================================
elif page == "Error Analysis":
    st.header("Error Analysis")
    st.write(
        "Find slices with high false-positive or false-negative counts "
        "at the selected threshold."
    )

    threshold = st.slider(
        "Error analysis threshold",
        0.05,
        0.95,
        0.50,
        0.05,
    )

    
    min_predicted_pixels = st.number_input(
        "Minimum predicted lesion pixels per patient",
        min_value=1,
        max_value=10000,
        value=10,
        step=1,
        help=(
            "A patient is predicted Lesion-positive if the total "
            "number of predicted lesion pixels across all slices "
            "is at least this value."
        ),
    )


    if st.button("Analyze validation errors", type="primary"):
        results = evaluate_validation(model, dataset)
        rows = []

        for r in results:
            m = calculate_metrics(r["mask"], r["probability"] >= threshold)
            rows.append({
                "patient_id": r["patient_id"],
                "slice_index": r["slice_index"],
                "GT pixels": int(r["mask"].sum()),
                "Predicted pixels": int((r["probability"] >= threshold).sum()),
                "Dice": m["Dice"],
                "FP": m["FP"],
                "FN": m["FN"],
            })

        error_df = pd.DataFrame(rows)

        
        # Patient-level lesion classification
        patient_counts = {}

        for r in results:
            patient_id = str(r["patient_id"])

            counts = patient_counts.setdefault(
                patient_id,
                {
                    "GT lesion pixels": 0,
                    "Predicted lesion pixels": 0,
                },
            )

            gt_mask = np.asarray(r["mask"], dtype=bool)
            pred_mask = np.asarray(
                r["probability"] >= threshold,
                dtype=bool,
            )

            counts["GT lesion pixels"] += int(gt_mask.sum())
            counts["Predicted lesion pixels"] += int(
                pred_mask.sum()
            )

        patient_rows = []

        for patient_id, counts in sorted(patient_counts.items()):
            actual_positive = counts["GT lesion pixels"] > 0

            predicted_positive = (
                counts["Predicted lesion pixels"]
                >= int(min_predicted_pixels)
            )

            patient_rows.append({
                "patient_id": patient_id,
                "Actual": (
                    "Lesion-positive"
                    if actual_positive
                    else "Lesion-negative"
                ),
                "Predicted": (
                    "Lesion-positive"
                    if predicted_positive
                    else "Lesion-negative"
                ),
                "GT lesion pixels": counts["GT lesion pixels"],
                "Predicted lesion pixels": (
                    counts["Predicted lesion pixels"]
                ),
                "Correct": actual_positive == predicted_positive,
            })

        patient_df = pd.DataFrame(patient_rows)

        # Confusion matrix: rows = actual, columns = predicted
        tp_patients = int(
            (
                (patient_df["Actual"] == "Lesion-positive")
                & (patient_df["Predicted"] == "Lesion-positive")
            ).sum()
        )
        fn_patients = int(
            (
                (patient_df["Actual"] == "Lesion-positive")
                & (patient_df["Predicted"] == "Lesion-negative")
            ).sum()
        )
        fp_patients = int(
            (
                (patient_df["Actual"] == "Lesion-negative")
                & (patient_df["Predicted"] == "Lesion-positive")
            ).sum()
        )
        tn_patients = int(
            (
                (patient_df["Actual"] == "Lesion-negative")
                & (patient_df["Predicted"] == "Lesion-negative")
            ).sum()
        )

        cm = np.array([
            [tn_patients, fp_patients],
            [fn_patients, tp_patients],
        ])

        st.divider()
        st.subheader("Patient-Level Confusion Matrix")
        st.caption(
            f"Patients evaluated: {len(patient_df)} | "
            f"Probability threshold: {threshold:.2f} | "
            f"Minimum predicted pixels: {int(min_predicted_pixels)}"
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            "Lesion-negative",
            "Lesion-positive",
        ])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([
            "Lesion-negative",
            "Lesion-positive",
        ])
        ax.set_xlabel("Predicted patient class")
        ax.set_ylabel("Actual patient class")
        ax.set_title("Patient-Level Confusion Matrix")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if cm[i, j] > cm.max() / 2
                        else "black"
                    ),
                    fontsize=14,
                )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        total_patients = len(patient_df)
        accuracy = (
            (tp_patients + tn_patients) / total_patients
            if total_patients else 0.0
        )
        precision_patient = (
            tp_patients / (tp_patients + fp_patients)
            if tp_patients + fp_patients else 0.0
        )
        recall_patient = (
            tp_patients / (tp_patients + fn_patients)
            if tp_patients + fn_patients else 0.0
        )
        specificity_patient = (
            tn_patients / (tn_patients + fp_patients)
            if tn_patients + fp_patients else 0.0
        )
        f1_patient = (
            2 * precision_patient * recall_patient
            / (precision_patient + recall_patient)
            if precision_patient + recall_patient else 0.0
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{accuracy:.3f}")
        c2.metric("Precision", f"{precision_patient:.3f}")
        c3.metric("Sensitivity", f"{recall_patient:.3f}")
        c4.metric("Specificity", f"{specificity_patient:.3f}")
        c5.metric("F1-score", f"{f1_patient:.3f}")

        st.subheader("Patient Classification Details")
        st.dataframe(
            patient_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Patient-Level Confusion Matrix CSV",
            data=patient_df.to_csv(index=False),
            file_name="patient_level_confusion_matrix.csv",
            mime="text/csv",
            key="download_patient_confusion_matrix",
        )

        summary_df = pd.DataFrame([{
            "Threshold": threshold,
            "Minimum predicted pixels": int(min_predicted_pixels),
            "Patients": total_patients,
            "TP patients": tp_patients,
            "TN patients": tn_patients,
            "FP patients": fp_patients,
            "FN patients": fn_patients,
            "Accuracy": accuracy,
            "Precision": precision_patient,
            "Sensitivity": recall_patient,
            "Specificity": specificity_patient,
            "F1-score": f1_patient,
        }])

        st.download_button(
            "Download Patient-Level Summary CSV",
            data=summary_df.to_csv(index=False),
            file_name="patient_level_confusion_matrix_summary.csv",
            mime="text/csv",
            key="download_patient_confusion_summary",
        )

        # Aggregate pixel-level confusion matrix
        total_counts = {
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
        }

        for r in results:
            m = calculate_metrics(
                r["mask"],
                r["probability"] >= threshold,
            )

            for key in total_counts:
                total_counts[key] += m[key]

        tp = total_counts["TP"]
        tn = total_counts["TN"]
        fp = total_counts["FP"]
        fn = total_counts["FN"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        st.divider()
        st.subheader("Pixel-Level Confusion Matrix")
        st.caption(
            f"Aggregated over {len(results)} validation slices "
            f"at threshold {threshold:.2f}. "
            "Rows = ground truth; columns = prediction."
        )

        confusion_matrix = np.array([
            [tn, fp],
            [fn, tp],
        ])

        fig, ax = plt.subplots(figsize=(6, 4))
        image = ax.imshow(confusion_matrix, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Predicted Background", "Predicted Lesion"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual Background", "Actual Lesion"])
        ax.set_title("Pixel-Level Confusion Matrix")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground Truth")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i,
                    f"{confusion_matrix[i, j]:,}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if confusion_matrix[i, j] > confusion_matrix.max() / 2
                        else "black"
                    ),
                    fontsize=12,
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{precision:.4f}")
        c2.metric("Recall / Sensitivity", f"{recall:.4f}")
        c3.metric("Specificity", f"{specificity:.4f}")
        c4.metric("F1-score", f"{f1:.4f}")

        confusion_df = pd.DataFrame(
            [
                {
                    "Threshold": threshold,
                    "Validation slices": len(results),
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                    "Precision": precision,
                    "Recall": recall,
                    "Specificity": specificity,
                    "F1-score": f1,
                }
            ]
        )

        st.download_button(
            "Download Confusion Matrix Metrics CSV",
            data=confusion_df.to_csv(index=False),
            file_name="validation_confusion_matrix.csv",
            mime="text/csv",
            key="download_confusion_matrix_csv",
        )

        st.session_state["error_report"] = error_df

        st.subheader("Largest false-positive counts")
        st.dataframe(
            error_df.sort_values("FP", ascending=False).head(20),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Largest false-negative counts")
        st.dataframe(
            error_df.sort_values("FN", ascending=False).head(20),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Lowest Dice scores")
        st.dataframe(
            error_df.sort_values("Dice", ascending=True).head(20),
            use_container_width=True,
            hide_index=True,
        )

    if "error_report" in st.session_state:
        st.download_button(
            "Download error analysis CSV",
            data=st.session_state["error_report"].to_csv(index=False),
            file_name="validation_error_analysis.csv",
            mime="text/csv",
        )


# =========================================================
# Export Reports
# =========================================================
elif page == "Export Reports":
    st.header("Export Reports")
    st.write(
        "Generate a CSV report of validation performance by patient "
        "and threshold."
    )

    threshold = st.selectbox(
        "Report threshold",
        options=[0.3, 0.5, 0.7, 0.9],
        index=1,
    )

    if st.button("Generate patient report", type="primary"):
        results = evaluate_validation(model, dataset)
        patient_counts = {}

        for r in results:
            m = calculate_metrics(
                r["mask"], r["probability"] >= threshold
            )
            p = patient_counts.setdefault(
                r["patient_id"],
                {"TP": 0, "FP": 0, "FN": 0, "slices": 0},
            )

            for key in ("TP", "FP", "FN"):
                p[key] += m[key]
            p["slices"] += 1

        rows = []
        for patient_id, counts in sorted(patient_counts.items()):
            m = calculate_metrics_from_counts(counts)
            rows.append({
                "patient_id": patient_id,
                "slices": counts["slices"],
                "threshold": threshold,
                "Dice": m["Dice"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "TP": counts["TP"],
                "FP": counts["FP"],
                "FN": counts["FN"],
            })

        report = pd.DataFrame(rows)
        st.session_state["patient_report"] = report
        st.dataframe(report, use_container_width=True, hide_index=True)

    if "patient_report" in st.session_state:
        st.download_button(
            "Download patient report CSV",
            data=st.session_state["patient_report"].to_csv(index=False),
            file_name="patient_level_report.csv",
            mime="text/csv",
        )


# =========================================================
# Shared pooled metric helper
# =========================================================
def calculate_metrics_from_counts(counts):
    tp = int(counts.get("TP", 0))
    fp = int(counts.get("FP", 0))
    fn = int(counts.get("FN", 0))

    denom = 2 * tp + fp + fn
    dice = 2 * tp / denom if denom else 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
    }
