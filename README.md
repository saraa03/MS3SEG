
# MS3SEG: Multiple Sclerosis Lesion Segmentation Using U-Net

## Overview

MS3SEG is a research project focused on segmenting
white matter hyperintensities (WMH) associated with
Multiple Sclerosis (MS) from brain MRI scans.

The project uses a U-Net deep learning architecture
to generate pixel-wise lesion segmentation predictions
from FLAIR MRI images.

## Project Objectives

- Load and preprocess FLAIR MRI volumes.
- Load corresponding lesion segmentation masks.
- Train a U-Net model for lesion segmentation.
- Evaluate segmentation performance.
- Analyze false-positive and false-negative predictions.
- Investigate the effect of prediction thresholds.

## Dataset

The project uses the MS3SEG dataset.

The dataset contains preprocessed FLAIR MRI volumes
and corresponding lesion masks.

**Important:** Patient MRI scans and medical masks are
not included in this repository. Users must obtain the
dataset through its authorized distribution channel
and comply with its license and usage requirements.

Expected local data structure:

```text
MS3SEG/
├── MS_100_patient_preprocessed/
│   └── MS_100_patient_preprocessed/
│       └── [patient_id]/
│           └── [patient_id]_FLAIR.nii.gz
│
└── MS_100_patient_masks/
    └── MS_100_patient_masks/
        └── abWMH_Masks/
            └── [patient_id]/
                └── [patient_id]_abWMH_Mask.nii.gz
```

## Model Architecture

The project uses a U-Net architecture for
pixel-level lesion segmentation.

The model receives 2D FLAIR slices and predicts
a probability map representing the likelihood
of lesion pixels.

The sigmoid function is applied to the model logits
to obtain probabilities.

## Preprocessing

The dataset pipeline includes:

- Loading NIfTI volumes using NiBabel.
- Checking image and mask dimensions.
- Checking spatial affine alignment.
- Converting lesion labels to binary masks.
- Robust intensity normalization using the 1st
  and 99th percentiles.
- Extracting 2D slices from 3D volumes.

## Evaluation

The project evaluates segmentation using:

- Dice Similarity Coefficient
- Precision
- Recall

The validation split is defined in the local
`outputs/patient_split.json` file.

The patient split file is not included by default
because it may contain dataset-related identifiers.
A compatible split must be prepared locally.

## Baseline Results

The existing baseline model achieved the following
validation results:

| Metric | Value |
|---|---:|
| Pixel Dice | 0.7431 |
| Pixel Precision | 0.7139 |
| Pixel Recall | 0.7749 |
| Mean Patient Dice | 0.6819 |

These values are specific to the reported validation
split and evaluation implementation.

They should not be interpreted as clinical diagnostic
performance.

## Error Analysis

Additional analysis scripts investigate:

- Patient-level segmentation performance.
- False-positive regions.
- Prediction confidence maps.
- The effect of different probability thresholds.
- Connected components in predicted masks.

A high model probability does not necessarily mean
that a predicted region is a true lesion.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/MS3SEG.git
cd MS3SEG
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data Setup

Place the authorized dataset files in the expected
directory structure described in the Dataset section.

Do not upload patient MRI data or medical masks
to this repository.

## Usage

Run the training script:

```bash
python train.py
```

Run evaluation:

```bash
python evaluate.py
```

Inspect prediction comparisons:

```bash
python compare_predictions.py
```

Analyze false-positive components:

```bash
python analyze_false_positive_components.py
```

Inspect false-positive regions:

```bash
python inspect_false_positive_regions.py
```

Note: The scripts require the dataset, split configuration,
and any model checkpoints they depend on to be available
locally. Check each script's configuration before running.

## Limitations

- The dataset used in this project contains MS cases.
- The segmentation model is not a healthy-versus-MS classifier.
- Segmentation performance does not establish a clinical diagnosis.
- False-positive predictions require further investigation.
- Results depend on the dataset, patient split, preprocessing,
  and evaluation threshold.

## Reproducibility

For reproducible experiments, record:

- Python and library versions.
- Dataset version and authorized source.
- Patient-level training and validation split.
- Model configuration.
- Random seeds.
- Checkpoint selection criteria.
- Evaluation threshold.

## Disclaimer

This project is intended for research and educational
purposes only. It is not a medical device and must not
be used to diagnose, treat, or make clinical decisions.

## License

A license will be added after confirming the permissions
for the project code and the dataset.