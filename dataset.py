
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class MS3SEGDataset(Dataset):
    def __init__(self, root, patient_ids):
        self.root = Path(root)
        self.samples = []

        # Correct paths for your extracted folder structure
        image_root = (
            self.root
            / "MS_100_patient_preprocessed"
            / "MS_100_patient_preprocessed"
        )

        mask_root = (
            self.root
            / "MS_100_patient_masks"
            / "MS_100_patient_masks"
            / "abWMH_Masks"
        )

        for pid in patient_ids:
            image_path = image_root / pid / f"{pid}_FLAIR.nii.gz"
            mask_path = mask_root / pid / f"{pid}_abWMH_Mask.nii.gz"

            if not image_path.exists():
                raise FileNotFoundError(
                    f"FLAIR image not found: {image_path}"
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Lesion mask not found: {mask_path}"
                )

            image_nii = nib.load(str(image_path))
            mask_nii = nib.load(str(mask_path))

            if image_nii.shape != mask_nii.shape:
                raise ValueError(
                    f"Shape mismatch for patient {pid}: "
                    f"{image_nii.shape} vs {mask_nii.shape}"
                )

            if not np.allclose(
                image_nii.affine, mask_nii.affine, atol=1e-3
            ):
                raise ValueError(
                    f"Spatial alignment mismatch for patient {pid}"
                )

            image = np.asarray(
                image_nii.dataobj, dtype=np.float32
            )
            mask = np.asarray(
                mask_nii.dataobj, dtype=np.float32
            )

            # Convert lesion labels 0 and 255 into 0 and 1
            mask = (mask > 0).astype(np.float32)

            # Robust intensity normalization
            brain = image > 0

            if brain.any():
                lo, hi = np.percentile(image[brain], [1, 99])
                image = np.clip(
                    (image - lo) / max(float(hi - lo), 1e-8),
                    0,
                    1
                )
                image[~brain] = 0

            # Store all slices, including slices without lesions
            for z in range(image.shape[2]):
                x = image[:, :, z]
                y = mask[:, :, z]

                self.samples.append(
                    (x.copy(), y.copy(), pid, z)
                )

        print(
            f"Loaded {len(patient_ids)} patients "
            f"and {len(self.samples)} slices."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y, pid, z = self.samples[index]

        x = torch.from_numpy(x[None, :, :].copy()).float()
        y = torch.from_numpy(y[None, :, :].copy()).float()

        return x, y, pid, z

