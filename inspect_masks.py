from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent

def find_file(filename):
    matches = [
        p for p in ROOT.rglob(filename)
        if not any(
            part.lower() in {".venv", ".venv-1", "__pycache__"}
            for part in p.parts
        )
    ]
    return matches[0] if matches else None

for pid in ["001", "002", "003"]:
    flair_path = find_file(f"{pid}_FLAIR.nii.gz")
    mask_path = find_file(f"{pid}_abWMH_Mask.nii.gz")

    print(f"\n{'=' * 45}")
    print("Patient:", pid)

    if flair_path is None:
        print("FLAIR not found")
        continue

    if mask_path is None:
        print("abWMH lesion mask not found")
        continue

    flair_nii = nib.load(str(flair_path))
    mask_nii = nib.load(str(mask_path))

    flair = np.asarray(flair_nii.dataobj)
    mask = np.asarray(mask_nii.dataobj)

    print("FLAIR:", flair_path.relative_to(ROOT))
    print("Mask:", mask_path.relative_to(ROOT))
    print("FLAIR shape:", flair.shape)
    print("Mask shape:", mask.shape)
    print("Mask values:", np.unique(mask))
    print(
        "Same shape:",
        flair.shape == mask.shape
    )
    print(
        "Same affine:",
        np.allclose(flair_nii.affine, mask_nii.affine, atol=1e-3)
    )
    print("Lesion voxels:", int(np.count_nonzero(mask)))
