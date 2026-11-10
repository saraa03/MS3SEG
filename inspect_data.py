
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED = {".venv", ".venv-1", "__pycache__", ".git"}

print("Project:", ROOT)

nii_files = []

for p in ROOT.rglob("*"):
    if not p.is_file():
        continue

    relative = p.relative_to(ROOT)

    if any(part in EXCLUDED for part in relative.parts):
        continue

    if p.name.endswith((".nii.gz", ".nii")):
        nii_files.append(p)

print("\nTotal MRI/mask files:", len(nii_files))

for p in nii_files[:50]:
    print(p.relative_to(ROOT))

if not nii_files:
    print(
        "\nNo NIfTI files found. "
        "Check that both RAR archives were extracted."
    )
