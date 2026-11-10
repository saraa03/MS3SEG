from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
CSV_PATH = OUTPUT_DIR / "patient_level_metrics.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find: {CSV_PATH}")

# Read patient-level metrics without pandas
with open(CSV_PATH, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

if not rows:
    raise ValueError("The patient-level CSV file is empty.")

# Convert Dice scores to numbers
for row in rows:
    row["dice"] = float(row["dice"])

# Sort patients from lowest Dice to highest Dice
rows.sort(key=lambda row: row["dice"])

dice_scores = [row["dice"] for row in rows]
patient_ids = [row["patient_id"] for row in rows]

mean_dice = sum(dice_scores) / len(dice_scores)
median_dice = dice_scores[len(dice_scores) // 2]

print("\n========== PATIENT-LEVEL ANALYSIS ==========")
print(f"Patients evaluated: {len(rows)}")
print(f"Mean Dice:          {mean_dice:.4f}")
print(f"Median Dice:        {median_dice:.4f}")
print(f"Minimum Dice:       {min(dice_scores):.4f}")
print(f"Maximum Dice:       {max(dice_scores):.4f}")

print("\n10 patients with the lowest Dice scores:")
for row in rows[:10]:
    print(f"Patient {row['patient_id']}: Dice = {row['dice']:.4f}")

# Save ranked patient results
ranked_path = OUTPUT_DIR / "patient_dice_ranked.csv"
with open(ranked_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Plot patient-level Dice scores
plt.figure(figsize=(12, 5))
plt.bar(patient_ids, dice_scores)
plt.axhline(
    mean_dice,
    linestyle="--",
    label=f"Mean Dice = {mean_dice:.3f}"
)
plt.xlabel("Patient ID")
plt.ylabel("Dice Score")
plt.title("Validation Dice Score per Patient")
plt.ylim(0, 1)
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()

plot_path = OUTPUT_DIR / "patient_dice_plot.png"
plt.savefig(plot_path, dpi=200)
plt.show()

print(f"\nRanked CSV saved to: {ranked_path}")
print(f"Plot saved to: {plot_path}")