
# Experimental Results

## Baseline Validation Performance

| Metric | Result |
|---|---:|
| Pixel Dice | 0.7431 |
| Pixel Precision | 0.7139 |
| Pixel Recall | 0.7749 |
| Mean Patient Dice | 0.6819 |

## Threshold Analysis

The baseline model was evaluated using different
probability thresholds.

A threshold of 0.9 produced the following reported
aggregate validation results:

| Metric | Result |
|---|---:|
| Mean Patient Dice | 0.6870 |
| Pixel Dice | 0.7445 |
| Pixel Precision | 0.7489 |
| Pixel Recall | 0.7401 |

The threshold change produced a small aggregate
difference and did not improve every individual slice.

## Patient-Level Error Analysis

Patient 011 showed a challenging segmentation case.

Some slices with empty reference masks contained
connected predicted regions with high probabilities.

These findings require further investigation and
must not automatically be classified as confirmed
clinical false positives.

## Interpretation

Aggregate metrics do not describe all patient-level
behavior. Patient-level and slice-level analysis is
important when assessing segmentation models.
