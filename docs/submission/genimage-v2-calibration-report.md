# GenImage v2 Validation-Only Threshold Calibration

> The numeric threshold selector consumes clean SID and GenImage validation predictions only. The earlier 0.50 test review motivated this calibration, so the saved test re-score is exploratory rather than a fresh holdout. WildFake remains excluded.

## Locked operating point

- Threshold: `0.40042864`
- SID validation false positives: `76/300` (25.33%)
- SID validation generated recall: `99.67%`
- GenImage validation balanced accuracy: `76.79%`
- GenImage validation generated recall: `67.50%`

The selected point has the lowest SID validation false-positive count among all thresholds that met the recall and GenImage guardrails.
The SID validation interval is descriptive only because the same validation rows were searched across many thresholds. GenImage validation was also used for training early stopping, so it is not an independent calibration set.

## Exploratory re-score of the previously observed tests

- SID test false positives: `74/300` (24.67%)
- SID test balanced accuracy: `87.17%`
- SID test generated recall: `99.00%`
- GenImage test balanced accuracy: `79.11%`
- SID mean/worst transformed balanced accuracy: `88.18%` / `82.67%`
- GenImage mean/worst transformed balanced accuracy: `78.46%` / `70.18%`
- Exploratory deployment recommendation: **retain v1**

These tests had already been inspected at threshold 0.50 and therefore are not a fresh, unbiased holdout. They are an exploratory deployment check only. The threshold must not be changed after this re-score.
