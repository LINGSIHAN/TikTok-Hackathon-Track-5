# WildFake Demonstration Evaluation

> **Scope:** WildFake was evaluated only after the model checkpoint and 0.50 threshold were locked. It is demonstration-only and was never used for training, threshold selection, or model selection.

This external benchmark measures the frozen RealityCheck detector on the exact organizer subset. It is evidence of behavior on this subset, not proof of image provenance or a claim of universal detector performance.

## Locked evaluation identity

- WildFake revision: `18f53ff36ad9da60644039f0452b0e7b3907af6f`
- Dataset digest: `28ff0e20c5bbe0ceb6e9ebb39c50c7594a3af89f3efd894227e8fd77687b790d`
- Checkpoint SHA-256: `806fbabc5ecae8394369d08738cbf0c993568137d323a8133167e4557d04eed2`
- Fixed threshold: `0.50`
- Scenario mode: `clean`
- Images: 13,841 total (4,998 COCO val2017 real; 8,843 Advanced DALL-E 3 generated)
- Duplicate-content audit: 1,808 same-label group(s), 5,124 additional file(s), and 8,717 unique byte hashes; duplicates retained to preserve the exact organizer subset; no conflicting labels

## Clean results

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8554 |
| Average precision | 0.9175 |
| Balanced accuracy | 0.7689 |
| F1 | 0.7630 |
| False-positive rate | 0.1212 |
| False-negative rate | 0.3409 |
| Brier score | 0.1922 |

## Confusion counts at 0.50

| Actual class | Predicted real | Predicted generated |
|---|---:|---:|
| Real | 4,392 (0.8788) | 606 (0.1212) |
| Generated | 3,015 (0.3409) | 5,828 (0.6591) |

Metrics weight every organizer row, including the disclosed same-label duplicate-content rows. They should not be interpreted as an estimate from 13,841 independent images.

The compact JSON and figure beside this report contain only aggregate evidence. Raw predictions, the deterministic sample manifest, execution metadata, and all source images remain local and ignored by Git.
