# GenImage v2 Held-Out Evaluation

> **Scope:** Held-out GenImage and SID comparison for the frozen v1 and warm-started v2 checkpoints. WildFake was not accessed or used for model selection.

The v2 checkpoint was warm-started from v1. Results below use untouched test splits and a fixed 0.50 threshold. They support a manual deployment decision; this workflow does not replace the application checkpoint.

## Locked identity

- v1 checkpoint SHA-256: `806fbabc5ecae8394369d08738cbf0c993568137d323a8133167e4557d04eed2`
- v2 checkpoint SHA-256: `b45022d9dab2a02300934c239eee24dd40ef8e402f24c1f27fc2d63a46117c12`
- v2 parent SHA-256: `806fbabc5ecae8394369d08738cbf0c993568137d323a8133167e4557d04eed2`
- Fixed threshold: `0.50`
- Evaluation scenarios: 20 (clean plus 19 transforms)

## GenImage held-out test

1,120 images: 560 real and 560 generated. Test content digest: `03b47414aad49389f634f6705cc55914d59c711863da2f0fabea872bc717d398`.

### Clean metrics

| Metric | v1 | v2 | v2 − v1 |
|---|---:|---:|---:|
| ROC-AUC | 0.6032 | 0.8633 | +0.2600 |
| Average precision | 0.6442 | 0.8701 | +0.2258 |
| Balanced accuracy | 0.5580 | 0.7580 | +0.2000 |
| F1 | 0.2688 | 0.7120 | +0.4432 |
| False-positive rate | 0.0464 | 0.0821 | +0.0357 |
| False-negative rate | 0.8375 | 0.4018 | -0.4357 |
| Brier score | 0.3467 | 0.1687 | -0.1780 |

### Mean transformed metrics

| Metric | v1 | v2 | v2 − v1 |
|---|---:|---:|---:|
| ROC-AUC | 0.6122 | 0.8636 | +0.2513 |
| Average precision | 0.6433 | 0.8655 | +0.2222 |
| Balanced accuracy | 0.5604 | 0.7600 | +0.1996 |
| F1 | 0.2721 | 0.7134 | +0.4413 |
| False-positive rate | 0.0466 | 0.0947 | +0.0481 |
| False-negative rate | 0.8326 | 0.3852 | -0.4474 |
| Brier score | 0.3460 | 0.1676 | -0.1784 |

- v1 worst scenario: `gaussian_noise:0.05` at ROC-AUC 0.5591; clean-to-worst drop 0.0442.
- v1 clean confusion (TN / FP / FN / TP): 534 / 26 / 469 / 91.
- v1 class-normalized clean confusion: real → real 0.9536, real → generated 0.0464; generated → real 0.8375, generated → generated 0.1625.

- v2 worst scenario: `gaussian_noise:0.05` at ROC-AUC 0.8077; clean-to-worst drop 0.0555.
- v2 clean confusion (TN / FP / FN / TP): 514 / 46 / 225 / 335.
- v2 class-normalized clean confusion: real → real 0.9179, real → generated 0.0821; generated → real 0.4018, generated → generated 0.5982.

Deltas are v2 minus v1. Positive is normally better, except for false-positive rate, false-negative rate, Brier score, and degradation.

## SID regression test

600 images: 300 real and 300 generated. Test content digest: `061fc4a2ac5464d12575b239705462b56ba77ac9008db95fb07c63c5afd784b3`.

### Clean metrics

| Metric | v1 | v2 | v2 − v1 |
|---|---:|---:|---:|
| ROC-AUC | 0.9964 | 0.9931 | -0.0033 |
| Average precision | 0.9965 | 0.9940 | -0.0024 |
| Balanced accuracy | 0.9633 | 0.9000 | -0.0633 |
| F1 | 0.9637 | 0.9080 | -0.0557 |
| False-positive rate | 0.0467 | 0.1867 | +0.1400 |
| False-negative rate | 0.0267 | 0.0133 | -0.0133 |
| Brier score | 0.0283 | 0.0658 | +0.0375 |

### Mean transformed metrics

| Metric | v1 | v2 | v2 − v1 |
|---|---:|---:|---:|
| ROC-AUC | 0.9958 | 0.9915 | -0.0043 |
| Average precision | 0.9958 | 0.9925 | -0.0034 |
| Balanced accuracy | 0.9583 | 0.9086 | -0.0497 |
| F1 | 0.9575 | 0.9151 | -0.0425 |
| False-positive rate | 0.0363 | 0.1630 | +0.1267 |
| False-negative rate | 0.0470 | 0.0198 | -0.0272 |
| Brier score | 0.0319 | 0.0640 | +0.0321 |

- v1 worst scenario: `gaussian_noise:0.1` at ROC-AUC 0.9903; clean-to-worst drop 0.0061.
- v1 clean confusion (TN / FP / FN / TP): 286 / 14 / 8 / 292.
- v1 class-normalized clean confusion: real → real 0.9533, real → generated 0.0467; generated → real 0.0267, generated → generated 0.9733.

- v2 worst scenario: `gaussian_noise:0.1` at ROC-AUC 0.9820; clean-to-worst drop 0.0111.
- v2 clean confusion (TN / FP / FN / TP): 244 / 56 / 4 / 296.
- v2 class-normalized clean confusion: real → real 0.8133, real → generated 0.1867; generated → real 0.0133, generated → generated 0.9867.

Deltas are v2 minus v1. Positive is normally better, except for false-positive rate, false-negative rate, Brier score, and degradation.

## Evidence handling

The compact JSON, report, and figure contain aggregate results only. Detailed manifests, per-image predictions, raw scenario metrics, source images, and execution paths remain in the local audit layer.
