# Evaluation and Error Analysis

This note reports the final controlled Kaggle experiment generated on
2026-08-30 from repository commit `e32257d`. Both models used the same
EfficientNet-B0 architecture, SID_Set sample, split, seed, trainable layers,
optimizer, and four-epoch budget. The only experimental difference was whether
the published robustness transformations were sampled during training.

## Evaluation population

- Dataset: `saberzl/SID_Set`, pinned revision
  `dc03ead57929879319ce30a82bfcfb8d317b10bd` (CC BY 4.0 per its dataset card).
- Prepared subset: 6,000 unique images, balanced between authentic and fully
  synthetic classes.
- Test split: 600 images, with 300 images per class.
- Evaluation: the same test images under one clean and 19 transformed
  scenarios. The fixed decision threshold is 0.50.
- The organizer-provided WildFake demonstration data was not used for training,
  threshold selection, or model selection.

## Clean and transformed results

| Result | Clean baseline | Robustness-trained | Robust-model difference |
| --- | ---: | ---: | ---: |
| Clean ROC-AUC | 0.997056 | 0.996400 | -0.000656 |
| Mean transformed ROC-AUC | 0.995635 | 0.995784 | +0.000149 |
| Worst transformed ROC-AUC | 0.986133 | 0.990278 | +0.004144 |
| Mean transformed balanced accuracy | 95.47% | 95.83% | +0.36 pp |

The robust model does not win every scenario: it has higher ROC-AUC in 8 of 20
cases and gives up 0.066 percentage points of clean ROC-AUC. Its useful gain is
narrower and more defensible: the worst-case ROC-AUC improves by 0.414
percentage points, and the largest threshold-level improvement occurs under
strong Gaussian noise.

Both models' worst ROC-AUC scenario is Gaussian noise at `sigma=0.10`. In that
case, robust training raises balanced accuracy from 84.67% to 89.00%, reduces
the false-negative rate from 30.67% to 21.67%, and improves ROC-AUC from
0.986133 to 0.990278. The mean transformed ROC-AUC difference is too small to
present as a confident universal gain.

The complete clean-versus-transform chart is
[`artifacts/metrics/robustness.png`](../../artifacts/metrics/robustness.png).

## Clean-image errors

At the fixed 0.50 threshold, the selected model correctly classifies 578 of 600
clean test images:

| Actual class | Predicted authentic | Predicted AIGC |
| --- | ---: | ---: |
| Authentic | 286 true negatives | 14 false positives |
| AIGC | 8 false negatives | 292 true positives |

The rendered matrix is
[`artifacts/figures/confusion_matrix.png`](../../artifacts/figures/confusion_matrix.png).

Representative high-confidence errors, identified by the normalized-image
SHA-256 filename, are:

- False positive: authentic image
  `eeef90141da68a06595a2b72a2d3ebcc6668acbede8e0951b9a15b071ff204b6.jpg`,
  scored `0.83218` AIGC.
- False negative: generated image
  `61e9cef8dcbd2b9fa1ed6276cf855e3c7b70e2709524335d81b95d1eb3b671c4.jpg`,
  scored `0.31974` AIGC.

Raw dataset images are intentionally excluded from Git, so this public note
records reproducible identifiers and scores rather than redistributing the
pixels. Visual inspection of these exact examples requires reacquiring them
from the pinned SID_Set revision under its license.

## Transformation failure pattern

Strong noise mostly creates false negatives rather than false positives. Under
Gaussian noise at `sigma=0.10`, the selected model produces 65 false negatives
and one false positive, compared with eight false negatives and 14 false
positives on clean images. In practical terms, heavy noise can suppress an
AIGC score enough to make generated images look authentic at the fixed
threshold even while rank-based ROC-AUC remains high.

The ranked error-count figure is
[`artifacts/figures/error_analysis.png`](../../artifacts/figures/error_analysis.png).
It is generated reproducibly with:

```bash
python scripts/generate_submission_evidence.py
```

## Limitations and interpretation

- This is a same-source held-out SID_Set test, not evidence of performance on
  every generator, social platform, screenshot pipeline, or camera domain.
- SID_Set's selected synthetic images are all square, while most selected real
  images are not. The shared preprocessing contract removes direct input-shape
  leakage by preserving aspect ratio during resize and then applying the same
  center crop, but broader source and content biases may remain.
- The 0.50 threshold was fixed for the project. It was not tuned on the test set.
- The 600-image test split supports a hackathon prototype, not a production
  moderation claim. Small model differences should not be presented as
  statistically decisive without a larger external benchmark.
- The model is a screening aid. A score is not proof of provenance and must not
  be the sole basis for moderation, attribution, legal, safety, or disciplinary
  decisions.

All headline metrics were independently recomputed from the exported prediction
rows with zero discrepancy. The export is bound to the manifest and both model
files by SHA-256 values in `artifacts/metrics/run_context.json`.
