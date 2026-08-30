# Devpost Draft

## Project name

TBD

## One-line summary

A lightweight detector that estimates whether an image is AI-generated and
measures whether that decision remains stable after common social-media edits.

## Inspiration and problem

Highly realistic generated images make misinformation, impersonation, and fraud
easier. Detection becomes harder after ordinary operations such as messaging-app
compression, resizing, cropping, filtering, and blur. Our goal is therefore not
only to classify clean images, but to measure and improve transformation
robustness explicitly.

## What it does

The prototype accepts an image and returns an AIGC likelihood between 0 and 1.
Its robustness view applies representative JPEG, blur, resizing, noise, color,
and crop transformations, then visualizes how much the prediction changes. A
batch command also accepts an image directory and writes the required JSON output.

Unlike a clean-accuracy-only demo, RealityCheck exposes a **robustness passport**:
the score range, boundary stability, largest shift, and most destabilizing edit
for each upload. The offline evaluation compares a clean baseline against the
robustness-trained model on the same held-out source groups.

## How it is built

- Python and PyTorch/TorchVision
- ImageNet-pretrained EfficientNet-B0 with a binary classification head
- A streamed, balanced 6,000-image subset of SID_Set containing only real
  (label 0) and fully synthetic (label 1) images
- Source-grouped splitting by SID `img_id`, normalized-image deduplication, and
  the same JPEG normalization for both classes to reduce leakage and shortcuts
- Deterministic evaluation across the published transformation grid
- Streamlit for the optional live demonstration
- Kaggle's free GPU notebook environment for training

SID_Set is listed as CC BY 4.0 on its dataset card. Exact retained and split
counts will be copied from `manifest_summary.json` after the reproducible run.

## Results (fill only from exported artifacts)

| Model | Clean ROC-AUC | Mean transformed ROC-AUC | Worst transform / severity | Worst ROC-AUC | Clean-to-worst drop |
| --- | ---: | ---: | --- | ---: | ---: |
| Clean baseline | TBD | TBD | TBD | TBD | TBD |
| Robustness-trained | TBD | TBD | TBD | TBD | TBD |

Also report balanced accuracy, false-positive rate, and false-negative rate at
the fixed 0.5 threshold. Link the generated clean-versus-transformed figure;
do not manually transcribe values without checking `metrics.json`.

## Challenges and lessons (complete after error analysis)

- Representative false positive: TBD from `predictions.csv`
- Representative false negative: TBD from `predictions.csv`
- Most damaging transformation: TBD from `metrics.json`
- Trade-off between clean accuracy and robustness: TBD from the two model runs

## Limitations

- The model is a prototype rather than a production moderation system.
- Performance on unseen generators may differ from the sampled test data.
- Severe compression or blur can remove forensic signals from either class.
- The confidence is evidence from one detector, not proof of image provenance.

## Team contributions

Replace each placeholder with the actual contributor before submission:

| Contributor | Verified contribution |
| --- | --- |
| Name TBD | Core model, training, evaluation, and integration |
| Name TBD | Dataset preparation and manifest pipeline |
| Name TBD | Deterministic robustness transformations |
| Name TBD | Streamlit robustness-passport demo |
