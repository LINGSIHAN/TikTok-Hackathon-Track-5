# Devpost Draft

## Project name

RealityCheck

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
- Codex for collaborative implementation/review and GitHub for version control

SID_Set is listed as CC BY 4.0 on its dataset card. The pinned run retained
3,000 images per class, split into 4,800 training, 600 validation, and 600 test
images with equal class counts in every split.

## Results

| Model | Clean ROC-AUC | Mean transformed ROC-AUC | Worst transform / severity | Worst ROC-AUC | Clean-to-worst drop |
| --- | ---: | ---: | --- | ---: | ---: |
| Clean baseline | 0.997056 | 0.995635 | Gaussian noise / 0.10 | 0.986133 | 0.010922 |
| Robustness-trained | 0.996400 | 0.995784 | Gaussian noise / 0.10 | 0.990278 | 0.006122 |

At the fixed 0.50 threshold, the selected model reaches 96.33% clean balanced
accuracy, with a 4.67% false-positive rate and 2.67% false-negative rate. Under
the hardest Gaussian-noise setting, robust training improves balanced accuracy
from 84.67% to 89.00% and reduces the false-negative rate from 30.67% to
21.67%. The complete figures and methodology are linked in the repository's
evaluation and error-analysis note.

## Challenges and lessons

- Representative false positive: authentic case `eeef9014...204b6.jpg`, scored
  0.83218 AIGC.
- Representative false negative: generated case `61e9cef8...671c4.jpg`, scored
  0.31974 AIGC.
- Strong Gaussian noise is the most damaging transformation. It primarily
  suppresses generated-image scores, producing false negatives.
- Robust training improves worst-case ROC-AUC by 0.004144 but reduces clean
  ROC-AUC by 0.000656. Its mean transformed gain is small, so our claim is
  improved worst-case resilience rather than universal superiority.

## Limitations

- The model is a prototype rather than a production moderation system.
- Performance on unseen generators may differ from the sampled test data.
- Severe compression or blur can remove forensic signals from either class.
- The confidence is evidence from one detector, not proof of image provenance.
- The same-source SID_Set test does not establish performance on unseen
  generators or real social-platform distributions.

## What we would improve with more time

- Evaluate on multiple generator families and real platform re-encoding
  pipelines, including the organizer demonstration set only as a final
  untouched benchmark.
- Add source-balanced data whose aspect-ratio and content distributions do not
  correlate with the class label.
- Calibrate the decision threshold on a larger validation set and expose an
  explicit abstention range for uncertain cases.
- Test model quantization and caching to reduce cold-start and stress-test time
  on free CPU hosting.

## Links

- GitHub: <https://github.com/LINGSIHAN/TikTok-Hackathon-Track-5>
- Live Streamlit demo: TBD after deployment and smoke testing
- Public demo video: TBD after recording

## Team contributions

Replace each placeholder with the actual contributor before submission:

| Contributor | Verified contribution |
| --- | --- |
| Name TBD | Core model, training, evaluation, and integration |
| Name TBD | Dataset preparation and manifest pipeline |
| Name TBD | Deterministic robustness transformations |
| Name TBD | Streamlit robustness-passport demo |
