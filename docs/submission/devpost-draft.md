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

## How it is built

- Python and PyTorch/TorchVision
- ImageNet-pretrained EfficientNet-B0 with a binary classification head
- A balanced subset of SID_Set containing authentic and fully synthetic images
- Deterministic evaluation across the published transformation grid
- Streamlit for the optional live demonstration
- Kaggle's free GPU notebook environment for training

Implementation details, final dataset counts, metrics, and runtime will be added
after the reproducible training run finishes.

## Results

TBD — insert measured clean and transformed results only.

## Challenges and lessons

TBD after implementation and error analysis.

## Limitations

- The model is a prototype rather than a production moderation system.
- Performance on unseen generators may differ from the sampled test data.
- Severe compression or blur can remove forensic signals from either class.
- The confidence is evidence from one detector, not proof of image provenance.

## Team contributions

TBD — add member names and their actual contributions.
