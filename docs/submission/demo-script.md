# Demo Video Script

Target length: approximately 2-3 minutes.

## 1. Problem (15-20 seconds)

Explain that AI-image detection often looks strong on clean files but can fail
after compression, blur, resizing, filters, or crops—the normal lifecycle of a
social-media image.

## 2. Solution overview (20 seconds)

Show the repository and explain the lightweight EfficientNet-B0 detector, the
balanced/source-grouped SID_Set subset, and the comparison between clean-only
and robustness-aware training on the same held-out images.

## 3. Single-image prediction (30 seconds)

Open the Streamlit app locally. Upload one generated image and show its estimated
AIGC probability. Repeat with one authentic image.

## 4. Robustness fingerprint (40 seconds)

Run the stress test. Show the transformed previews, probability chart, and
worst-case transformation. Explain that the full evaluation uses every published
severity, while the UI keeps the interactive view concise.

## 5. Batch contract and evidence (30 seconds)

Show the directory inference command and resulting JSON containing `image_path`
and `pred`. Then show `artifacts/figures/model_comparison.png` and the error-count
figures. Briefly show the GenImage v2 calibration decision: the threshold was
locked on validation data to minimize SID false positives under recall and
cross-generator safeguards, but v2 failed four deployment gates. Disclose that
the project owner subsequently selected v2 at its original threshold 0.50 for
the broader-generator demo and retained v1 for rollback; do not claim the gates
passed. The public repository records
representative false-positive and
false-negative identifiers and scores but not their raw pixels. Show those two
images only if they have been reacquired from the pinned SID_Set revision under
its license; otherwise show the identifiers and state that limitation.

## 6. Limitations and close (20 seconds)

State that this is a small, zero-cost prototype, not proof of provenance. Mention
unseen-generator generalization and destructive transformations as limitations,
then show the GitHub and optional live-demo URLs.
