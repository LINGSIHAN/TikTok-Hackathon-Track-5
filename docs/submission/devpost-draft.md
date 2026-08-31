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
- A post-lock WildFake demonstration benchmark that selectively retrieves the
  exact COCO val2017 and Advanced DALL-E 3 organizer subset, verifies every
  source byte, and keeps raw external data out of Git
- Streamlit for the optional live demonstration
- Kaggle's free GPU notebook environment for training
- Codex for collaborative implementation/review and GitHub for version control

We also built a separate, review-gated v2 workflow that warm-starts from the
frozen model using only SID training rows plus a balanced subset of Unbiased
Tiny GenImage. It holds out GenImage validation/test data, compares v1 and v2 on
both GenImage and SID across the same 20 scenarios, and never reads WildFake.
The exported comparison and validation-only calibration were reviewed; v2
failed the deployment gates, so the application remains on v1.

SID_Set is listed as CC BY 4.0 on its dataset card. The pinned run retained
3,000 images per class, split into 4,800 training, 600 validation, and 600 test
images with equal class counts in every split.

WildFake was reserved until after the final checkpoint and `0.50` threshold
were locked. It was never used for training, threshold selection, or model
selection; its result is reported only as a demonstration on one external
subset.

## How we used AI and Codex

RealityCheck's AI capability is its locally hosted EfficientNet-B0 image
classifier. It produces an AIGC probability for each image, while deterministic
computer-vision transforms probe whether that probability remains stable after
real-world edits. The live demo does not call a paid inference API; it runs the
frozen checkpoint on CPU.

Codex served as an engineering and review collaborator. We used it to divide
merge-safe team tasks, implement and test the data, transformation, inference,
evaluation, and Streamlit layers, troubleshoot Kaggle and Streamlit deployment,
and independently replay exported metrics and lineage hashes. The team retained
the final decision authority: when the v2 candidate failed the predeclared
safety gates, we kept v1 rather than optimizing the story around the new model.

## How to test it

1. Install `requirements.txt` and run `streamlit run app/streamlit_app.py`.
2. Upload a JPG or PNG and select **Analyze image** to view its AIGC score.
3. Select **Run robustness stress test** to display the robustness passport and
   compare the clean score with representative compression, blur, resize,
   noise, color, and crop edits.
4. Run the documented directory inference command and confirm the JSON output
   contains only `image_path` and `pred` for each file.
5. For the automated suite, install `requirements-train.txt`, then run
   `python -m pytest -q`.

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

On the post-lock WildFake demonstration subset, the frozen model reaches
0.8554 ROC-AUC, 0.9175 average precision, 76.89% balanced accuracy, and 0.7630
F1 on clean images. Its false-positive rate is 12.12% and false-negative rate is
34.09%. This result covers the exact 4,998 COCO val2017 real and 8,843 Advanced
DALL-E 3 generated organizer rows. A byte-level audit found 1,808 same-label
duplicate groups and 8,717 unique content hashes; metrics retain all organizer
rows, so this is not a sample of 13,841 independent images. We present it as a
transparent external demonstration, not universal detector performance.

The review-gated GenImage v2 run completed, but we deliberately did not deploy
it. On the held-out 1,120-image GenImage test at threshold 0.50, v2 improves
ROC-AUC from 0.6032 to 0.8633 and balanced accuracy from 55.80% to 75.80%.
However, its SID false-positive rate rises from 4.67% to 18.67%. We then ran one
predeclared validation-only calibration with no retraining. The locked threshold
of 0.40042864 still produced a 24.67% exploratory SID false-positive rate and
failed four of eight deployment gates. We therefore retained the safer v1 model
at 0.50 rather than hiding the trade-off or tuning again on observed tests.
Because the calibration policy followed the earlier 0.50 test review, this
saved test re-score is exploratory rather than a fresh holdout.

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
- The external WildFake row set contains repeated same-label image bytes, and
  its DALL-E 3/COCO composition covers only one generated and one real source.

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

## Screenshot shot list

1. Streamlit landing page showing **Model checkpoint available**.
2. Authentic-image result with its probability and interpretation.
3. Generated-image result with its probability and interpretation.
4. Robustness passport chart and most destabilizing transformation.
5. Model-comparison figure beside the honest retain-v1 calibration decision.

## Team contributions

Replace each placeholder with the actual contributor before submission:

| Contributor | Verified contribution |
| --- | --- |
| Name TBD | Core model, training, evaluation, and integration |
| Name TBD | Dataset preparation and manifest pipeline |
| Name TBD | Deterministic robustness transformations |
| Name TBD | Streamlit robustness-passport demo |
