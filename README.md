# RealityCheck — Robust AI-Generated Image Detection

RealityCheck is a hackathon-scale image classifier that estimates whether an
image is AI-generated and measures whether that score remains stable after
ordinary social-media transformations. It treats robustness as a first-class
result rather than reporting clean-image accuracy alone.

The prototype uses an ImageNet-pretrained EfficientNet-B0 binary classifier
(well below the 2-billion-parameter limit), deterministic transformation tests,
and a CPU-friendly Streamlit demo. It is a screening aid—not proof of image
provenance.

## What makes the project useful

- One required batch command emits an AIGC confidence for every image.
- The complete challenge grid covers JPEG compression, blur, rescaling, noise,
  brightness/contrast/saturation jitter, and center cropping.
- A clean baseline and robustness-trained model are evaluated on the same held-
  out images, revealing whether augmentation improves stability rather than only
  clean performance.
- The interactive **robustness passport** shows how one upload's score changes
  under every configured edit and identifies the most destabilizing case.
- Dataset preparation is balanced, duplicate-aware, source-grouped, and small
  enough for a free Kaggle GPU workflow.
- A shared short-edge resize and center crop preserves aspect ratio across
  training, evaluation, CLI inference, and Streamlit instead of stretching
  differently shaped source images into a class-correlated shortcut.

## Zero-cost architecture

```text
Hugging Face SID_Set stream
        │  6,000 selected images (labels 0 and 1 only)
        ▼
grouped train / val / test manifest
        │
        ├── clean EfficientNet-B0 baseline
        └── robustness-trained EfficientNet-B0 ──► final .safetensors checkpoint
                                                    │
                 full clean + transform evaluation ◄┘
                                                    │
                        CLI JSON + Streamlit CPU demo
```

- Training/evaluation: Kaggle Notebook with its free GPU quota.
- Demo hosting: Streamlit Community Cloud free tier.
- Data: streamed from the public
  [`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set) dataset;
  the script stops once the requested balanced subset is complete instead of
  downloading the full 140 GB collection.
- No paid API, database, storage service, or inference endpoint is required.

Free services can impose quotas, sleep when idle, or change availability. The
local demo and recorded video remain the reliable submission fallback.

## Verified held-out results

The final controlled Kaggle run used 6,000 balanced SID_Set images and evaluated
both models on the same 600-image test split under the complete 20-scenario
clean and transformation grid.

| Result | Clean baseline | Robustness-trained | Robust-model difference |
| --- | ---: | ---: | ---: |
| Clean ROC-AUC | 0.997056 | 0.996400 | -0.000656 |
| Mean transformed ROC-AUC | 0.995635 | 0.995784 | +0.000149 |
| Worst transformed ROC-AUC | 0.986133 | 0.990278 | +0.004144 |
| Mean transformed balanced accuracy | 95.47% | 95.83% | +0.36 pp |

The defensible gain is improved worst-case resilience, not universal model
superiority. Both models are weakest under Gaussian noise at `sigma=0.10`; the
robust model improves balanced accuracy there from 84.67% to 89.00%. It is
slightly worse on clean ROC-AUC and wins ROC-AUC in 8 of 20 scenarios. See the
full [`evaluation and error analysis`](docs/submission/error-analysis.md), the
[`controlled model comparison`](artifacts/figures/model_comparison.png), and the exported
[`run context`](artifacts/metrics/run_context.json).

![Controlled comparison of the clean-training baseline and robustness-trained model](artifacts/figures/model_comparison.png)

### Post-lock external demonstration result

After checkpoint and threshold lock, the clean WildFake demonstration run
evaluated the exact 13,841-row organizer subset (4,998 COCO val2017 real and
8,843 Advanced DALL-E 3 generated):

| ROC-AUC | Average precision | Balanced accuracy | F1 | FPR | FNR | Brier score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8554 | 0.9175 | 0.7689 | 0.7630 | 0.1212 | 0.3409 | 0.1922 |

At the unchanged `0.50` threshold, the confusion counts are 4,392 true
negatives, 606 false positives, 3,015 false negatives, and 5,828 true
positives. This is a demonstration on one external subset, not a universal
performance claim. The byte-level audit found 1,808 same-label duplicate groups
containing 5,124 additional file rows, leaving 8,717 unique content hashes;
metrics retain every organizer row to preserve the exact requested subset and
therefore do not represent 13,841 independent images.

See the sanitized [`WildFake demonstration report`](docs/submission/wildfake-demo-report.md),
[`aggregate JSON`](artifacts/metrics/wildfake_demo_summary.json), and
[`clean-result figure`](artifacts/figures/wildfake_demo.png). WildFake was used
only after model and threshold lock and never for training, threshold selection,
or model selection.

## Setup and installation

### Prerequisites

- Git and Python 3.11.
- About 2 GB of free space for the repository, virtual environment, and CPU
  dependencies. External datasets require additional space and are not included
  in Git.
- A Kaggle T4 GPU is recommended only for retraining. The included Streamlit
  demo and checkpoints run on CPU.

Clone the public repository and enter its root directory:

```bash
git clone https://github.com/LINGSIHAN/TikTok-Hackathon-Track-5.git
cd TikTok-Hackathon-Track-5
```

On macOS or Linux, create the environment and install the CPU demo dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, use:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch the demo:

```bash
streamlit run app/streamlit_app.py
```

No API key, database, or Streamlit secret is required. The app loads the tracked
GenImage v2 checkpoint on CPU and displays
`Model checkpoint available · GenImage v2` in the sidebar. For dataset
preparation, evaluation, training, and the complete test suite, additionally
install:

```bash
python -m pip install -r requirements-train.txt
python -m pytest -q -p no:cacheprovider
```

The expected test result for this revision is **256 passed**. Platform-specific
troubleshooting, PowerShell activation recovery, Kaggle GPU steps, and dataset
preparation details are in [`SETUP.md`](SETUP.md).

## Repository layout

```text
app/                  Streamlit robustness-passport demo
configs/              Clean and robustness-training configurations
data/                 Provenance docs and ignored runtime images/manifests
notebooks/            Fail-closed Kaggle training/evaluation notebook
scripts/              Dataset and presentation helpers
src/data/             Validated manifest-backed dataset
src/transforms/       Deterministic challenge transformation grid
src/models/           EfficientNet-B0 detector and safe checkpoint helpers
src/training/         Config-driven training and early stopping
src/evaluation/       Metrics, complete robustness evaluation, and plot
src/inference/        Predictor API and required directory-to-JSON CLI
tests/                Unit and contract tests
artifacts/            Selected checkpoint and compact public evidence
```

## Reproduce the workflow

Create an environment and install the training dependencies (platform-specific
activation commands are in [`SETUP.md`](SETUP.md)):

```bash
python -m pip install -r requirements-train.txt
```

Stream a balanced 6,000-image subset and create `train`, `val`, and `test`
splits. Labels are 0 (real) and 1 (fully synthetic); SID label 2 (tampered) is
excluded.

```bash
python scripts/prepare_sid_subset.py --total 6000 --seed 42
```

Train the baseline and robustness-aware model:

The two configurations use the same seed, data, trainable backbone block,
optimizer settings, and epoch budget. Their only experimental difference is
whether robustness transformations are sampled during training.

```bash
python -m src.training.train --config configs/train_clean.yaml
python -m src.training.train --config configs/train_robust.yaml
```

Evaluate the selected checkpoint on clean images and every published transform
severity:

```bash
python -m src.evaluation.evaluate \
  --manifest data/processed/manifest.csv \
  --checkpoint artifacts/checkpoints/model.safetensors \
  --split test \
  --output-dir artifacts/metrics
```

### Post-lock WildFake demonstration benchmark

The external benchmark is deliberately separate from training and internal
model selection. Its downloader pins WildFake revision
`18f53ff36ad9da60644039f0452b0e7b3907af6f`, verifies the two source metadata
files, and retrieves only the contiguous ZIP ranges containing 4,998 COCO
`val2017` images and 8,843 Advanced DALL-E 3 images. Original encoded image
bytes are preserved.

```bash
python scripts/download_wildfake_demo.py
python scripts/evaluate_wildfake_demo.py
```

The recommended command evaluates clean images only with the frozen checkpoint
and fixed `0.50` threshold. The optional complete 20-scenario run is:

```bash
python scripts/evaluate_wildfake_demo.py --mode full
```

Local manifests, predictions, raw metrics, execution metadata, and all images
remain ignored. A completed evaluation generates a sanitized aggregate JSON,
Markdown report, and figure suitable for public evidence. The evaluator rejects
changed checkpoint bytes, wrong class counts, duplicate or conflicting image
hashes, corrupt images, and a dataset digest that differs from the downloader's
verified manifest.

For the intended free GPU run, upload
[`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb) to Kaggle, enable
a GPU and internet, and run all cells. The notebook validates both model runs,
the metrics, and a checkpoint inference before creating `hackathon_export.zip`.
Extract that ZIP into the repository root; `.gitignore` exposes only the compact
final checkpoint, manifest/provenance, metrics, predictions, and plots that are
useful for the public submission. Raw images and intermediate runs stay ignored.

### GenImage v2 training and promotion workflow

The separate
[`train_genimage_v2_kaggle.ipynb`](notebooks/train_genimage_v2_kaggle.ipynb)
workflow warm-starts a **candidate** EfficientNet-B0 from the frozen v1
checkpoint. It combines the 4,800 SID training rows with a deterministic,
balanced 11,200-image subset of Kaggle's
[`cartografia/unbiased-tiny-genimage`](https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage)
version 1. GenImage validation and test rows remain held out, and SID validation
and test rows are never added to v2 training.

In Kaggle, enable a T4 GPU and Internet, attach the dataset with **Add Input**,
then choose **Run All**. The notebook verifies the exact source inventory,
recreates the pinned SID sample, normalizes and deduplicates GenImage, runs tests
and a CUDA smoke check, trains for at most three epochs, compares v1 and v2 on
both held-out tests under all 20 scenarios, and creates:

```text
/kaggle/working/genimage_v2_export.zip
```

The export contains `local_audit/` (checkpoint, manifests, predictions, raw
metrics, hashes, and environment) and `public/` (sanitized JSON, Markdown, and
figure). The workflow never auto-promotes v2. Manual review and validation-only
calibration recommended retaining v1 because v2 failed four of eight
predeclared deployment gates. The completed comparison is recorded in the
[`aggregate summary`](artifacts/metrics/genimage_v2_summary.json),
[`review report`](docs/submission/genimage-v2-report.md), and
[`comparison figure`](artifacts/figures/genimage_v2_comparison.png). Full run
instructions are in [`SETUP.md`](SETUP.md).

Because the reviewed v2 checkpoint ranks SID images well but is miscalibrated
at `0.50`, the separate
[`calibrate_genimage_v2_kaggle.ipynb`](notebooks/calibrate_genimage_v2_kaggle.ipynb)
workflow performs **no retraining**. It chooses the lowest-SID-false-positive
threshold from clean validation predictions under fixed SID-recall and
GenImage-performance safeguards, then exploratorily re-scores the locked
threshold on the saved, previously inspected test predictions. Those tests are
not presented as a fresh unbiased holdout.

The single predeclared calibration run is complete. It locked threshold
`0.40042864`, but the exploratory SID test false-positive rate was still
24.67%, and four of eight deployment gates failed. No second tuning attempt is
permitted. After reviewing that recommendation, the project owner explicitly
selected v2 at the original evaluated threshold `0.50` for the broader-generator
Streamlit demonstration on 2026-09-01, accepting and disclosing its higher SID
false-positive rate. V1 remains the rollback baseline. The reviewed outputs are
the
[`aggregate calibration evidence`](artifacts/metrics/genimage_v2_calibration.json)
and [`calibration report`](docs/submission/genimage-v2-calibration-report.md).
The completed decision and remaining deadline steps are recorded in
[`docs/submission/final-runbook.md`](docs/submission/final-runbook.md).

### Reproduction verification checklist

Run commands from the repository root. Python 3.11 is the local reference;
the recorded Kaggle runs used Python 3.12.13, PyTorch 2.10.0+cu128, and a Tesla
T4. Install `requirements-train.txt` before reproducing data preparation,
training, evaluation, or tests.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
python -m pytest -q -p no:cacheprovider
```

The current repository expectation is **256 passing tests**. Windows activation
commands and the complete Kaggle setup are documented in [`SETUP.md`](SETUP.md).

| Evidence being reproduced | Required input and action | Expected verification |
| --- | --- | --- |
| SID preparation | Run `python scripts/prepare_sid_subset.py --total 6000 --seed 42`. | 6,000 images: 4,800 train, 600 validation, and 600 test, balanced by class. `data/processed/manifest.csv` SHA-256: `6267a8d7e7749c1870601e196fe7ce1cc0fc2542a9975fa939832817e7fd3d9d`. |
| Controlled v1 comparison | In a fresh checkout or Kaggle workspace, run all cells in `train_kaggle.ipynb`; do not retrain over the tracked checkpoints in the submission checkout. | Clean ROC-AUC `0.996400`; mean transformed ROC-AUC `0.995784`; worst transformed ROC-AUC `0.990278`; mean transformed balanced accuracy `95.83%`. Evidence: [`metrics.json`](artifacts/metrics/metrics.json) and [`model comparison`](artifacts/figures/model_comparison.png). |
| WildFake clean demonstration | Run the downloader and default evaluator above after SID model lock. | 13,841 rows: 4,998 real and 8,843 generated. ROC-AUC `0.8554`, average precision `0.9175`, balanced accuracy `76.89%`, F1 `0.7630`. Evidence: [`summary JSON`](artifacts/metrics/wildfake_demo_summary.json) and [`report`](docs/submission/wildfake-demo-report.md). |
| GenImage v2 run | In Kaggle, attach `cartografia/unbiased-tiny-genimage` version 1 and run all cells in `train_genimage_v2_kaggle.ipynb`. | `genimage_v2_export.zip`; v2 SHA-256 `b45022d9dab2a02300934c239eee24dd40ef8e402f24c1f27fc2d63a46117c12`. On held-out GenImage, clean ROC-AUC `0.8633` and balanced accuracy `75.80%`; mean transformed ROC-AUC `0.8636`. Evidence: [`summary JSON`](artifacts/metrics/genimage_v2_summary.json) and [`report`](docs/submission/genimage-v2-report.md). |
| SID regression for v2 | Produced by the same v2 notebook on the untouched 600-image SID test split. | Clean ROC-AUC `0.9931`, balanced accuracy `90.00%`, FPR `18.67%`, and FNR `1.33%`. This documents the false-positive regression rather than hiding it. |
| Validation-only calibration audit | Inspect the locked calibration JSON and report; do not run another tuning attempt or alter the threshold from observed tests. | Selected validation threshold `0.40042864`; exploratory SID test FPR `24.67%`; four of eight deployment gates failed. Evidence: [`calibration JSON`](artifacts/metrics/genimage_v2_calibration.json) and [`calibration report`](docs/submission/genimage-v2-calibration-report.md). |
| Streamlit deployment | Run `streamlit run app/streamlit_app.py`, upload one authentic and one generated image, and run the robustness passport for each. | Sidebar shows `Model checkpoint available · GenImage v2`; predictions use `model_v2.safetensors` at boundary `0.50`. |

Verify the two tracked model identities without modifying either checkpoint:

```bash
shasum -a 256 \
  artifacts/checkpoints/model.safetensors \
  artifacts/checkpoints/model_v2.safetensors
```

Expected hashes, in order:

```text
806fbabc5ecae8394369d08738cbf0c993568137d323a8133167e4557d04eed2  model.safetensors
b45022d9dab2a02300934c239eee24dd40ef8e402f24c1f27fc2d63a46117c12  model_v2.safetensors
```

Evaluation of the committed checkpoints against the pinned manifests should
match the published evidence, apart from insignificant floating-point display
rounding. A fresh GPU retraining run can vary slightly because hardware kernels
are not guaranteed to be bit-for-bit identical. WildFake remains
demonstration-only, and its outcome must never be used to retrain, select, or
retune a model.

## Required batch inference

```bash
python -m src.inference.cli \
  --input path/to/images \
  --output predictions.json \
  --checkpoint artifacts/checkpoints/model_v2.safetensors
```

The output is a deterministically ordered JSON array whose records contain
exactly:

```json
{"image_path": "path/to/image.png", "pred": 0.8732}
```

`pred` is the model's estimated probability that the image is AI-generated.

## Run and host the demo

The reviewed Streamlit deployment uses
`artifacts/checkpoints/model_v2.safetensors` by default. The frozen v1 file
remains at `artifacts/checkpoints/model.safetensors` for rollback and historical
evaluation.

```bash
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

For Streamlit Community Cloud, select this GitHub repository and set the app
entrypoint to `app/streamlit_app.py`. No secrets are required. Confirm the
public URL with both an authentic and generated test image before recording the
demo; deployment is not considered verified until that smoke test passes.

## Data isolation and provenance

The SID_Set dataset card lists CC BY 4.0 and documents labels 0 (real), 1 (full
synthetic), and 2 (tampered). Full provenance, normalization, leakage controls,
and attribution notes are in [`data/README.md`](data/README.md).

The organizer-provided WildFake demonstration subset is reserved for post-lock
demonstration/evaluation. It was not and must not be used for training,
threshold selection, or model selection. See the exact immutable subset and
audit rules in [`data/README.md`](data/README.md).

The v2 model uses Unbiased Tiny GenImage version 1 only after an explicit
licence confirmation. Its prepared data and detailed manifests stay ignored,
and WildFake was excluded from training and model selection. Validation-only
calibration did not meet the predeclared false-positive gate; that result
remains published even though the project owner subsequently selected v2 for
the Streamlit demonstration. Only the reviewed checkpoint and compact lineage
metadata are tracked for deployment.

## Limitations and future improvements

RealityCheck is a screening prototype, not a provenance authority. Its score is
an estimate of whether an image resembles the training data; it cannot prove who
created an image, identify the exact generator, or establish malicious intent.
Scores close to the `0.50` boundary should be treated as uncertain evidence.

Current limitations:

- **Cross-dataset false positives:** v2 generalizes much better to the held-out
  GenImage generators, but its SID false-positive rate increased from 4.67% to
  18.67% at the fixed `0.50` boundary. The validation-only calibration also
  failed four of eight deployment gates. This trade-off is disclosed rather
  than presented as universal improvement.
- **Limited generator coverage:** SID and the selected GenImage subset cannot
  represent every current or future image generator, editing pipeline,
  screenshot process, illustration style, camera, or social-media platform.
- **External benchmark scope:** WildFake contains only the exact organizer
  subset used for the post-lock demonstration and includes same-label duplicate
  content. Its result is useful evidence, but it is not a universal estimate of
  real-world accuracy.
- **Transformation coverage:** the 20 scenarios test common compression, blur,
  resize, noise, colour, and crop operations individually. They do not cover
  every combination, repeated re-upload, screenshot, watermark, overlay, or
  adversarial manipulation.
- **Confidence calibration:** an AIGC score is a model output rather than a
  guaranteed real-world probability. One global boundary may behave differently
  across image domains.
- **Prototype operations:** the free-tier CPU deployment can be slower under
  load and has no production monitoring, human-review queue, abuse controls, or
  formal privacy audit.

Given more time, we would:

1. Build a larger, properly licensed cross-source dataset containing newer
   generators, diverse authentic cameras, screenshots, illustrations, and
   multi-step social-media edits.
2. Reserve a new untouched calibration set and add an explicit **inconclusive**
   range around the decision boundary, prioritizing a lower false-positive rate
   for real images.
3. Evaluate combined transformations, repeated re-encoding, unseen platforms,
   and generator-family holdouts instead of relying only on single-operation
   stress tests.
4. Compare the EfficientNet classifier with complementary forensic signals and
   provenance metadata, while keeping the final decision interpretable.
5. Conduct user testing with journalists, moderators, and ordinary users to
   improve warnings, accessibility, confidence explanations, and human-review
   workflows.
6. Add production monitoring for distribution drift, latency, failures, and
   false-positive reports before considering consequential use.

## Current verified status

- Core training, inference, transformation, evaluation, and Streamlit code is
  integrated.
- The final Kaggle T4 run completed against repository commit `e32257d`. Its
  selected checkpoint, manifest, metrics, predictions, provenance, and figures
  are included in the public repository.
- The 6,000-row manifest is balanced and duplicate-free, with no source ID
  crossing train, validation, or test splits. Both 12,000-row prediction files
  cover all 20 scenarios and all exported metrics were independently recomputed
  with zero discrepancy.
- The Kaggle pipeline passed its CUDA checkpoint smoke test before packaging.
  The retained v1 checkpoint also passes local CPU loading/inference, the
  required directory-to-JSON CLI, the complete repository test suite, and a
  headless Streamlit health check. The remaining external gates are a
  fresh-environment setup check, verified contributor details, public Streamlit
  deployment, two-image live smoke test, demo video, and final Devpost links.
- The GenImage v2 Kaggle run completed against commit `34cc2fb`. All archive,
  checkpoint-lineage, manifest, prediction, and metric hashes passed independent
  validation. Its checkpoint SHA-256 is
  `b45022d9dab2a02300934c239eee24dd40ef8e402f24c1f27fc2d63a46117c12`.
  At threshold `0.50`, clean GenImage ROC-AUC improved from 0.6032 to 0.8633
  and balanced accuracy from 55.80% to 75.80%, while SID false positives rose
  from 4.67% to 18.67%.
- The separate validation-only calibration completed against commit `2de9d2b`
  and locked threshold `0.40042864` without retraining. Its exploratory SID test
  false-positive rate was 24.67%, balanced accuracy was 87.17%, and SID
  transformed mean/worst balanced accuracy was 88.18%/82.67%. Four of eight
  deployment gates failed. The recommendation remains part of the evidence;
  the project owner later directed a transparent v2 demo deployment at `0.50`,
  with v1 preserved for rollback.

See [`docs/submission/requirements-checklist.md`](docs/submission/requirements-checklist.md)
for the remaining evidence and submission gates.

## Team contributions

### Toh Wei Jun — Robustness and Preprocessing Engineer

- Implemented deterministic JPEG compression, blur, resize, noise, colour-jitter, and cropping transformations.
- Helped align image preprocessing across training, evaluation, and inference.
- Set up the local machine-learning environment.
- Tested transformations for repeatability and correct output.

### Jonah Wee — Dataset, Training, and Deployment Engineer

- Prepared and evaluated the WildFake demonstration subset.
- Developed the GenImage v2 Kaggle training workflow.
- Trained and compared the v1 and v2 checkpoints.
- Integrated the selected v2 checkpoint into the Streamlit application.
- Verified model hashes, dataset isolation, and deployment behaviour.

### Saichandar — Evaluation and Quality-Assurance Engineer

- Designed test cases for authentic, generated, and transformed images.
- Reviewed model metrics, confusion counts, false positives, and false negatives.
- Performed manual testing of the Streamlit upload and robustness-test workflow.
- Helped document limitations, test findings, and reproducibility steps.
- Supported demo preparation and presentation testing.

### Ling Si Han — Project Lead and Application Integration Engineer

- Developed and integrated the core EfficientNet-B0 detection pipeline.
- Built the Streamlit robustness-passport interface.
- Integrated the training, inference, evaluation, and data-processing components.
- Developed the validation-only calibration workflow and reviewed deployment trade-offs.
- Maintained repository integration, documentation, and release readiness.
