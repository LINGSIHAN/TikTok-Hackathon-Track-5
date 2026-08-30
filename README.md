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

For the intended free GPU run, upload
[`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb) to Kaggle, enable
a GPU and internet, and run all cells. The notebook validates both model runs,
the metrics, and a checkpoint inference before creating `hackathon_export.zip`.
Extract that ZIP into the repository root; `.gitignore` exposes only the compact
final checkpoint, manifest/provenance, metrics, predictions, and plots that are
useful for the public submission. Raw images and intermediate runs stay ignored.

## Required batch inference

```bash
python -m src.inference.cli \
  --input path/to/images \
  --output predictions.json \
  --checkpoint artifacts/checkpoints/model.safetensors
```

The output is a deterministically ordered JSON array whose records contain
exactly:

```json
{"image_path": "path/to/image.png", "pred": 0.8732}
```

`pred` is the model's estimated probability that the image is AI-generated.

## Run and host the demo

After placing the trained checkpoint at
`artifacts/checkpoints/model.safetensors`:

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

The organizer-provided WildFake demonstration subset is reserved for final
demonstration/evaluation. It must not be used for training, threshold selection,
or model selection.

## Current verified status

- Core training, inference, transformation, evaluation, and Streamlit code is
  integrated.
- The data pipeline passes a direct synthetic-fixture smoke test for exact class
  balance, deterministic splits, source isolation, RGB loading, and manifest
  safety.
- The final Kaggle training run has **not** happened yet. Consequently, no
  checkpoint, accuracy claim, robustness result, public Streamlit URL, or error
  analysis is claimed here.
- The next gate is the complete Kaggle notebook run; measured artifacts then
  feed the README results, error analysis, Devpost write-up, and demo video.

See [`docs/submission/requirements-checklist.md`](docs/submission/requirements-checklist.md)
for the remaining evidence and submission gates.
