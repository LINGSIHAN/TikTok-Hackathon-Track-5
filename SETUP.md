# Setup

Python 3.11 is the reference version. The repository does not include or assume
an existing virtual environment.

## Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for the current process, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` is sufficient for CPU inference and Streamlit. For dataset
preparation, Kaggle training, evaluation, and tests, install the extended set:

```bash
python -m pip install -r requirements-train.txt
```

Verify the environment with packages that are actually used by the project:

```bash
python -c "import torch, torchvision, streamlit; print(torch.__version__)"
python -m pytest --version
```

## Data preparation

The recommended command streams a shuffled, balanced subset from
[`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set):

```bash
python scripts/prepare_sid_subset.py --total 6000 --seed 42
```

Internet access is required while this command runs. Hugging Face streaming
reads only enough source records to fill 3,000 real and 3,000 fully synthetic
slots; it does not intentionally download the complete 140 GB dataset. Runtime
images and manifests remain ignored by Git, except for compact final provenance
and evidence files explicitly allowlisted in `.gitignore`.

The command writes:

- `data/raw/authentic/` and `data/raw/generated/`
- `data/processed/manifest.csv`
- `data/processed/manifest_summary.json`

It fails if the requested balanced sample cannot be filled. Do not use the
organizer-provided WildFake demonstration subset for training, threshold
selection, or model selection.

## Kaggle GPU run

The reproducible free-training path is
`notebooks/train_kaggle.ipynb`. In Kaggle:

1. Create a notebook and import the repository notebook.
2. In **Settings**, choose **GPU T4 x2** when it is available. The code uses one
   T4. Current CUDA 12.8 PyTorch builds may omit the P100's `sm_60` kernels; the
   notebook runs a real CUDA operation immediately and stops with a targeted
   recovery message before it touches prepared data.
3. Enable internet access so the notebook can clone the repository, install
   dependencies, and stream SID_Set.
4. Run all cells. On a retry, the notebook fast-forwards an existing trusted
   clone and reuses a complete validated 6,000-image subset instead of deleting
   and downloading it again.
5. Download `hackathon_export.zip` only after the final validation cell passes.

If the 6,000 prepared images already exist and only training must be repeated,
run this in one fresh Kaggle code cell. It validates every stored image hash,
reuses the subset, retrains both controlled configurations, and recreates the
ZIP without downloading SID_Set again:

```python
%cd /kaggle/working/TikTok-Hackathon-Track-5
!git pull --ff-only origin master
!python -m pip install -q -r requirements-train.txt
!python scripts/retrain_kaggle_subset.py
```

Wait for `Validated export ready:` before downloading the replacement archive.
Run the script with `!python`, not `%run`, so training and validation use a fresh
interpreter.

No Kaggle API token or paid service is needed for this path. Free GPU quota and
accelerator availability are controlled by Kaggle.

## Kaggle GenImage v2 candidate run

Use [`notebooks/train_genimage_v2_kaggle.ipynb`](notebooks/train_genimage_v2_kaggle.ipynb)
for the separate warm-start experiment. This run does not replace the v1
checkpoint and does not use WildFake.

1. Push the implementation to GitHub, then import the v2 notebook into Kaggle.
2. In **Settings**, select a T4 GPU and enable Internet.
3. Choose **Add Input** and attach
   [`cartografia/unbiased-tiny-genimage`](https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage)
   version 1.
4. Confirm that your intended use complies with the GenImage and upstream
   ImageNet terms represented by the notebook's licence-confirmation flag.
5. Select **Run All**. Do not run the training cell by itself: the preceding
   inventory, checkpoint, test, and CUDA gates are intentional.
6. Wait for `Validated GenImage v2 export ready:` and download
   `/kaggle/working/genimage_v2_export.zip`.
7. Review `public/genimage-v2-report.md` and its compact JSON/figure. Keep the
   detailed `local_audit/` layer for reproducibility before deciding whether to
   deploy v2.

The attached input must match the pinned inventory exactly: 2,500 images from
each of seven generators, 5,828 Nature images, the fixed metadata digest,
23,329 files, and 2,528,629,592 bytes. The workflow selects 5,600 real and
5,600 generated images with seed 42, creates balanced 80/10/10 splits, and adds
only the 4,800 SID training rows to the v2 training split. A mismatch stops the
run instead of silently training on different data.

The candidate uses the frozen v1 checkpoint as initialization, not as an exact
resume, because v1 has no optimizer state. Training uses three epochs at most,
early-stopping patience 1, AdamW at `3e-5`, batch size 64, mixed precision, and
the existing 35% clean / 65% transformed sampler. Evaluation keeps the threshold
at `0.50` and compares both models across all 20 scenarios on the 1,120-image
GenImage test and 600-image SID regression test.

Expect roughly 1–2 hours on a T4. SID streaming, Kaggle storage speed, and the
four complete transformation evaluations are the main sources of variation.

## Local checks and demo

```bash
python -m pytest
streamlit run app/streamlit_app.py
```

Without `artifacts/checkpoints/model.safetensors`, the Streamlit interface loads
but deliberately disables predictions. Use local CPU work for tests and the
demo; run the full training notebook on Kaggle's GPU.
