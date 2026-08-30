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
2. In **Settings**, enable an available GPU accelerator.
3. Enable internet access so the notebook can clone the repository, install
   dependencies, and stream SID_Set.
4. Run all cells from a clean session.
5. Download `hackathon_export.zip` only after the final validation cell passes.

No Kaggle API token or paid service is needed for this path. Free GPU quota and
accelerator availability are controlled by Kaggle.

## Local checks and demo

```bash
python -m pytest
streamlit run app/streamlit_app.py
```

Without `artifacts/checkpoints/model.safetensors`, the Streamlit interface loads
but deliberately disables predictions. Use local CPU work for tests and the
demo; run the full training notebook on Kaggle's GPU.
