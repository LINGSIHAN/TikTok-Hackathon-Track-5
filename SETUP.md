# Local setup

This repository uses a project-local Python environment. It has already been
created at `.venv` with the dependencies in `requirements.txt`.

```bash
source .venv/bin/activate
python --version
```

For any Kaggle command in this project, keep the optional Kaggle configuration
in the ignored local folder rather than your home directory:

```bash
export KAGGLE_CONFIG_DIR="$PWD/.kaggle"
kaggle --version
```

Do not add a Kaggle token unless you choose to download data locally. Training
can instead be run directly in a Kaggle Notebook without placing credentials
in this repository.

To reproduce the lightweight inference and Streamlit environment on another
machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For dataset preparation, training, evaluation, and tests, install the extended
requirements instead:

```bash
python -m pip install -r requirements-train.txt
```

## Included tools

- **PyTorch, TorchVision** — EfficientNet-B0 training and inference.
- **NumPy and Pillow** — JPEG, blur, resize, noise, colour, and crop transforms.
- **datasets, Hugging Face Hub** — streaming a small, balanced SID_Set subset.
- **Kaggle client** — optional CIFAKE access and Kaggle workflow support.
- **scikit-learn, pandas, Matplotlib** — metrics, robustness tables, and figures.
- **Streamlit** — lightweight local and free-hosted demo.
- **pytest, PyYAML, safetensors** — validation, configuration, and compact model checkpoints.

## Dataset guardrails

Downloaded data belongs under `data/raw/` and generated intermediates under
`data/processed/`; both are deliberately ignored by Git. Do not train or tune
on the organiser-provided WildFake COCO/DALL-E demonstration subset. Use that
subset only for final demonstration/evaluation.

The intended training path is a streamed, balanced SID_Set subset (initially
5,000 authentic + 5,000 fully synthetic images); exclude the tampered class.
Use CIFAKE only for pipeline smoke tests or supplementary experiments.

## Working locally

```bash
source .venv/bin/activate
python -c "import torch, timm; print(torch.__version__)"
streamlit --version
```

This machine does not currently expose an accelerator to PyTorch, so use it
for setup, small smoke tests, and the demo. Run model training on Kaggle's
free GPU environment and keep only the selected final checkpoint and metrics.
