"""Safe, pickle-free model checkpoint helpers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


DEFAULT_METADATA = {
    "architecture": "efficientnet_b0_binary",
    "output": "aigc_logit",
}


def _checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix.lower() != ".safetensors":
        raise ValueError("checkpoint path must end in .safetensors")
    return checkpoint_path


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Save model weights in the non-executable safetensors format."""

    checkpoint_path = _checkpoint_path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict: OrderedDict[str, Tensor] = OrderedDict(
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in model.state_dict().items()
    )
    checkpoint_metadata = dict(DEFAULT_METADATA)
    if metadata:
        checkpoint_metadata.update(
            {str(key): str(value) for key, value in metadata.items()}
        )

    save_file(state_dict, str(checkpoint_path), metadata=checkpoint_metadata)
    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: str = "cpu",
    strict: bool = True,
) -> dict[str, str]:
    """Load safetensors weights into ``model`` and return file metadata."""

    checkpoint_path = _checkpoint_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    # Loading onto CPU first avoids a transient second model-sized allocation
    # on a constrained deployment GPU. Predictor moves the model afterwards.
    state_dict = load_file(str(checkpoint_path), device="cpu")
    model.load_state_dict(state_dict, strict=strict)

    with safe_open(
        str(checkpoint_path), framework="pt", device="cpu"
    ) as checkpoint:
        metadata = checkpoint.metadata() or {}

    # ``device`` is accepted to keep this helper convenient for callers that
    # load and move in one step. Predictor handles its own move explicitly.
    model.to(device)
    return dict(metadata)
