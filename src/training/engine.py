"""Small, testable PyTorch training primitives."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import compute_binary_metrics


@dataclass(frozen=True)
class EpochResult:
    """Aggregate loss, predictions, and metrics from one dataset pass."""

    loss: float
    labels: list[int]
    probabilities: list[float]
    metrics: dict[str, float]


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNGs inside a PyTorch data-loader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when present and validate explicit choices."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_grad_scaler(enabled: bool) -> Any:
    """Create a GradScaler across supported PyTorch versions."""

    active = bool(enabled and torch.cuda.is_available())
    try:
        return torch.amp.GradScaler("cuda", enabled=active)
    except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
        return torch.cuda.amp.GradScaler(enabled=active)


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        images = batch.get("image", batch.get("images"))
        labels = batch.get("label", batch.get("labels"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, labels = batch[0], batch[1]
    else:
        raise TypeError(
            "A batch must be a mapping with image/label keys or a two-item sequence"
        )
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Batch images and labels must be torch tensors")
    return images, labels


def extract_logits(output: Any) -> torch.Tensor:
    """Extract a one-logit-per-image tensor from common model outputs."""

    if isinstance(output, Mapping):
        for key in ("logits", "output", "predictions"):
            if key in output:
                output = output[key]
                break
    elif isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model returned an empty sequence")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model output must contain a torch tensor")
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    elif output.ndim != 1:
        raise ValueError(
            f"Expected one logit per image with shape [B] or [B, 1], got {tuple(output.shape)}"
        )
    return output


def _set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Disable running-stat updates for fully frozen BatchNorm modules.

    Calling ``model.train()`` normally puts every BatchNorm layer into training
    mode even when its affine parameters are frozen.  Warm-start fine-tuning
    can opt into retaining the parent model's running statistics for only
    those BatchNorm modules that have no trainable direct parameters.
    """

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        ):
            module.eval()


def run_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    mixed_precision: bool = False,
    scaler: Any | None = None,
    criterion: nn.Module | None = None,
    threshold: float = 0.5,
    freeze_frozen_batchnorm: bool = False,
) -> EpochResult:
    """Run one training or evaluation epoch.

    Passing an optimizer enables training; omitting it runs inference under
    ``no_grad`` semantics. Mixed precision is activated only for CUDA.
    """

    training = optimizer is not None
    model.train(training)
    if training and freeze_frozen_batchnorm:
        _set_frozen_batchnorm_eval(model)
    criterion = criterion or nn.BCEWithLogitsLoss()
    use_amp = bool(mixed_precision and device.type == "cuda")
    if training and use_amp and scaler is None:
        scaler = make_grad_scaler(True)

    total_loss = 0.0
    total_samples = 0
    all_labels: list[int] = []
    all_probabilities: list[float] = []

    for batch in loader:
        images, labels = _unpack_batch(batch)
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda").float().reshape(-1)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = extract_logits(model(images))
                if logits.shape[0] != labels.shape[0]:
                    raise ValueError("Model output and labels have different batch sizes")
                loss = criterion(logits, labels)

            if training:
                if use_amp:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probabilities = torch.sigmoid(logits.detach()).cpu().numpy()
        label_values = labels.detach().cpu().to(torch.int64).numpy()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        all_probabilities.extend(float(value) for value in probabilities)
        all_labels.extend(int(value) for value in label_values)

    if total_samples == 0:
        raise ValueError("Cannot run an epoch over an empty data loader")

    metrics = compute_binary_metrics(all_labels, all_probabilities, threshold=threshold)
    return EpochResult(
        loss=total_loss / total_samples,
        labels=all_labels,
        probabilities=all_probabilities,
        metrics=metrics,
    )


class EarlyStopping:
    """Track the best validation loss and retain its model weights in memory."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.best_epoch: int | None = None
        self._best_state: dict[str, torch.Tensor] | None = None

    def update(self, loss: float, model: nn.Module, epoch: int) -> bool:
        """Observe ``loss`` and return ``True`` when training should stop."""

        if not np.isfinite(loss):
            raise ValueError("Validation loss must be finite")
        if loss < self.best_loss - self.min_delta:
            self.best_loss = float(loss)
            self.bad_epochs = 0
            self.best_epoch = int(epoch)
            self._best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience if self.patience > 0 else True

    def restore(self, model: nn.Module) -> None:
        """Restore the best observed model weights."""

        if self._best_state is None:
            raise RuntimeError("No best model state has been recorded")
        model.load_state_dict(self._best_state)
