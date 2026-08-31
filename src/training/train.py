"""Config-driven clean or robustness-augmented training entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.preprocessing import (
    PREPROCESSING_CONTRACT_ID,
    PREPROCESSING_METADATA_KEY,
    build_image_preprocess,
    normalize_pil_image,
    resolve_checkpoint_image_size,
)
from src.training.config import ExperimentConfig, load_config
from src.training.engine import (
    EarlyStopping,
    make_grad_scaler,
    resolve_device,
    run_epoch,
    seed_worker,
    set_global_seed,
)


class TrainingImageTransform:
    """Apply optional robustness sampling followed by model preprocessing."""

    def __init__(
        self,
        image_size: int,
        *,
        robust: bool,
        clean_probability: float,
    ) -> None:
        self.robust = robust
        self.clean_probability = clean_probability
        self.preprocess = build_image_preprocess(image_size)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = normalize_pil_image(image)
        if self.robust:
            from src.transforms.robustness import sample_training_transform

            image, _ = sample_training_transform(
                image,
                seed=random.randrange(0, 2**32),
                clean_probability=self.clean_probability,
            )
        return self.preprocess(image)


def create_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build deterministic train/validation loaders from the manifest."""

    from src.data.dataset import ImageManifestDataset

    train_transform = TrainingImageTransform(
        config.data.image_size,
        robust=config.robustness.enabled,
        clean_probability=config.robustness.clean_probability,
    )
    validation_transform = TrainingImageTransform(
        config.data.image_size,
        robust=False,
        clean_probability=1.0,
    )
    train_dataset = ImageManifestDataset(
        config.data.manifest_path,
        config.data.train_split,
        transform=train_transform,
    )
    validation_dataset = ImageManifestDataset(
        config.data.manifest_path,
        config.data.val_split,
        transform=validation_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    common: dict[str, Any] = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.data.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader


def _write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary_path.replace(output_path)


def _sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of ``path`` without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initialize_model_from_checkpoint(
    model: torch.nn.Module,
    config: ExperimentConfig,
    *,
    device: torch.device,
) -> str | None:
    """Verify and load an optional warm-start checkpoint.

    Metadata is inspected before any weights are applied, and this function is
    called before optimizer construction so optimizer state always references
    the initialized model parameters.
    """

    initialization = config.initialization
    if initialization is None:
        return None

    checkpoint_path = Path(initialization.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"initialization checkpoint not found: {checkpoint_path}"
        )
    actual_sha256 = _sha256_file(checkpoint_path)
    if not hmac.compare_digest(
        actual_sha256, initialization.expected_sha256.lower()
    ):
        raise ValueError(
            "initialization checkpoint SHA-256 mismatch: "
            f"expected {initialization.expected_sha256.lower()}, "
            f"got {actual_sha256}"
        )

    from safetensors import safe_open

    with safe_open(
        str(checkpoint_path), framework="pt", device="cpu"
    ) as checkpoint:
        checkpoint_metadata = dict(checkpoint.metadata() or {})

    architecture = checkpoint_metadata.get("architecture")
    if architecture != "efficientnet_b0_binary":
        raise ValueError(
            "initialization checkpoint architecture must be "
            f"'efficientnet_b0_binary', got {architecture!r}"
        )
    resolve_checkpoint_image_size(
        checkpoint_metadata,
        requested_image_size=config.data.image_size,
    )

    from src.models.checkpoints import load_checkpoint

    load_checkpoint(model, checkpoint_path, device=str(device), strict=True)
    return actual_sha256


def _build_checkpoint_metadata(
    config: ExperimentConfig,
    early_stopping: EarlyStopping,
    *,
    parent_checkpoint_sha256: str | None = None,
) -> dict[str, str]:
    """Return the weight-file metadata required for safe inference."""

    metadata = {
        "architecture": "efficientnet_b0_binary",
        "best_epoch": str(early_stopping.best_epoch),
        "best_validation_loss": f"{early_stopping.best_loss:.10g}",
        "image_size": str(config.data.image_size),
        PREPROCESSING_METADATA_KEY: PREPROCESSING_CONTRACT_ID,
        "robust_training": str(config.robustness.enabled).lower(),
        "seed": str(config.seed),
    }
    if parent_checkpoint_sha256 is not None:
        metadata["parent_checkpoint_sha256"] = parent_checkpoint_sha256
    return metadata


def train_from_config(
    config: ExperimentConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train one experiment and write its best checkpoint and metadata."""

    from src.models.checkpoints import save_checkpoint
    from src.models.efficientnet import build_model, count_parameters

    set_global_seed(config.seed)
    device = resolve_device(device_name)
    train_loader, validation_loader = create_dataloaders(config)
    model = build_model(
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone,
        unfreeze_last_blocks=config.model.unfreeze_last_blocks,
    ).to(device)
    parent_checkpoint_sha256 = _initialize_model_from_checkpoint(
        model,
        config,
        device=device,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = make_grad_scaler(config.training.mixed_precision)
    early_stopping = EarlyStopping(config.training.patience)
    history: list[dict[str, Any]] = []
    started_at = time.time()
    freeze_frozen_batchnorm = bool(
        config.initialization is not None
        and config.initialization.freeze_frozen_batchnorm
    )

    for epoch in range(1, config.training.epochs + 1):
        train_result = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            mixed_precision=config.training.mixed_precision,
            scaler=scaler,
            freeze_frozen_batchnorm=freeze_frozen_batchnorm,
        )
        validation_result = run_epoch(
            model,
            validation_loader,
            device=device,
            mixed_precision=config.training.mixed_precision,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "validation_loss": validation_result.loss,
            "train_metrics": train_result.metrics,
            "validation_metrics": validation_result.metrics,
        }
        history.append(epoch_record)
        _write_json(config.output.history_path, history)
        if early_stopping.update(validation_result.loss, model, epoch):
            break

    early_stopping.restore(model)
    checkpoint_metadata = _build_checkpoint_metadata(
        config,
        early_stopping,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    checkpoint_path = Path(config.output.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, checkpoint_path, metadata=checkpoint_metadata)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        PREPROCESSING_METADATA_KEY: PREPROCESSING_CONTRACT_ID,
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": early_stopping.best_epoch,
        "best_validation_loss": early_stopping.best_loss,
        "elapsed_seconds": time.time() - started_at,
        "parameter_count": count_parameters(model),
        "trainable_parameter_count": count_parameters(model, trainable_only=True),
        "config": config.to_dict(),
    }
    if parent_checkpoint_sha256 is not None:
        metadata["parent_checkpoint_sha256"] = parent_checkpoint_sha256
    _write_json(config.output.metadata_path, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the EfficientNet-B0 AIGC detector from a YAML/JSON config."
    )
    parser.add_argument("--config", required=True, help="Path to experiment config")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (default: auto, preferring CUDA)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    metadata = train_from_config(config, device_name=args.device)
    print(
        f"Saved best checkpoint from epoch {metadata['best_epoch']} "
        f"to {metadata['checkpoint_path']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
