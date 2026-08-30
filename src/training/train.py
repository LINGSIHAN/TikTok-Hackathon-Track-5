"""Config-driven clean or robustness-augmented training entrypoint."""

from __future__ import annotations

import argparse
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


def _build_checkpoint_metadata(
    config: ExperimentConfig,
    early_stopping: EarlyStopping,
) -> dict[str, str]:
    """Return the weight-file metadata required for safe inference."""

    return {
        "architecture": "efficientnet_b0_binary",
        "best_epoch": str(early_stopping.best_epoch),
        "best_validation_loss": f"{early_stopping.best_loss:.10g}",
        "image_size": str(config.data.image_size),
        PREPROCESSING_METADATA_KEY: PREPROCESSING_CONTRACT_ID,
        "robust_training": str(config.robustness.enabled).lower(),
        "seed": str(config.seed),
    }


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

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
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

    for epoch in range(1, config.training.epochs + 1):
        train_result = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            mixed_precision=config.training.mixed_precision,
            scaler=scaler,
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
    checkpoint_metadata = _build_checkpoint_metadata(config, early_stopping)
    checkpoint_path = Path(config.output.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, checkpoint_path, metadata=checkpoint_metadata)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint_path),
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
