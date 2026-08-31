"""Strict experiment configuration parsing and validation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or invalid."""


@dataclass(frozen=True)
class DataConfig:
    manifest_path: str
    train_split: str
    val_split: str
    test_split: str
    image_size: int
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class ModelConfig:
    pretrained: bool
    freeze_backbone: bool
    unfreeze_last_blocks: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    mixed_precision: bool


@dataclass(frozen=True)
class RobustnessConfig:
    enabled: bool
    clean_probability: float


@dataclass(frozen=True)
class OutputConfig:
    checkpoint_path: str
    metadata_path: str
    history_path: str


@dataclass(frozen=True)
class InitializationConfig:
    """Optional frozen-checkpoint warm-start configuration."""

    checkpoint_path: str
    expected_sha256: str
    freeze_frozen_batchnorm: bool


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    robustness: RobustnessConfig
    output: OutputConfig
    initialization: InitializationConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this configuration."""

        payload = asdict(self)
        if self.initialization is None:
            payload.pop("initialization")
        return payload


_T = TypeVar("_T")


def _section(
    raw: Mapping[str, Any],
    name: str,
    cls: type[_T],
) -> _T:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping")

    field_names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(value) - field_names
    missing = field_names - set(value)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{name}': {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ConfigError(
            f"Missing key(s) in '{name}': {', '.join(sorted(missing))}"
        )
    try:
        return cls(**dict(value))
    except TypeError as exc:
        raise ConfigError(f"Invalid '{name}' configuration: {exc}") from exc


def config_from_mapping(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Construct and validate an :class:`ExperimentConfig` from a mapping."""

    required = {"seed", "data", "model", "training", "robustness", "output"}
    expected = required | {"initialization"}
    unknown = set(raw) - expected
    missing = required - set(raw)
    if unknown:
        raise ConfigError(f"Unknown top-level key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"Missing top-level key(s): {', '.join(sorted(missing))}")

    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("'seed' must be a non-negative integer")

    initialization = (
        _section(raw, "initialization", InitializationConfig)
        if "initialization" in raw
        else None
    )
    config = ExperimentConfig(
        seed=seed,
        data=_section(raw, "data", DataConfig),
        model=_section(raw, "model", ModelConfig),
        training=_section(raw, "training", TrainingConfig),
        robustness=_section(raw, "robustness", RobustnessConfig),
        output=_section(raw, "output", OutputConfig),
        initialization=initialization,
    )
    _validate(config)
    return config


def _validate(config: ExperimentConfig) -> None:
    data = config.data
    training = config.training
    robustness = config.robustness
    model = config.model

    if not data.manifest_path.strip():
        raise ConfigError("data.manifest_path must not be empty")
    if any(not split.strip() for split in (data.train_split, data.val_split, data.test_split)):
        raise ConfigError("data split names must not be empty")
    if data.image_size <= 0:
        raise ConfigError("data.image_size must be greater than zero")
    if data.batch_size <= 0:
        raise ConfigError("data.batch_size must be greater than zero")
    if data.num_workers < 0:
        raise ConfigError("data.num_workers must be non-negative")

    if model.unfreeze_last_blocks < 0:
        raise ConfigError("model.unfreeze_last_blocks must be non-negative")
    if training.epochs <= 0:
        raise ConfigError("training.epochs must be greater than zero")
    if training.learning_rate <= 0:
        raise ConfigError("training.learning_rate must be greater than zero")
    if training.weight_decay < 0:
        raise ConfigError("training.weight_decay must be non-negative")
    if training.patience < 0:
        raise ConfigError("training.patience must be non-negative")
    if not 0.0 <= robustness.clean_probability <= 1.0:
        raise ConfigError("robustness.clean_probability must be in [0, 1]")

    for name, value in (
        ("output.checkpoint_path", config.output.checkpoint_path),
        ("output.metadata_path", config.output.metadata_path),
        ("output.history_path", config.output.history_path),
    ):
        if not value.strip():
            raise ConfigError(f"{name} must not be empty")
    if Path(config.output.checkpoint_path).suffix.lower() != ".safetensors":
        raise ConfigError("output.checkpoint_path must end in '.safetensors'")
    output_paths = {
        Path(config.output.checkpoint_path).expanduser().resolve(),
        Path(config.output.metadata_path).expanduser().resolve(),
        Path(config.output.history_path).expanduser().resolve(),
    }
    if len(output_paths) != 3:
        raise ConfigError("checkpoint, metadata, and history output paths must be distinct")

    initialization = config.initialization
    if initialization is None:
        return
    if not isinstance(initialization.checkpoint_path, str) or not (
        initialization.checkpoint_path.strip()
    ):
        raise ConfigError("initialization.checkpoint_path must not be empty")
    if Path(initialization.checkpoint_path).suffix.lower() != ".safetensors":
        raise ConfigError("initialization.checkpoint_path must end in '.safetensors'")
    if not isinstance(initialization.expected_sha256, str) or re.fullmatch(
        r"[0-9a-fA-F]{64}", initialization.expected_sha256
    ) is None:
        raise ConfigError(
            "initialization.expected_sha256 must be a 64-character hexadecimal SHA-256"
        )
    if not isinstance(initialization.freeze_frozen_batchnorm, bool):
        raise ConfigError("initialization.freeze_frozen_batchnorm must be a boolean")
    if model.pretrained:
        raise ConfigError(
            "model.pretrained must be false when initialization is configured"
        )

    initial_path = Path(initialization.checkpoint_path).expanduser().resolve()
    if initial_path in output_paths:
        raise ConfigError(
            "initialization.checkpoint_path and all output paths must be different"
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a JSON or YAML experiment configuration from ``path``."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            raw = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "PyYAML is required to read YAML configuration files"
                ) from exc
            raw = yaml.safe_load(handle)
        else:
            raise ConfigError("Configuration path must end in .json, .yaml, or .yml")

    if not isinstance(raw, Mapping):
        raise ConfigError("The top level of the configuration must be a mapping")
    return config_from_mapping(raw)
