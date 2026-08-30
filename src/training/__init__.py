"""Training utilities for the binary AIGC detector."""

from importlib import import_module
from typing import Any

from .config import ExperimentConfig, load_config

__all__ = [
    "EarlyStopping",
    "EpochResult",
    "ExperimentConfig",
    "load_config",
    "run_epoch",
    "set_global_seed",
]


def __getattr__(name: str) -> Any:
    """Load torch-dependent engine exports only when they are requested."""

    if name in {"EarlyStopping", "EpochResult", "run_epoch", "set_global_seed"}:
        engine = import_module(".engine", __name__)
        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
