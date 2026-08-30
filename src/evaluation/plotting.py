"""Compact robustness-summary visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _scenario_label(scenario: Mapping[str, Any]) -> str:
    name = str(scenario["transform"])
    severity = scenario.get("severity")
    return "clean" if name == "clean" else f"{name}:{severity}"


def save_robustness_plot(
    scenarios: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    metric: str = "roc_auc",
) -> None:
    """Save a clean-versus-transformed metric chart as a PNG."""

    if not scenarios:
        raise ValueError("At least one evaluation scenario is required")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to create robustness.png") from exc

    labels = [_scenario_label(scenario) for scenario in scenarios]
    values = [float(scenario["metrics"][metric]) for scenario in scenarios]
    finite_values = [value if np.isfinite(value) else 0.0 for value in values]
    colors = ["#2563eb"] + ["#94a3b8"] * (len(values) - 1)

    width = max(10.0, 0.55 * len(values))
    figure, axis = plt.subplots(figsize=(width, 5.2))
    bars = axis.bar(range(len(values)), finite_values, color=colors)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel(metric.replace("_", " ").upper())
    axis.set_title("AIGC detector robustness: clean vs transformed images")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        label = f"{value:.2f}" if np.isfinite(value) else "N/A"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(float(bar.get_height()) + 0.02, 0.98),
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    figure.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
