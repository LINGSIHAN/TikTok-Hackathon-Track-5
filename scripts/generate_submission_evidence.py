"""Generate compact, reproducible figures from validated evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def _scenario_key(transform: str, severity: object) -> tuple[str, str]:
    return transform, "" if severity in (None, "") else str(severity)


def _scenario_label(transform: str, severity: str) -> str:
    name = "JPEG" if transform == "jpeg" else transform.replace("_", " ").title()
    return "Clean" if transform == "clean" else f"{name} · {severity}"


def confusion_counts(rows: Sequence[dict[str, str]], threshold: float) -> dict[str, int]:
    """Return TN/FP/FN/TP counts for prediction rows at ``threshold``."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for row in rows:
        label = int(row["label"])
        probability = float(row["pred"])
        if label not in (0, 1):
            raise ValueError(f"unsupported label: {label}")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid probability: {probability}")
        predicted = int(probability >= threshold)
        if label == 1:
            key = "tp" if predicted == 1 else "fn"
        else:
            key = "fp" if predicted == 1 else "tn"
        counts[key] += 1
    if not rows:
        raise ValueError("prediction rows cannot be empty")
    return counts


def _load_scenarios(metrics_root: Path) -> tuple[float, list[dict[str, Any]]]:
    metrics_path = metrics_root / "metrics.json"
    predictions_path = metrics_root / "predictions.csv"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = float(payload["threshold"])
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[(row["transform"], row["severity"])].append(row)

    scenarios: list[dict[str, Any]] = []
    for scenario in payload["scenarios"]:
        key = _scenario_key(scenario["transform"], scenario.get("severity"))
        rows = groups.get(key, [])
        if len(rows) != int(scenario["num_samples"]):
            raise ValueError(f"prediction count does not match metrics for {key}")
        counts = confusion_counts(rows, threshold)
        expected_fp = round(float(scenario["metrics"]["false_positive_rate"]) * 300)
        expected_fn = round(float(scenario["metrics"]["false_negative_rate"]) * 300)
        if counts["fp"] != expected_fp or counts["fn"] != expected_fn:
            raise ValueError(f"confusion counts do not match metrics for {key}")
        scenarios.append({**scenario, "key": key, "rows": rows, "counts": counts})
    if len(scenarios) != 20 or len(groups) != 20:
        raise ValueError("the complete 20-scenario evaluation grid is required")
    return threshold, scenarios


def _configure_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to generate evidence figures") from error
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#1f2937",
            "axes.titlecolor": "#111827",
            "text.color": "#1f2937",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    return plt


def save_confusion_matrix(
    scenario: dict[str, Any], threshold: float, output_path: Path
) -> None:
    """Save the clean-test confusion matrix with exact counts and row shares."""

    import numpy as np

    plt = _configure_matplotlib()
    counts = scenario["counts"]
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    figure, axis = plt.subplots(figsize=(7.4, 6.2), facecolor="white")
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    axis.set_xticks([0, 1], labels=["Predicted authentic", "Predicted AIGC"])
    axis.set_yticks([0, 1], labels=["Actual authentic", "Actual AIGC"])
    axis.set_title("Clean test confusion matrix", fontsize=16, weight="bold", pad=34)
    axis.text(
        0.5,
        1.035,
        f"Robustness-trained model · n={int(matrix.sum())} · threshold={threshold:.2f}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#475569",
    )
    for row_index in range(2):
        row_total = int(matrix[row_index].sum())
        for column_index in range(2):
            value = int(matrix[row_index, column_index])
            color = "white" if value > matrix.max() / 2 else "#111827"
            axis.text(
                column_index,
                row_index,
                f"{value}\n{value / row_total:.1%} of class",
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                color=color,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Images")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_error_analysis(scenarios: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Save ranked false-positive and false-negative counts for all scenarios."""

    import numpy as np

    plt = _configure_matplotlib()
    ranked = sorted(
        scenarios,
        key=lambda item: (
            item["counts"]["fp"] + item["counts"]["fn"],
            item["counts"]["fn"],
        ),
        reverse=True,
    )
    labels = [_scenario_label(*item["key"]) for item in ranked]
    false_positives = [item["counts"]["fp"] for item in ranked]
    false_negatives = [item["counts"]["fn"] for item in ranked]
    positions = np.arange(len(ranked))
    height = 0.38
    figure, axis = plt.subplots(figsize=(12.8, 10.5), facecolor="white")
    fp_bars = axis.barh(
        positions - height / 2,
        false_positives,
        height,
        color="#2563eb",
        edgecolor="#1e3a8a",
        label="False positives (authentic → AIGC)",
    )
    fn_bars = axis.barh(
        positions + height / 2,
        false_negatives,
        height,
        color="#f59e0b",
        edgecolor="#92400e",
        hatch="//",
        label="False negatives (AIGC → authentic)",
    )
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Misclassified images (out of 300 per class)")
    axis.set_title(
        "False positives and false negatives by evaluation scenario",
        fontsize=16,
        weight="bold",
        pad=34,
    )
    axis.text(
        0.5,
        1.018,
        "Robustness-trained model · 600 images per scenario · fixed 0.50 threshold",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#475569",
    )
    axis.grid(axis="x", color="#cbd5e1", linewidth=0.8, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False)
    maximum = max(false_positives + false_negatives)
    axis.set_xlim(0, maximum + max(5, int(maximum * 0.13)))
    axis.bar_label(fp_bars, padding=3, fontsize=8, color="#1e3a8a")
    axis.bar_label(fn_bars, padding=3, fontsize=8, color="#92400e")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate confusion-matrix and error-analysis submission figures."
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("artifacts/metrics"),
        help="Directory containing metrics.json and predictions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/figures"),
        help="Directory receiving the two PNG figures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    threshold, scenarios = _load_scenarios(args.metrics_root)
    clean = next(item for item in scenarios if item["key"] == ("clean", ""))
    confusion_path = args.output_dir / "confusion_matrix.png"
    error_path = args.output_dir / "error_analysis.png"
    save_confusion_matrix(clean, threshold, confusion_path)
    save_error_analysis(scenarios, error_path)
    print(f"Generated {confusion_path}")
    print(f"Generated {error_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
