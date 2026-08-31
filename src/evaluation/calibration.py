"""Validation-only decision-threshold selection for binary image detection.

The policy is deliberately asymmetric: it minimizes SID false positives while
guarding generated-image recall and cross-generator GenImage performance.  Test
predictions are not accepted by the selection API; they are scored only after a
threshold has been locked.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.evaluation.metrics import compute_binary_metrics


POLICY_VERSION = "sid-fpr-min-v1"


@dataclass(frozen=True)
class ThresholdPolicy:
    """Predeclared constraints for the validation-only search."""

    min_sid_recall: float = 0.95
    max_genimage_balanced_accuracy_drop: float = 0.02
    min_genimage_balanced_accuracy: float = 0.73
    min_genimage_recall: float = 0.55

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1]")


def _validated_arrays(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.size != y_score.size:
        raise ValueError(f"{name} labels and scores must be equal-length 1-D arrays")
    if y_true.size == 0:
        raise ValueError(f"{name} predictions cannot be empty")
    if not np.all(np.isin(y_true, (0, 1))):
        raise ValueError(f"{name} labels must contain only 0 and 1")
    if set(int(value) for value in np.unique(y_true)) != {0, 1}:
        raise ValueError(f"{name} predictions must contain both classes")
    if not np.all(np.isfinite(y_score)) or np.any((y_score < 0) | (y_score > 1)):
        raise ValueError(f"{name} scores must be finite probabilities in [0, 1]")
    return y_true.astype(np.int64, copy=False), y_score


def candidate_thresholds(*score_groups: Sequence[float] | np.ndarray) -> list[float]:
    """Enumerate every attainable strict-interior operating point.

    Midpoints avoid putting the locked threshold directly on an observed score.
    The explicit 0.50 candidate preserves the familiar default for deterministic
    tie-breaking and reporting.
    """

    if not score_groups:
        raise ValueError("at least one score group is required")
    arrays = [np.asarray(group, dtype=np.float64).reshape(-1) for group in score_groups]
    if any(array.size == 0 for array in arrays):
        raise ValueError("score groups cannot be empty")
    merged = np.concatenate(arrays)
    if not np.all(np.isfinite(merged)) or np.any((merged < 0) | (merged > 1)):
        raise ValueError("scores must be finite probabilities in [0, 1]")
    boundaries = np.unique(np.concatenate((np.array([0.0, 1.0]), merged)))
    midpoints = (boundaries[:-1] + boundaries[1:]) / 2.0
    candidates = np.unique(np.concatenate((midpoints, np.array([0.5]))))
    return [float(value) for value in candidates if 0.0 < value < 1.0]


def operating_point(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Return confusion counts and standard metrics at one threshold."""

    y_true, y_score = _validated_arrays(labels, scores, name="operating-point")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    predicted = (y_score >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (predicted == 1)))
    tn = int(np.sum((y_true == 0) & (predicted == 0)))
    fp = int(np.sum((y_true == 0) & (predicted == 1)))
    fn = int(np.sum((y_true == 1) & (predicted == 0)))
    metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)
    return {
        "threshold": float(threshold),
        "samples": int(y_true.size),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "recall": float(metrics["false_negative_rate"] * -1.0 + 1.0),
        "specificity": float(metrics["false_positive_rate"] * -1.0 + 1.0),
        **{name: float(value) for name, value in metrics.items()},
    }


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided Wilson interval for a binomial proportion."""

    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("successes and total must be integers")
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("require 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def select_sid_fpr_threshold(
    *,
    sid_labels: Sequence[int] | np.ndarray,
    sid_scores: Sequence[float] | np.ndarray,
    genimage_labels: Sequence[int] | np.ndarray,
    genimage_scores: Sequence[float] | np.ndarray,
    policy: ThresholdPolicy = ThresholdPolicy(),
) -> dict[str, Any]:
    """Lock the lowest-SID-FPR threshold that satisfies all guardrails.

    Only validation arrays belong here.  Test scoring is intentionally a
    separate function so threshold selection cannot accidentally depend on it.
    """

    sid_y, sid_p = _validated_arrays(sid_labels, sid_scores, name="SID validation")
    gen_y, gen_p = _validated_arrays(
        genimage_labels, genimage_scores, name="GenImage validation"
    )
    thresholds = candidate_thresholds(sid_p, gen_p)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        rows.append(
            {
                "threshold": threshold,
                "sid": operating_point(sid_y, sid_p, threshold=threshold),
                "genimage": operating_point(gen_y, gen_p, threshold=threshold),
            }
        )
    best_genimage_ba = max(
        row["genimage"]["balanced_accuracy"] for row in rows
    )
    genimage_floor = max(
        policy.min_genimage_balanced_accuracy,
        best_genimage_ba - policy.max_genimage_balanced_accuracy_drop,
    )
    epsilon = 1e-12
    feasible = [
        row
        for row in rows
        if row["sid"]["recall"] + epsilon >= policy.min_sid_recall
        and row["genimage"]["balanced_accuracy"] + epsilon >= genimage_floor
        and row["genimage"]["recall"] + epsilon >= policy.min_genimage_recall
    ]
    policy_payload = {
        "version": POLICY_VERSION,
        **asdict(policy),
        "derived_genimage_balanced_accuracy_floor": genimage_floor,
    }
    if not feasible:
        return {
            "status": "no_feasible_threshold",
            "policy": policy_payload,
            "candidate_count": len(rows),
            "feasible_candidate_count": 0,
            "best_genimage_validation_balanced_accuracy": best_genimage_ba,
            "selected": None,
        }

    def ranking(row: Mapping[str, Any]) -> tuple[float, ...]:
        sid = row["sid"]
        genimage = row["genimage"]
        return (
            float(sid["false_positive_rate"]),
            -min(
                float(sid["balanced_accuracy"]),
                float(genimage["balanced_accuracy"]),
            ),
            -float(genimage["balanced_accuracy"]),
            -float(genimage["recall"]),
            -float(sid["recall"]),
            -float(row["threshold"]),
        )

    selected = min(feasible, key=ranking)
    sid_selected = dict(selected["sid"])
    sid_selected["false_positive_rate_wilson_95_descriptive"] = wilson_interval(
        sid_selected["false_positives"], sid_selected["negatives"]
    )
    selected_payload = {
        "threshold": selected["threshold"],
        "sid_validation": sid_selected,
        "genimage_validation": selected["genimage"],
    }
    return {
        "status": "selected",
        "policy": policy_payload,
        "candidate_count": len(rows),
        "feasible_candidate_count": len(feasible),
        "best_genimage_validation_balanced_accuracy": best_genimage_ba,
        "selected": selected_payload,
    }


def score_scenarios_at_locked_threshold(
    rows: Sequence[Mapping[str, Any]], *, threshold: float
) -> list[dict[str, Any]]:
    """Recompute scenario metrics after selection, without changing threshold."""

    if not rows:
        raise ValueError("test prediction rows cannot be empty")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        transform = str(row.get("transform", "")).strip()
        severity = str(row.get("severity", "")).strip()
        if not transform:
            raise ValueError("every prediction row requires a transform")
        key = (transform, severity)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    results: list[dict[str, Any]] = []
    for transform, severity in order:
        group = groups[(transform, severity)]
        try:
            labels = [int(row["label"]) for row in group]
            scores = [float(row["pred"]) for row in group]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid test prediction row") from error
        results.append(
            {
                "transform": transform,
                "severity": severity,
                "metrics": operating_point(labels, scores, threshold=threshold),
            }
        )
    return results
