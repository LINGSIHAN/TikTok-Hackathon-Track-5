"""Dependency-light binary classification metrics.

The implementations intentionally avoid scikit-learn so the required CLI has a
smaller deployment footprint. ROC-AUC uses the rank statistic with average ranks
for ties; average precision integrates precision at each distinct score.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


METRIC_NAMES = (
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "brier_score",
)


def _validate_inputs(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("labels and probabilities must have the same length")
    if y_true.size == 0:
        raise ValueError("labels and probabilities must not be empty")
    if not np.all(np.isin(y_true, (0, 1))):
        raise ValueError("labels must contain only 0 and 1")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("probabilities must be finite")
    if np.any((y_score < 0.0) | (y_score > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return y_true.astype(np.int64, copy=False), y_score


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(y_true.sum())
    negatives = int(y_true.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(y_score)
    positive_rank_sum = float(ranks[y_true == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]
    true_positives = np.cumsum(sorted_labels, dtype=np.float64)
    false_positives = np.cumsum(1 - sorted_labels, dtype=np.float64)

    distinct_ends = np.r_[
        np.flatnonzero(np.diff(sorted_scores)),
        sorted_scores.size - 1,
    ]
    precision = true_positives[distinct_ends] / (
        true_positives[distinct_ends] + false_positives[distinct_ends]
    )
    recall = true_positives[distinct_ends] / positives
    recall_increments = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increments * precision))


def compute_binary_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all required binary metrics at a fixed decision threshold.

    ROC-AUC is undefined when only one class is present. Average precision is
    undefined when there are no positive labels. Undefined metrics are returned
    as ``nan``; artifact writers convert them to JSON ``null``.
    """

    y_true, y_score = _validate_inputs(labels, probabilities, threshold)
    y_pred = (y_score >= threshold).astype(np.int64)

    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive

    true_positive_rate = (
        true_positive / positive_count if positive_count else float("nan")
    )
    true_negative_rate = (
        true_negative / negative_count if negative_count else float("nan")
    )
    balanced_accuracy = (
        (true_positive_rate + true_negative_rate) / 2.0
        if np.isfinite(true_positive_rate) and np.isfinite(true_negative_rate)
        else float("nan")
    )
    f1_denominator = 2 * true_positive + false_positive + false_negative

    return {
        "roc_auc": float(_roc_auc(y_true, y_score)),
        "average_precision": float(_average_precision(y_true, y_score)),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": 2 * true_positive / f1_denominator if f1_denominator else 0.0,
        "false_positive_rate": (
            false_positive / negative_count if negative_count else float("nan")
        ),
        "false_negative_rate": (
            false_negative / positive_count if positive_count else float("nan")
        ),
        "brier_score": float(np.mean((y_score - y_true) ** 2)),
    }
