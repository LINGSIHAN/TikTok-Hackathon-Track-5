import math

import pytest

from src.evaluation.metrics import compute_binary_metrics


def test_binary_metrics_match_known_example():
    metrics = compute_binary_metrics(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.35, 0.8],
    )

    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)
    assert metrics["false_negative_rate"] == pytest.approx(0.5)
    assert metrics["brier_score"] == pytest.approx(0.158125)


def test_auc_and_average_precision_handle_tied_scores():
    metrics = compute_binary_metrics([0, 1], [0.5, 0.5])

    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx(0.5)


def test_single_class_metrics_are_explicitly_undefined():
    metrics = compute_binary_metrics([0, 0], [0.1, 0.2])

    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["average_precision"])
    assert math.isnan(metrics["balanced_accuracy"])
    assert math.isnan(metrics["false_negative_rate"])


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0], [], "same length"),
        ([2], [0.5], "only 0 and 1"),
        ([0], [1.2], r"in \[0, 1\]"),
    ],
)
def test_invalid_metric_inputs_are_rejected(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        compute_binary_metrics(labels, scores)
