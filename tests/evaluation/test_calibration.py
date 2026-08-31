from __future__ import annotations

import pytest

from src.evaluation.calibration import (
    ThresholdPolicy,
    candidate_thresholds,
    score_scenarios_at_locked_threshold,
    select_sid_fpr_threshold,
    wilson_interval,
)


def test_selection_minimizes_sid_false_positives_with_recall_guard() -> None:
    result = select_sid_fpr_threshold(
        sid_labels=[0, 0, 0, 0, 1, 1, 1, 1],
        sid_scores=[0.10, 0.20, 0.60, 0.70, 0.65, 0.75, 0.80, 0.90],
        genimage_labels=[0, 0, 0, 0, 1, 1, 1, 1],
        genimage_scores=[0.05, 0.10, 0.15, 0.20, 0.70, 0.80, 0.90, 0.95],
        policy=ThresholdPolicy(
            min_sid_recall=0.75,
            max_genimage_balanced_accuracy_drop=0.20,
            min_genimage_balanced_accuracy=0.73,
            min_genimage_recall=0.55,
        ),
    )

    assert result["status"] == "selected"
    selected = result["selected"]
    assert selected["sid_validation"]["false_positives"] == 0
    assert selected["sid_validation"]["recall"] == pytest.approx(0.75)
    assert selected["genimage_validation"]["balanced_accuracy"] == pytest.approx(0.875)


def test_high_threshold_cannot_win_by_predicting_everything_authentic() -> None:
    result = select_sid_fpr_threshold(
        sid_labels=[0, 0, 1, 1],
        sid_scores=[0.60, 0.70, 0.10, 0.20],
        genimage_labels=[0, 0, 1, 1],
        genimage_scores=[0.50, 0.60, 0.80, 0.90],
    )

    assert result["status"] == "no_feasible_threshold"
    assert result["selected"] is None


def test_genimage_recall_guard_rejects_specificity_only_solution() -> None:
    result = select_sid_fpr_threshold(
        sid_labels=[0, 0, 1, 1],
        sid_scores=[0.10, 0.20, 0.80, 0.90],
        genimage_labels=[0, 0, 1, 1],
        genimage_scores=[0.10, 0.20, 0.30, 0.90],
        policy=ThresholdPolicy(
            min_sid_recall=0.50,
            max_genimage_balanced_accuracy_drop=0.50,
            min_genimage_balanced_accuracy=0.70,
            min_genimage_recall=0.75,
        ),
    )

    assert result["status"] == "selected"
    assert result["selected"]["genimage_validation"]["recall"] >= 0.75


def test_candidate_thresholds_cover_partitions_and_are_strict_interior() -> None:
    thresholds = candidate_thresholds([0.2, 0.8], [0.4, 0.6])

    assert thresholds == sorted(set(thresholds))
    assert 0.5 in thresholds
    assert all(0.0 < threshold < 1.0 for threshold in thresholds)
    assert {sum(score >= threshold for score in [0.2, 0.4, 0.6, 0.8]) for threshold in thresholds} == {0, 1, 2, 3, 4}


def test_selection_is_permutation_invariant() -> None:
    kwargs = {
        "sid_labels": [0, 0, 1, 1],
        "sid_scores": [0.1, 0.4, 0.7, 0.9],
        "genimage_labels": [0, 0, 1, 1],
        "genimage_scores": [0.2, 0.3, 0.8, 0.95],
    }
    forward = select_sid_fpr_threshold(**kwargs)
    reverse = select_sid_fpr_threshold(
        sid_labels=list(reversed(kwargs["sid_labels"])),
        sid_scores=list(reversed(kwargs["sid_scores"])),
        genimage_labels=list(reversed(kwargs["genimage_labels"])),
        genimage_scores=list(reversed(kwargs["genimage_scores"])),
    )

    assert forward == reverse


def test_score_scenarios_uses_one_locked_threshold() -> None:
    rows = [
        {"transform": "clean", "severity": "", "label": label, "pred": pred}
        for label, pred in [(0, 0.2), (1, 0.8)]
    ] + [
        {"transform": "jpeg", "severity": "30", "label": label, "pred": pred}
        for label, pred in [(0, 0.7), (1, 0.6)]
    ]

    scenarios = score_scenarios_at_locked_threshold(rows, threshold=0.65)

    assert [scenario["transform"] for scenario in scenarios] == ["clean", "jpeg"]
    assert scenarios[0]["metrics"]["balanced_accuracy"] == pytest.approx(1.0)
    assert scenarios[1]["metrics"]["false_positive_rate"] == pytest.approx(1.0)
    assert scenarios[1]["metrics"]["false_negative_rate"] == pytest.approx(1.0)


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(15, 300)

    assert lower < 0.05 < upper


@pytest.mark.parametrize(
    ("labels", "scores"),
    [([0, 0], [0.1, 0.2]), ([0, 1], [0.1, float("nan")]), ([0, 2], [0.1, 0.2])],
)
def test_selection_rejects_invalid_validation_inputs(labels, scores) -> None:
    with pytest.raises(ValueError):
        select_sid_fpr_threshold(
            sid_labels=labels,
            sid_scores=scores,
            genimage_labels=[0, 1],
            genimage_scores=[0.1, 0.9],
        )
