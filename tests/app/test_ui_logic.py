from __future__ import annotations

import math

import pytest

from app.ui_logic import (
    aggregate_transform_scores,
    build_stress_table,
    format_transform_name,
    interpret_probability,
    summarize_robustness,
    validate_probability,
)


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf])
def test_validate_probability_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError):
        validate_probability(value)


@pytest.mark.parametrize("value", [True, "0.8", None])
def test_validate_probability_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises(TypeError):
        validate_probability(value)


def test_interpret_probability_uses_an_inconclusive_band() -> None:
    result = interpret_probability(0.55)

    assert result.label == "Inconclusive"
    assert result.uncertainty == "High uncertainty"
    assert result.generated_side is None


@pytest.mark.parametrize(
    ("score", "label", "side"),
    [
        (0.2, "Likely authentic", False),
        (0.8, "Likely AI-generated", True),
    ],
)
def test_interpret_probability_labels_clearer_scores(
    score: float, label: str, side: bool
) -> None:
    result = interpret_probability(score)

    assert result.label == label
    assert result.generated_side is side
    assert "not proof" in result.explanation.lower()


def test_summarize_robustness_for_generated_side() -> None:
    results = [
        {"transform": "jpeg", "severity": "90", "pred": 0.7},
        {"transform": "center_crop", "severity": "0.8", "pred": 0.4},
        {"transform": "resize", "severity": "0.5", "pred": 0.9},
    ]

    summary = summarize_robustness(0.8, results)

    assert summary.case_count == 3
    assert summary.consistent_count == 2
    assert summary.family_count == 3
    assert summary.label_stability == pytest.approx(2 / 3)
    assert summary.score_consistency == pytest.approx(0.8)
    assert summary.average_probability == pytest.approx(2 / 3)
    assert summary.largest_shift == pytest.approx(0.4)
    assert summary.largest_shift_delta == pytest.approx(-0.4)
    assert summary.largest_shift_transform == "Center crop · retain 80%"
    assert summary.largest_shift_probability == pytest.approx(0.4)
    assert summary.largest_shift_flipped is True


def test_summarize_robustness_finds_largest_authentic_side_shift() -> None:
    results = [
        {"transform": "jpeg", "severity": "50", "pred": 0.1},
        {"transform": "gaussian_noise", "severity": "0.1", "pred": 0.6},
    ]

    summary = summarize_robustness(0.2, results)

    assert summary.largest_shift_transform == "Gaussian noise · σ=0.1"
    assert summary.largest_shift_probability == pytest.approx(0.6)
    assert summary.consistent_count == 1


def test_summary_balances_transform_families_not_raw_case_count() -> None:
    summary = summarize_robustness(
        0.8,
        [
            {"transform": "jpeg", "severity": "90", "pred": 0.6},
            {"transform": "jpeg", "severity": "30", "pred": 0.8},
            {"transform": "gaussian_blur", "severity": "2", "pred": 0.4},
        ],
    )

    assert summary.average_probability == pytest.approx(0.55)
    assert summary.mean_absolute_shift == pytest.approx(0.25)
    assert summary.score_consistency == pytest.approx(0.75)
    assert summary.label_stability == pytest.approx(0.5)


def test_summary_treats_exact_boundary_as_generated_side() -> None:
    summary = summarize_robustness(
        0.5,
        [{"transform": "jpeg", "severity": "90", "pred": 0.499}],
    )

    assert summary.label_stability == 0.0
    assert summary.largest_shift_flipped is True


def test_summary_preserves_first_result_when_largest_shift_ties() -> None:
    summary = summarize_robustness(
        0.5,
        [
            {"transform": "jpeg", "severity": "90", "pred": 0.25},
            {"transform": "resize", "severity": "0.5", "pred": 0.75},
        ],
    )

    assert summary.largest_shift_name == "jpeg"


@pytest.mark.parametrize(
    "results",
    [
        [
            {"transform": "jpeg", "severity": "90", "pred": 0.5},
            {"transform": "JPEG", "severity": "90", "pred": 0.6},
        ],
        [{"transform": "", "severity": "90", "pred": 0.5}],
        [{"transform": "jpeg", "severity": "", "pred": 0.5}],
        [{"transform": "clean", "severity": "original", "pred": 0.5}],
    ],
)
def test_summary_rejects_malformed_stress_descriptors(
    results: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError):
        summarize_robustness(0.5, results)


def test_summary_rejects_results_that_do_not_match_expected_grid() -> None:
    with pytest.raises(ValueError, match="configured transform grid"):
        summarize_robustness(
            0.5,
            [{"transform": "jpeg", "severity": "90", "pred": 0.5}],
            expected_pairs=[("jpeg", "90"), ("jpeg", "30")],
        )


@pytest.mark.parametrize(
    "results",
    [
        ["not-a-row"],
        [{"transform": "jpeg", "severity": "90"}],
        [{"transform": None, "severity": "90", "pred": 0.5}],
        [{"transform": "jpeg", "severity": 90, "pred": 0.5}],
        [{"transform": "jpeg", "severity": "90", "pred": "0.5"}],
        [{"transform": "jpeg", "severity": "90", "pred": math.nan}],
    ],
)
def test_summary_rejects_malformed_stress_rows(results: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        summarize_robustness(0.5, results)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_stress_table_rejects_degenerate_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        build_stress_table(
            0.5,
            [{"transform": "jpeg", "severity": "90", "pred": 0.5}],
            threshold=threshold,
        )


def test_summarize_robustness_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_robustness(0.5, [])


def test_chart_aggregation_preserves_family_order_and_averages() -> None:
    rows = aggregate_transform_scores(
        0.6,
        [
            {"transform": "jpeg", "severity": "90", "pred": 0.5},
            {"transform": "jpeg", "severity": "30", "pred": 0.3},
            {"transform": "resize", "severity": "0.5", "pred": 0.7},
        ],
    )

    assert rows == [
        {"Scenario": "Clean image", "AIGC score": 0.6},
        {"Scenario": "JPEG", "AIGC score": pytest.approx(0.4)},
        {"Scenario": "Resize", "AIGC score": 0.7},
    ]


def test_stress_table_has_percentage_point_shift_and_label_consistency() -> None:
    rows = build_stress_table(
        0.75,
        [{"transform": "jpeg", "severity": "30", "pred": 0.45}],
    )

    assert rows == [
        {
            "Transform": "JPEG",
            "Severity": "30",
            "AIGC score (%)": 45.0,
            "Shift (pp)": pytest.approx(-30.0),
            "Label flipped": "Yes",
        }
    ]


def test_transform_name_formatting_is_human_readable() -> None:
    assert format_transform_name("gaussian_blur", "2.0") == "Gaussian blur · σ=2.0"
    assert format_transform_name("center_crop", "0.8") == "Center crop · retain 80%"
    assert format_transform_name("jpeg") == "JPEG"
