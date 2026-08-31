from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import calibrate_genimage_v2_threshold as calibration


FIELDS = ["image_path", "label", "transform", "severity", "pred"]


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _clean_rows(prefix: str, negatives: list[float], positives: list[float]) -> list[dict[str, str]]:
    rows = []
    for label, scores in ((0, negatives), (1, positives)):
        for index, score in enumerate(scores):
            rows.append(
                {
                    "image_path": f"{prefix}-{label}-{index}.jpg",
                    "label": str(label),
                    "transform": "clean",
                    "severity": "",
                    "pred": str(score),
                }
            )
    return rows


def _test_rows(prefix: str, negatives: list[float], positives: list[float]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for transform, severity in (("clean", ""), ("jpeg", "30")):
        for row in _clean_rows(prefix, negatives, positives):
            row = dict(row)
            row["transform"] = transform
            row["severity"] = severity
            rows.append(row)
    return rows


def test_run_calibration_locks_validation_before_scoring_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibration, "SID_VALIDATION_COUNTS", {0: 2, 1: 2})
    monkeypatch.setattr(calibration, "GENIMAGE_VALIDATION_COUNTS", {0: 2, 1: 2})
    monkeypatch.setattr(calibration, "SID_TEST_COUNTS", {0: 2, 1: 2})
    monkeypatch.setattr(calibration, "GENIMAGE_TEST_COUNTS", {0: 2, 1: 2})
    monkeypatch.setattr(calibration, "EXPECTED_SCENARIOS", 2)
    sid_val = tmp_path / "sid-val.csv"
    gen_val = tmp_path / "gen-val.csv"
    sid_test = tmp_path / "sid-test.csv"
    gen_test = tmp_path / "gen-test.csv"
    _write_predictions(sid_val, _clean_rows("sid-val", [0.1, 0.2], [0.8, 0.9]))
    _write_predictions(gen_val, _clean_rows("gen-val", [0.1, 0.2], [0.8, 0.9]))
    _write_predictions(sid_test, _test_rows("sid-test", [0.1, 0.2], [0.8, 0.9]))
    _write_predictions(gen_test, _test_rows("gen-test", [0.1, 0.2], [0.8, 0.9]))
    output_json = tmp_path / "calibration.json"
    output_report = tmp_path / "calibration.md"

    result = calibration.run_calibration(
        sid_validation_csv=sid_val,
        genimage_validation_csv=gen_val,
        sid_test_csv=sid_test,
        genimage_test_csv=gen_test,
        output_json=output_json,
        output_report=output_report,
    )

    assert result["selection"]["status"] == "selected"
    assert result["selection"]["selected"]["sid_validation"]["false_positives"] == 0
    assert result["promotion"]["recommendation"] == "promote_v2"
    assert result["numeric_selector_consumed_test_predictions"] is False
    assert result["policy_was_motivated_by_prior_test_review"] is True
    assert json.loads(output_json.read_text(encoding="utf-8"))["promotion"]["passed"] is True
    assert "validation predictions only" in output_report.read_text(encoding="utf-8")


def test_changed_test_scores_do_not_change_locked_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibration, "SID_VALIDATION_COUNTS", {0: 2, 1: 2})
    monkeypatch.setattr(calibration, "GENIMAGE_VALIDATION_COUNTS", {0: 2, 1: 2})
    monkeypatch.setattr(calibration, "SID_TEST_COUNTS", {0: 2, 1: 2})
    monkeypatch.setattr(calibration, "GENIMAGE_TEST_COUNTS", {0: 2, 1: 2})
    monkeypatch.setattr(calibration, "EXPECTED_SCENARIOS", 2)
    sid_val = tmp_path / "sid-val.csv"
    gen_val = tmp_path / "gen-val.csv"
    _write_predictions(sid_val, _clean_rows("sid-val", [0.1, 0.2], [0.8, 0.9]))
    _write_predictions(gen_val, _clean_rows("gen-val", [0.1, 0.2], [0.8, 0.9]))

    thresholds = []
    for suffix, test_scores in (("good", ([0.1, 0.2], [0.8, 0.9])), ("bad", ([0.8, 0.9], [0.1, 0.2]))):
        sid_test = tmp_path / f"sid-{suffix}.csv"
        gen_test = tmp_path / f"gen-{suffix}.csv"
        _write_predictions(sid_test, _test_rows(f"sid-{suffix}", *test_scores))
        _write_predictions(gen_test, _test_rows(f"gen-{suffix}", *test_scores))
        result = calibration.run_calibration(
            sid_validation_csv=sid_val,
            genimage_validation_csv=gen_val,
            sid_test_csv=sid_test,
            genimage_test_csv=gen_test,
            output_json=tmp_path / f"{suffix}.json",
            output_report=tmp_path / f"{suffix}.md",
        )
        thresholds.append(result["selection"]["selected"]["threshold"])

    assert thresholds[0] == thresholds[1]


def test_validation_reader_rejects_transformed_or_duplicate_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibration, "SID_VALIDATION_COUNTS", {0: 1, 1: 1})
    path = tmp_path / "invalid.csv"
    rows = _clean_rows("sid", [0.1], [0.9])
    rows[1]["transform"] = "jpeg"
    _write_predictions(path, rows)

    with pytest.raises(RuntimeError, match="clean-only"):
        calibration.read_clean_validation_predictions(
            path, name="SID validation", expected_counts={0: 1, 1: 1}
        )


def test_no_feasible_threshold_fails_closed_without_reading_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibration, "SID_VALIDATION_COUNTS", {0: 2, 1: 2})
    monkeypatch.setattr(calibration, "GENIMAGE_VALIDATION_COUNTS", {0: 2, 1: 2})
    sid_val = tmp_path / "sid-val.csv"
    gen_val = tmp_path / "gen-val.csv"
    _write_predictions(sid_val, _clean_rows("sid", [0.6, 0.7], [0.1, 0.2]))
    _write_predictions(gen_val, _clean_rows("gen", [0.5, 0.6], [0.8, 0.9]))

    result = calibration.run_calibration(
        sid_validation_csv=sid_val,
        genimage_validation_csv=gen_val,
        sid_test_csv=tmp_path / "missing-sid-test.csv",
        genimage_test_csv=tmp_path / "missing-gen-test.csv",
        output_json=tmp_path / "result.json",
        output_report=tmp_path / "result.md",
    )

    assert result["selection"]["status"] == "no_feasible_threshold"
    assert result["promotion"]["recommendation"] == "retain_v1"


def test_deployment_review_rejects_transformed_robustness_collapse() -> None:
    clean = {
        "false_positive_rate": 0.0,
        "balanced_accuracy": 1.0,
        "recall": 1.0,
    }
    sid = {
        "clean": clean,
        "mean_transformed_balanced_accuracy": 0.89,
        "worst_transformed_balanced_accuracy": 0.84,
    }
    genimage = {
        "clean": clean,
        "mean_transformed_balanced_accuracy": 0.80,
        "worst_transformed_balanced_accuracy": 0.70,
    }

    decision = calibration._promotion_decision(sid, genimage)

    assert decision["recommendation"] == "retain_v1"
    assert decision["checks"]["sid_mean_transformed_balanced_accuracy"]["passed"] is False
    assert decision["checks"]["sid_worst_transformed_balanced_accuracy"]["passed"] is False
    assert decision["evidence_status"].startswith("exploratory_rescore")
