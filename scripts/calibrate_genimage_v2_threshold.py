"""Lock a v2 threshold on validation predictions and audit held-out tests once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.calibration import (  # noqa: E402
    ThresholdPolicy,
    score_scenarios_at_locked_threshold,
    select_sid_fpr_threshold,
    wilson_interval,
)


SID_VALIDATION_COUNTS = {0: 300, 1: 300}
GENIMAGE_VALIDATION_COUNTS = {0: 560, 1: 560}
SID_TEST_COUNTS = {0: 300, 1: 300}
GENIMAGE_TEST_COUNTS = {0: 560, 1: 560}
EXPECTED_SCENARIOS = 20
PROMOTION_GATES = {
    "max_sid_test_false_positive_rate": 0.05,
    "min_sid_test_balanced_accuracy": 0.93,
    "min_sid_test_recall": 0.90,
    "min_genimage_test_balanced_accuracy": 0.73,
    "min_sid_mean_transformed_balanced_accuracy": 0.90,
    "min_sid_worst_transformed_balanced_accuracy": 0.85,
    "min_genimage_mean_transformed_balanced_accuracy": 0.73,
    "min_genimage_worst_transformed_balanced_accuracy": 0.65,
}


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Prediction CSV is missing or empty: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "label", "transform", "severity", "pred"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Prediction CSV schema is incomplete: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Prediction CSV contains no rows: {path}")
    return rows


def read_clean_validation_predictions(
    path: Path, *, name: str, expected_counts: Mapping[int, int]
) -> tuple[list[int], list[float], dict[str, Any]]:
    rows = _read_prediction_rows(path)
    labels: list[int] = []
    scores: list[float] = []
    paths: set[str] = set()
    for number, row in enumerate(rows, start=2):
        if row["transform"].strip().casefold() != "clean" or row["severity"].strip():
            raise RuntimeError(f"{name} row {number} is not a clean-only prediction")
        image_path = row["image_path"].strip()
        if not image_path or image_path in paths:
            raise RuntimeError(f"{name} contains a missing or duplicate image path")
        paths.add(image_path)
        try:
            label = int(row["label"])
            score = float(row["pred"])
        except ValueError as error:
            raise RuntimeError(f"{name} row {number} is not numeric") from error
        if label not in (0, 1) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(f"{name} row {number} has an invalid label or score")
        labels.append(label)
        scores.append(score)
    counts = Counter(labels)
    if dict(counts) != dict(expected_counts):
        raise RuntimeError(f"{name} class counts are {dict(counts)}, expected {dict(expected_counts)}")
    return labels, scores, {
        "sha256": file_sha256(path),
        "rows": len(rows),
        "class_counts": {str(key): value for key, value in sorted(counts.items())},
    }


def _validate_test_rows(
    path: Path, *, name: str, expected_counts: Mapping[int, int]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = _read_prediction_rows(path)
    scenario_counts: Counter[tuple[str, str, int]] = Counter()
    scenario_keys: set[tuple[str, str]] = set()
    identities: set[tuple[str, str, str]] = set()
    for number, row in enumerate(rows, start=2):
        transform = row["transform"].strip()
        severity = row["severity"].strip()
        image_path = row["image_path"].strip()
        try:
            label = int(row["label"])
            score = float(row["pred"])
        except ValueError as error:
            raise RuntimeError(f"{name} row {number} is not numeric") from error
        if (
            not transform
            or not image_path
            or label not in (0, 1)
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise RuntimeError(f"{name} row {number} is invalid")
        identity = (transform, severity, image_path)
        if identity in identities:
            raise RuntimeError(f"{name} contains duplicate scenario/image rows")
        identities.add(identity)
        scenario_keys.add((transform, severity))
        scenario_counts[(transform, severity, label)] += 1
    if len(scenario_keys) != EXPECTED_SCENARIOS:
        raise RuntimeError(
            f"{name} contains {len(scenario_keys)} scenarios, expected {EXPECTED_SCENARIOS}"
        )
    for transform, severity in scenario_keys:
        actual = {label: scenario_counts[(transform, severity, label)] for label in (0, 1)}
        if actual != dict(expected_counts):
            raise RuntimeError(
                f"{name} scenario {(transform, severity)} counts are {actual}, "
                f"expected {dict(expected_counts)}"
            )
    return rows, {
        "sha256": file_sha256(path),
        "rows": len(rows),
        "scenario_count": len(scenario_keys),
    }


def _summarize_locked_test(
    rows: Sequence[Mapping[str, Any]], *, threshold: float
) -> dict[str, Any]:
    scenarios = score_scenarios_at_locked_threshold(rows, threshold=threshold)
    clean = next(
        (
            scenario
            for scenario in scenarios
            if scenario["transform"].casefold() == "clean"
            and not scenario["severity"]
        ),
        None,
    )
    if clean is None:
        raise RuntimeError("Held-out predictions contain no clean scenario")
    transformed = [scenario for scenario in scenarios if scenario is not clean]
    transformed_ba = [scenario["metrics"]["balanced_accuracy"] for scenario in transformed]
    clean_metrics = dict(clean["metrics"])
    clean_metrics["false_positive_rate_wilson_95_descriptive"] = wilson_interval(
        clean_metrics["false_positives"], clean_metrics["negatives"]
    )
    return {
        "clean": clean_metrics,
        "mean_transformed_balanced_accuracy": sum(transformed_ba) / len(transformed_ba),
        "worst_transformed_balanced_accuracy": min(transformed_ba),
        "scenarios": scenarios,
    }


def _promotion_decision(sid: Mapping[str, Any], genimage: Mapping[str, Any]) -> dict[str, Any]:
    sid_clean = sid["clean"]
    genimage_clean = genimage["clean"]
    checks = {
        "sid_false_positive_rate": {
            "value": sid_clean["false_positive_rate"],
            "operator": "<=",
            "limit": PROMOTION_GATES["max_sid_test_false_positive_rate"],
            "passed": sid_clean["false_positive_rate"] <= PROMOTION_GATES["max_sid_test_false_positive_rate"],
        },
        "sid_balanced_accuracy": {
            "value": sid_clean["balanced_accuracy"],
            "operator": ">=",
            "limit": PROMOTION_GATES["min_sid_test_balanced_accuracy"],
            "passed": sid_clean["balanced_accuracy"] >= PROMOTION_GATES["min_sid_test_balanced_accuracy"],
        },
        "sid_recall": {
            "value": sid_clean["recall"],
            "operator": ">=",
            "limit": PROMOTION_GATES["min_sid_test_recall"],
            "passed": sid_clean["recall"] >= PROMOTION_GATES["min_sid_test_recall"],
        },
        "genimage_balanced_accuracy": {
            "value": genimage_clean["balanced_accuracy"],
            "operator": ">=",
            "limit": PROMOTION_GATES["min_genimage_test_balanced_accuracy"],
            "passed": genimage_clean["balanced_accuracy"] >= PROMOTION_GATES["min_genimage_test_balanced_accuracy"],
        },
        "sid_mean_transformed_balanced_accuracy": {
            "value": sid["mean_transformed_balanced_accuracy"],
            "operator": ">=",
            "limit": PROMOTION_GATES["min_sid_mean_transformed_balanced_accuracy"],
            "passed": sid["mean_transformed_balanced_accuracy"] >= PROMOTION_GATES["min_sid_mean_transformed_balanced_accuracy"],
        },
        "sid_worst_transformed_balanced_accuracy": {
            "value": sid["worst_transformed_balanced_accuracy"],
            "operator": ">=",
            "limit": PROMOTION_GATES["min_sid_worst_transformed_balanced_accuracy"],
            "passed": sid["worst_transformed_balanced_accuracy"] >= PROMOTION_GATES["min_sid_worst_transformed_balanced_accuracy"],
        },
        "genimage_mean_transformed_balanced_accuracy": {
            "value": genimage["mean_transformed_balanced_accuracy"],
            "operator": ">=",
            "limit": PROMOTION_GATES["min_genimage_mean_transformed_balanced_accuracy"],
            "passed": genimage["mean_transformed_balanced_accuracy"] >= PROMOTION_GATES["min_genimage_mean_transformed_balanced_accuracy"],
        },
        "genimage_worst_transformed_balanced_accuracy": {
            "value": genimage["worst_transformed_balanced_accuracy"],
            "operator": ">=",
            "limit": PROMOTION_GATES["min_genimage_worst_transformed_balanced_accuracy"],
            "passed": genimage["worst_transformed_balanced_accuracy"] >= PROMOTION_GATES["min_genimage_worst_transformed_balanced_accuracy"],
        },
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "passed": passed,
        "recommendation": "promote_v2" if passed else "retain_v1",
        "checks": checks,
        "evidence_status": "exploratory_rescore_of_previously_observed_test_predictions",
        "numeric_selector_consumed_test_predictions": False,
        "policy_was_motivated_by_prior_test_review": True,
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    selection = payload["selection"]
    lines = [
        "# GenImage v2 Validation-Only Threshold Calibration",
        "",
        "> The numeric threshold selector consumes clean SID and GenImage validation predictions only. The earlier 0.50 test review motivated this calibration, so the saved test re-score is exploratory rather than a fresh holdout. WildFake remains excluded.",
        "",
    ]
    if selection["status"] != "selected":
        lines.extend(
            [
                "## Decision",
                "",
                "No threshold satisfied the predeclared safeguards. Keep v1 deployed at 0.50.",
            ]
        )
    else:
        selected = selection["selected"]
        sid_val = selected["sid_validation"]
        gen_val = selected["genimage_validation"]
        lines.extend(
            [
                "## Locked operating point",
                "",
                f"- Threshold: `{selected['threshold']:.8f}`",
                f"- SID validation false positives: `{sid_val['false_positives']}/{sid_val['negatives']}` ({_percent(sid_val['false_positive_rate'])})",
                f"- SID validation generated recall: `{_percent(sid_val['recall'])}`",
                f"- GenImage validation balanced accuracy: `{_percent(gen_val['balanced_accuracy'])}`",
                f"- GenImage validation generated recall: `{_percent(gen_val['recall'])}`",
                "",
                "The selected point has the lowest SID validation false-positive count among all thresholds that met the recall and GenImage guardrails.",
                "The SID validation interval is descriptive only because the same validation rows were searched across many thresholds. GenImage validation was also used for training early stopping, so it is not an independent calibration set.",
                "",
            ]
        )
        test_rescore = payload.get("exploratory_test_rescore")
        if test_rescore:
            sid_test = test_rescore["sid"]["clean"]
            gen_test = test_rescore["genimage"]["clean"]
            promotion = payload["promotion"]
            lines.extend(
                [
                    "## Exploratory re-score of the previously observed tests",
                    "",
                    f"- SID test false positives: `{sid_test['false_positives']}/{sid_test['negatives']}` ({_percent(sid_test['false_positive_rate'])})",
                    f"- SID test balanced accuracy: `{_percent(sid_test['balanced_accuracy'])}`",
                    f"- SID test generated recall: `{_percent(sid_test['recall'])}`",
                    f"- GenImage test balanced accuracy: `{_percent(gen_test['balanced_accuracy'])}`",
                    f"- SID mean/worst transformed balanced accuracy: `{_percent(test_rescore['sid']['mean_transformed_balanced_accuracy'])}` / `{_percent(test_rescore['sid']['worst_transformed_balanced_accuracy'])}`",
                    f"- GenImage mean/worst transformed balanced accuracy: `{_percent(test_rescore['genimage']['mean_transformed_balanced_accuracy'])}` / `{_percent(test_rescore['genimage']['worst_transformed_balanced_accuracy'])}`",
                    f"- Exploratory deployment recommendation: **{promotion['recommendation'].replace('_', ' ')}**",
                    "",
                    "These tests had already been inspected at threshold 0.50 and therefore are not a fresh, unbiased holdout. They are an exploratory deployment check only. The threshold must not be changed after this re-score.",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_calibration(
    *,
    sid_validation_csv: Path,
    genimage_validation_csv: Path,
    output_json: Path,
    output_report: Path,
    sid_test_csv: Path | None = None,
    genimage_test_csv: Path | None = None,
    policy: ThresholdPolicy = ThresholdPolicy(),
) -> dict[str, Any]:
    sid_labels, sid_scores, sid_identity = read_clean_validation_predictions(
        sid_validation_csv,
        name="SID validation",
        expected_counts=SID_VALIDATION_COUNTS,
    )
    gen_labels, gen_scores, gen_identity = read_clean_validation_predictions(
        genimage_validation_csv,
        name="GenImage validation",
        expected_counts=GENIMAGE_VALIDATION_COUNTS,
    )
    selection = select_sid_fpr_threshold(
        sid_labels=sid_labels,
        sid_scores=sid_scores,
        genimage_labels=gen_labels,
        genimage_scores=gen_scores,
        policy=policy,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "validation-only numeric threshold selection followed by an exploratory re-score of previously observed tests",
        "selection": selection,
        "validation_inputs": {
            "sid": sid_identity,
            "genimage": gen_identity,
        },
        "wildfake_used": False,
        "numeric_selector_consumed_test_predictions": False,
        "policy_was_motivated_by_prior_test_review": True,
        "genimage_validation_was_previously_used_for_training_early_stopping": True,
    }
    supplied_tests = sid_test_csv is not None or genimage_test_csv is not None
    if supplied_tests and (sid_test_csv is None or genimage_test_csv is None):
        raise ValueError("supply both held-out test CSVs or neither")
    if selection["status"] == "selected" and sid_test_csv and genimage_test_csv:
        threshold = float(selection["selected"]["threshold"])
        sid_rows, sid_test_identity = _validate_test_rows(
            sid_test_csv, name="SID test", expected_counts=SID_TEST_COUNTS
        )
        gen_rows, gen_test_identity = _validate_test_rows(
            genimage_test_csv,
            name="GenImage test",
            expected_counts=GENIMAGE_TEST_COUNTS,
        )
        sid_test = _summarize_locked_test(sid_rows, threshold=threshold)
        gen_test = _summarize_locked_test(gen_rows, threshold=threshold)
        payload["exploratory_test_rescore"] = {
            "threshold": threshold,
            "sid": sid_test,
            "genimage": gen_test,
            "inputs": {"sid": sid_test_identity, "genimage": gen_test_identity},
        }
        payload["promotion"] = _promotion_decision(sid_test, gen_test)
    else:
        payload["promotion"] = {
            "passed": False,
            "recommendation": "retain_v1",
            "reason": (
                "no feasible validation threshold"
                if selection["status"] != "selected"
                else "held-out test predictions were not supplied"
            ),
        }
    _write_json(output_json, payload)
    write_report(output_report, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimize SID validation FPR under fixed recall and GenImage safeguards."
    )
    parser.add_argument("--sid-validation", type=Path, required=True)
    parser.add_argument("--genimage-validation", type=Path, required=True)
    parser.add_argument("--sid-test", type=Path)
    parser.add_argument("--genimage-test", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_calibration(
        sid_validation_csv=args.sid_validation,
        genimage_validation_csv=args.genimage_validation,
        sid_test_csv=args.sid_test,
        genimage_test_csv=args.genimage_test,
        output_json=args.output_json,
        output_report=args.output_report,
    )
    selection = result["selection"]
    if selection["status"] == "selected":
        print(
            "Locked validation-only threshold:",
            f"{selection['selected']['threshold']:.8f}",
        )
    else:
        print("No feasible threshold; v1 must remain deployed.")
    print("Promotion recommendation:", result["promotion"]["recommendation"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
