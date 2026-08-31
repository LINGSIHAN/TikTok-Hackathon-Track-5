"""Compare the frozen v1 and warm-started v2 models on held-out test data.

This script deliberately delegates image inference to the project's existing
evaluator.  Its responsibility is the stricter Kaggle v2 contract: validate the
two locked test manifests and checkpoint lineage, require the complete published
20-scenario grid at threshold 0.50, retain detailed artifacts locally, and emit
only aggregate, path-free public evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT_TEXT = str(REPOSITORY_ROOT)
if REPOSITORY_ROOT_TEXT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT_TEXT)

from src.evaluation.evaluate import build_scenarios, evaluate_checkpoint, summarize_scenarios
from src.transforms.robustness import TRANSFORM_GRID


FROZEN_V1_SHA256 = (
    "806fbabc5ecae8394369d08738cbf0c993568137d323a8133167e4557d04eed2"
)
FIXED_THRESHOLD = 0.50
FIXED_SEED = 42
IMAGE_SIZE = 224
EXPECTED_SCENARIO_COUNT = 20
GENIMAGE_DATASET_ID = "cartografia/unbiased-tiny-genimage"
SID_DATASET_ID = "saberzl/SID_Set"
GENIMAGE_EXPECTED_COUNTS = {0: 560, 1: 560}
SID_EXPECTED_COUNTS = {0: 300, 1: 300}
PREPROCESSING_CONTRACT = (
    "pil-exif-white-alpha-short-edge-bilinear-center-crop-imagenet-v1"
)
METRIC_KEYS = (
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "brier_score",
)
METRIC_LABELS = (
    ("ROC-AUC", "roc_auc"),
    ("Average precision", "average_precision"),
    ("Balanced accuracy", "balanced_accuracy"),
    ("F1", "f1"),
    ("False-positive rate", "false_positive_rate"),
    ("False-negative rate", "false_negative_rate"),
    ("Brier score", "brier_score"),
)
EXPECTED_SCENARIO_KEYS = tuple(
    (scenario.transform, "" if scenario.severity is None else str(scenario.severity))
    for scenario in build_scenarios(TRANSFORM_GRID)
)
if len(EXPECTED_SCENARIO_KEYS) != EXPECTED_SCENARIO_COUNT:  # pragma: no cover
    raise RuntimeError("Published transform grid no longer contains exactly 20 scenarios")


@dataclass(frozen=True)
class ManifestAudit:
    """Aggregate identity for one validated held-out manifest."""

    name: str
    dataset_id: str
    total_count: int
    class_counts: dict[int, int]
    manifest_sha256: str
    test_content_digest: str
    content_hashes: frozenset[str]


@dataclass(frozen=True)
class ModelIdentity:
    """Path-free checkpoint identity retained in public evidence."""

    v1_sha256: str
    v2_sha256: str
    parent_sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _reject_wildfake_input(path: Path, *, description: str) -> None:
    if "wildfake" in path.as_posix().casefold():
        raise RuntimeError(
            f"{description} must not reference WildFake; this workflow evaluates "
            "only the GenImage and SID held-out tests"
        )


def _validate_relative_image_path(value: str, *, manifest: Path) -> None:
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not value.strip()
        or pure.is_absolute()
        or ".." in pure.parts
        or re.match(r"^[A-Za-z]:/", normalized)
    ):
        raise RuntimeError(f"Unsafe image path in {manifest.name}: {value!r}")


def validate_test_manifest(
    manifest_path: Path,
    *,
    name: str,
    expected_dataset_id: str,
    expected_counts: Mapping[int, int],
) -> ManifestAudit:
    """Validate exact balanced test counts and return a path-free identity."""

    _reject_wildfake_input(manifest_path, description=f"{name} manifest")
    if not manifest_path.is_file():
        raise RuntimeError(f"{name} manifest not found: {manifest_path}")
    required = {"path", "label", "split", "dataset", "source_id", "sha256"}
    counts = {0: 0, 1: 0}
    seen_paths: set[str] = set()
    seen_source_ids: set[str] = set()
    seen_hashes: set[str] = set()
    canonical_rows: list[str] = []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = required - fieldnames
        if missing:
            raise RuntimeError(
                f"{name} manifest is missing column(s): {', '.join(sorted(missing))}"
            )
        for row_number, row in enumerate(reader, start=2):
            if any("wildfake" in str(value).casefold() for value in row.values()):
                raise RuntimeError(
                    f"{name} manifest row {row_number} references WildFake"
                )
            if row["split"] != "test":
                continue
            if row["dataset"] != expected_dataset_id:
                raise RuntimeError(
                    f"{name} test row {row_number} has dataset {row['dataset']!r}; "
                    f"expected {expected_dataset_id!r}"
                )
            if row["label"] not in {"0", "1"}:
                raise RuntimeError(
                    f"{name} test row {row_number} has unsupported label {row['label']!r}"
                )
            label = int(row["label"])
            image_path = row["path"]
            source_id = row["source_id"]
            content_hash = row["sha256"].lower()
            _validate_relative_image_path(image_path, manifest=manifest_path)
            if not source_id.strip():
                raise RuntimeError(f"{name} test row {row_number} has no source_id")
            if not _valid_sha256(content_hash):
                raise RuntimeError(f"{name} test row {row_number} has invalid sha256")
            if image_path in seen_paths:
                raise RuntimeError(f"Duplicate test path in {name} manifest: {image_path}")
            if source_id in seen_source_ids:
                raise RuntimeError(
                    f"Duplicate test source_id in {name} manifest: {source_id}"
                )
            if content_hash in seen_hashes:
                raise RuntimeError(
                    f"Duplicate test image content in {name} manifest: {content_hash}"
                )
            seen_paths.add(image_path)
            seen_source_ids.add(source_id)
            seen_hashes.add(content_hash)
            counts[label] += 1
            canonical_rows.append(
                f"{content_hash}\0{label}\0{expected_dataset_id}\0{source_id}\n"
            )

    expected = {int(label): int(count) for label, count in expected_counts.items()}
    if counts != expected:
        raise RuntimeError(
            f"{name} test counts are {counts}; expected exactly {expected}"
        )
    total = sum(counts.values())
    digest = hashlib.sha256()
    for row in sorted(canonical_rows):
        digest.update(row.encode("utf-8"))
    return ManifestAudit(
        name=name,
        dataset_id=expected_dataset_id,
        total_count=total,
        class_counts=counts,
        manifest_sha256=sha256_file(manifest_path),
        test_content_digest=digest.hexdigest(),
        content_hashes=frozenset(seen_hashes),
    )


def ensure_disjoint_test_content(first: ManifestAudit, second: ManifestAudit) -> None:
    overlap = first.content_hashes & second.content_hashes
    if overlap:
        raise RuntimeError(
            f"{first.name} and {second.name} test manifests share {len(overlap)} "
            "content hash(es)"
        )


def _read_checkpoint_metadata(path: Path) -> dict[str, str]:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - dependency preflight
        raise RuntimeError("safetensors is required to verify v2 lineage") from error
    try:
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            return dict(checkpoint.metadata() or {})
    except Exception as error:
        raise RuntimeError(f"Unable to read v2 checkpoint metadata: {path}") from error


def verify_checkpoint_lineage(
    v1_checkpoint: Path,
    v2_checkpoint: Path,
    v2_metadata_path: Path,
    *,
    expected_v1_sha256: str,
) -> ModelIdentity:
    """Verify frozen v1 bytes and v2's two independent lineage records."""

    for path, description in (
        (v1_checkpoint, "v1 checkpoint"),
        (v2_checkpoint, "v2 checkpoint"),
        (v2_metadata_path, "v2 training metadata"),
    ):
        _reject_wildfake_input(path, description=description)
        if not path.is_file():
            raise RuntimeError(f"{description} not found: {path}")
    if v1_checkpoint.resolve() == v2_checkpoint.resolve():
        raise RuntimeError("v1 and v2 checkpoint paths must be distinct")
    if not _valid_sha256(expected_v1_sha256):
        raise ValueError("expected_v1_sha256 must be a lowercase SHA-256 digest")

    actual_v1 = sha256_file(v1_checkpoint)
    if actual_v1 != expected_v1_sha256:
        raise RuntimeError(
            f"Frozen v1 checkpoint SHA-256 mismatch: expected {expected_v1_sha256}, "
            f"got {actual_v1}"
        )
    actual_v2 = sha256_file(v2_checkpoint)
    try:
        training_metadata = json.loads(v2_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("v2 training metadata is not valid JSON") from error
    if not isinstance(training_metadata, Mapping):
        raise RuntimeError("v2 training metadata must be a JSON object")
    parent = training_metadata.get("parent_checkpoint_sha256")
    if parent != expected_v1_sha256:
        raise RuntimeError(
            "v2 training metadata does not identify the frozen v1 checkpoint as parent"
        )
    recorded_v2 = training_metadata.get("checkpoint_sha256")
    if recorded_v2 is not None and recorded_v2 != actual_v2:
        raise RuntimeError("v2 training metadata checkpoint_sha256 does not match v2")

    checkpoint_metadata = _read_checkpoint_metadata(v2_checkpoint)
    if checkpoint_metadata.get("parent_checkpoint_sha256") != expected_v1_sha256:
        raise RuntimeError(
            "v2 safetensors metadata does not identify the frozen v1 checkpoint as parent"
        )
    expected_metadata = {
        "architecture": "efficientnet_b0_binary",
        "image_size": str(IMAGE_SIZE),
        "preprocessing_contract": PREPROCESSING_CONTRACT,
    }
    for key, expected_value in expected_metadata.items():
        if checkpoint_metadata.get(key) != expected_value:
            raise RuntimeError(
                f"v2 safetensors metadata {key!r} is incompatible: expected "
                f"{expected_value!r}, got {checkpoint_metadata.get(key)!r}"
            )
    return ModelIdentity(
        v1_sha256=actual_v1,
        v2_sha256=actual_v2,
        parent_sha256=str(parent),
    )


def _scenario_key(transform: object, severity: object) -> tuple[str, str]:
    return str(transform), "" if severity in (None, "") else str(severity)


def validate_evaluation_artifact(
    evaluation: Mapping[str, Any], *, expected_samples: int
) -> list[dict[str, Any]]:
    """Require a complete, finite, fixed-threshold evaluation artifact."""

    if evaluation.get("scenario_mode") != "full":
        raise RuntimeError("Evaluation artifact must use full scenario mode")
    if float(evaluation.get("threshold", -1.0)) != FIXED_THRESHOLD:
        raise RuntimeError("Evaluation artifact threshold must remain fixed at 0.50")
    scenarios_value = evaluation.get("scenarios")
    if not isinstance(scenarios_value, list):
        raise RuntimeError("Evaluation artifact has no scenario list")
    scenarios = scenarios_value
    if len(scenarios) != EXPECTED_SCENARIO_COUNT:
        raise RuntimeError(
            f"Evaluation contains {len(scenarios)} scenarios; expected exactly "
            f"{EXPECTED_SCENARIO_COUNT}"
        )
    if not all(isinstance(scenario, Mapping) for scenario in scenarios):
        raise RuntimeError("Every evaluation scenario must be a mapping")
    actual_keys = tuple(
        _scenario_key(scenario.get("transform"), scenario.get("severity"))
        for scenario in scenarios
    )
    if actual_keys != EXPECTED_SCENARIO_KEYS:
        raise RuntimeError("Evaluation does not match the published 20-scenario grid")
    seen: set[tuple[str, str]] = set()
    for scenario in scenarios:
        key = _scenario_key(scenario.get("transform"), scenario.get("severity"))
        if key in seen:
            raise RuntimeError(f"Duplicate evaluation scenario: {key}")
        seen.add(key)
        if scenario.get("num_samples") != expected_samples:
            raise RuntimeError(
                f"Scenario {key} contains {scenario.get('num_samples')} samples; "
                f"expected {expected_samples}"
            )
        metrics = scenario.get("metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"Scenario {key} has no metrics mapping")
        for metric in METRIC_KEYS:
            value = metrics.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise RuntimeError(f"Scenario {key} has invalid {metric}: {value!r}")
    return [dict(scenario) for scenario in scenarios]


def read_and_validate_predictions(
    path: Path,
    *,
    scenarios: Sequence[Mapping[str, Any]],
    expected_counts: Mapping[int, int],
) -> dict[str, Any]:
    """Validate detailed predictions and return aggregate clean confusion only."""

    if not path.is_file():
        raise RuntimeError(f"Evaluator did not write predictions: {path}")
    expected = {int(label): int(count) for label, count in expected_counts.items()}
    scenario_keys = {
        _scenario_key(item["transform"], item.get("severity")) for item in scenarios
    }
    groups: dict[tuple[str, str], list[tuple[str, int, float]]] = {
        key: [] for key in scenario_keys
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "label", "transform", "severity", "pred"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"Predictions are missing column(s): {', '.join(sorted(missing))}"
            )
        for row_number, row in enumerate(reader, start=2):
            if "wildfake" in row["image_path"].casefold():
                raise RuntimeError(f"Prediction row {row_number} references WildFake")
            key = _scenario_key(row["transform"], row["severity"])
            if key not in groups:
                raise RuntimeError(f"Prediction row {row_number} has unknown scenario {key}")
            if row["label"] not in {"0", "1"}:
                raise RuntimeError(f"Prediction row {row_number} has invalid label")
            probability = float(row["pred"])
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RuntimeError(f"Prediction row {row_number} has invalid probability")
            groups[key].append((row["image_path"], int(row["label"]), probability))

    clean_key = ("clean", "")
    identities_by_scenario: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for key, rows in groups.items():
        counts = {0: 0, 1: 0}
        identities: set[tuple[str, int]] = set()
        for image_path, label, _ in rows:
            identity = (image_path, label)
            if identity in identities:
                raise RuntimeError(f"Duplicate prediction identity in scenario {key}")
            identities.add(identity)
            counts[label] += 1
        if counts != expected:
            raise RuntimeError(
                f"Prediction counts for scenario {key} are {counts}; expected {expected}"
            )
        identities_by_scenario[key] = identities
    clean_identity = identities_by_scenario.get(clean_key)
    if clean_identity is None:
        raise RuntimeError("Predictions contain no clean scenario")
    for key, identities in identities_by_scenario.items():
        if identities != clean_identity:
            raise RuntimeError(f"Scenario {key} does not evaluate the clean image set")

    counts = {
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
    }
    for _, label, probability in groups[clean_key]:
        predicted = int(probability >= FIXED_THRESHOLD)
        if label == 0 and predicted == 0:
            counts["true_negative"] += 1
        elif label == 0:
            counts["false_positive"] += 1
        elif predicted == 0:
            counts["false_negative"] += 1
        else:
            counts["true_positive"] += 1
    real_count = expected[0]
    generated_count = expected[1]
    return {
        "raw": counts,
        "class_normalized": {
            "real": {
                "predicted_real": counts["true_negative"] / real_count,
                "predicted_generated": counts["false_positive"] / real_count,
            },
            "generated": {
                "predicted_real": counts["false_negative"] / generated_count,
                "predicted_generated": counts["true_positive"] / generated_count,
            },
        },
    }


def summarize_evaluation(
    evaluation: Mapping[str, Any],
    *,
    predictions_path: Path,
    expected_counts: Mapping[int, int],
) -> dict[str, Any]:
    expected_total = sum(int(value) for value in expected_counts.values())
    scenarios = validate_evaluation_artifact(
        evaluation, expected_samples=expected_total
    )
    confusion = read_and_validate_predictions(
        predictions_path,
        scenarios=scenarios,
        expected_counts=expected_counts,
    )
    robustness = summarize_scenarios(scenarios)
    worst = robustness["worst_case"]
    if worst is None:
        raise RuntimeError("Full evaluation has no transformed worst case")
    return {
        "clean": {
            "num_samples": expected_total,
            "metrics": {
                metric: float(scenarios[0]["metrics"][metric]) for metric in METRIC_KEYS
            },
            "confusion": confusion,
        },
        "robustness": {
            "mean_transformed": {
                metric: float(robustness["mean_transformed"][metric])
                for metric in METRIC_KEYS
            },
            "worst_case": {
                "metric": "roc_auc",
                "transform": str(worst["transform"]),
                "severity": worst["severity"],
                "value": float(worst["value"]),
            },
            "clean_to_worst_roc_auc_drop": float(
                robustness["clean_to_worst_roc_auc_drop"]
            ),
        },
    }


def _metric_deltas(v1: Mapping[str, Any], v2: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "definition": "v2 minus v1; negative is better for FPR, FNR, and Brier score",
        "clean_metrics": {
            metric: float(v2["clean"]["metrics"][metric])
            - float(v1["clean"]["metrics"][metric])
            for metric in METRIC_KEYS
        },
        "mean_transformed_metrics": {
            metric: float(v2["robustness"]["mean_transformed"][metric])
            - float(v1["robustness"]["mean_transformed"][metric])
            for metric in METRIC_KEYS
        },
        "worst_case_roc_auc": float(v2["robustness"]["worst_case"]["value"])
        - float(v1["robustness"]["worst_case"]["value"]),
        "clean_to_worst_roc_auc_drop": float(
            v2["robustness"]["clean_to_worst_roc_auc_drop"]
        )
        - float(v1["robustness"]["clean_to_worst_roc_auc_drop"]),
    }


def _public_manifest(audit: ManifestAudit) -> dict[str, Any]:
    return {
        "name": audit.name,
        "dataset_id": audit.dataset_id,
        "split": "test",
        "total_count": audit.total_count,
        "class_counts": {
            "real": audit.class_counts[0],
            "generated": audit.class_counts[1],
        },
        "manifest_sha256": audit.manifest_sha256,
        "test_content_digest": audit.test_content_digest,
    }


def build_public_summary(
    *,
    model_identity: ModelIdentity,
    genimage_manifest: ManifestAudit,
    sid_manifest: ManifestAudit,
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build the aggregate-only comparison that may be committed or submitted."""

    public_results: dict[str, Any] = {}
    for dataset_key in ("genimage", "sid"):
        if dataset_key not in results or set(results[dataset_key]) != {"v1", "v2"}:
            raise ValueError(f"Missing complete v1/v2 results for {dataset_key}")
        v1 = dict(results[dataset_key]["v1"])
        v2 = dict(results[dataset_key]["v2"])
        public_results[dataset_key] = {
            "v1": v1,
            "v2": v2,
            "delta_v2_minus_v1": _metric_deltas(v1, v2),
        }
    summary = {
        "schema_version": 1,
        "scope": (
            "Held-out GenImage and SID comparison for the frozen v1 and warm-started "
            "v2 checkpoints. WildFake was not accessed or used for model selection."
        ),
        "decision_threshold": FIXED_THRESHOLD,
        "seed": FIXED_SEED,
        "scenario_mode": "full",
        "scenario_count": EXPECTED_SCENARIO_COUNT,
        "models": {
            "v1": {
                "role": "frozen parent",
                "checkpoint_sha256": model_identity.v1_sha256,
            },
            "v2": {
                "role": "warm-start candidate; not deployed automatically",
                "checkpoint_sha256": model_identity.v2_sha256,
                "parent_checkpoint_sha256": model_identity.parent_sha256,
            },
        },
        "datasets": {
            "genimage": _public_manifest(genimage_manifest),
            "sid": _public_manifest(sid_manifest),
        },
        "results": public_results,
    }
    assert_public_summary_sanitized(summary)
    return summary


def assert_public_summary_sanitized(summary: Mapping[str, Any]) -> None:
    """Reject common local-path and per-image prediction leakage."""

    forbidden_keys = {
        "path",
        "paths",
        "manifest_path",
        "checkpoint_path",
        "root_dir",
        "predictions",
        "prediction_rows",
        "image_path",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise RuntimeError(f"Public summary contains forbidden key: {key}")
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            lowered = value.casefold()
            if (
                REPOSITORY_ROOT_TEXT.casefold() in lowered
                or "/users/" in lowered
                or "/kaggle/" in lowered
                or "file://" in lowered
                or "predictions.csv" in lowered
            ):
                raise RuntimeError("Public summary contains a local path or prediction file")

    walk(summary)
    json.dumps(summary, allow_nan=False)


def _format_metric(value: object, *, signed: bool = False) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def write_public_report(path: Path, summary: Mapping[str, Any]) -> None:
    assert_public_summary_sanitized(summary)
    models = summary["models"]
    lines = [
        "# GenImage v2 Held-Out Evaluation",
        "",
        f"> **Scope:** {summary['scope']}",
        "",
        "The v2 checkpoint was warm-started from v1. Results below use untouched "
        "test splits and a fixed 0.50 threshold. They support a manual deployment "
        "decision; this workflow does not replace the application checkpoint.",
        "",
        "## Locked identity",
        "",
        f"- v1 checkpoint SHA-256: `{models['v1']['checkpoint_sha256']}`",
        f"- v2 checkpoint SHA-256: `{models['v2']['checkpoint_sha256']}`",
        f"- v2 parent SHA-256: `{models['v2']['parent_checkpoint_sha256']}`",
        f"- Fixed threshold: `{summary['decision_threshold']:.2f}`",
        f"- Evaluation scenarios: {summary['scenario_count']} (clean plus 19 transforms)",
        "",
    ]
    for dataset_key, title in (
        ("genimage", "GenImage held-out test"),
        ("sid", "SID regression test"),
    ):
        dataset = summary["datasets"][dataset_key]
        result = summary["results"][dataset_key]
        v1 = result["v1"]
        v2 = result["v2"]
        delta = result["delta_v2_minus_v1"]
        lines.extend(
            [
                f"## {title}",
                "",
                f"{dataset['total_count']:,} images: "
                f"{dataset['class_counts']['real']:,} real and "
                f"{dataset['class_counts']['generated']:,} generated. Test content "
                f"digest: `{dataset['test_content_digest']}`.",
                "",
                "### Clean metrics",
                "",
                "| Metric | v1 | v2 | v2 − v1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, key in METRIC_LABELS:
            lines.append(
                f"| {label} | {_format_metric(v1['clean']['metrics'][key])} | "
                f"{_format_metric(v2['clean']['metrics'][key])} | "
                f"{_format_metric(delta['clean_metrics'][key], signed=True)} |"
            )
        lines.extend(
            [
                "",
                "### Mean transformed metrics",
                "",
                "| Metric | v1 | v2 | v2 − v1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, key in METRIC_LABELS:
            lines.append(
                f"| {label} | "
                f"{_format_metric(v1['robustness']['mean_transformed'][key])} | "
                f"{_format_metric(v2['robustness']['mean_transformed'][key])} | "
                f"{_format_metric(delta['mean_transformed_metrics'][key], signed=True)} |"
            )
        for model_key, model_label in (("v1", "v1"), ("v2", "v2")):
            model_result = result[model_key]
            worst = model_result["robustness"]["worst_case"]
            raw = model_result["clean"]["confusion"]["raw"]
            normalized = model_result["clean"]["confusion"]["class_normalized"]
            lines.extend(
                [
                    "",
                    f"- {model_label} worst scenario: `{worst['transform']}:{worst['severity']}` "
                    f"at ROC-AUC {_format_metric(worst['value'])}; clean-to-worst "
                    "drop "
                    f"{_format_metric(model_result['robustness']['clean_to_worst_roc_auc_drop'])}.",
                    f"- {model_label} clean confusion (TN / FP / FN / TP): "
                    f"{raw['true_negative']:,} / {raw['false_positive']:,} / "
                    f"{raw['false_negative']:,} / {raw['true_positive']:,}.",
                    f"- {model_label} class-normalized clean confusion: real → real "
                    f"{_format_metric(normalized['real']['predicted_real'])}, real → "
                    f"generated {_format_metric(normalized['real']['predicted_generated'])}; "
                    f"generated → real "
                    f"{_format_metric(normalized['generated']['predicted_real'])}, generated → "
                    f"generated "
                    f"{_format_metric(normalized['generated']['predicted_generated'])}.",
                ]
            )
        lines.extend(
            [
                "",
                "Deltas are v2 minus v1. Positive is normally better, except for "
                "false-positive rate, false-negative rate, Brier score, and degradation.",
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence handling",
            "",
            "The compact JSON, report, and figure contain aggregate results only. "
            "Detailed manifests, per-image predictions, raw scenario metrics, source "
            "images, and execution paths remain in the local audit layer.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def save_public_figure(summary: Mapping[str, Any], path: Path) -> None:
    """Render an aggregate-only v1/v2 comparison for both held-out datasets."""

    assert_public_summary_sanitized(summary)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Clean\nROC-AUC",
        "Clean\nbalanced acc.",
        "Mean transformed\nROC-AUC",
        "Worst\nROC-AUC",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    for axis, dataset_key, title in zip(
        axes,
        ("genimage", "sid"),
        ("GenImage held-out test", "SID regression test"),
    ):
        result = summary["results"][dataset_key]
        v1 = [
            result["v1"]["clean"]["metrics"]["roc_auc"],
            result["v1"]["clean"]["metrics"]["balanced_accuracy"],
            result["v1"]["robustness"]["mean_transformed"]["roc_auc"],
            result["v1"]["robustness"]["worst_case"]["value"],
        ]
        v2 = [
            result["v2"]["clean"]["metrics"]["roc_auc"],
            result["v2"]["clean"]["metrics"]["balanced_accuracy"],
            result["v2"]["robustness"]["mean_transformed"]["roc_auc"],
            result["v2"]["robustness"]["worst_case"]["value"],
        ]
        positions = list(range(len(labels)))
        width = 0.36
        v1_bars = axis.bar(
            [position - width / 2 for position in positions],
            v1,
            width,
            label="v1 frozen",
            color="#64748b",
        )
        v2_bars = axis.bar(
            [position + width / 2 for position in positions],
            v2,
            width,
            label="v2 candidate",
            color="#2563eb",
        )
        axis.set_title(title)
        axis.set_xticks(positions, labels=labels)
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.bar_label(v1_bars, fmt="%.3f", padding=2, fontsize=8, rotation=90)
        axis.bar_label(v2_bars, fmt="%.3f", padding=2, fontsize=8, rotation=90)
    axes[0].set_ylabel("Metric value")
    axes[1].legend(loc="lower right", frameon=False)
    figure.suptitle("RealityCheck v1 vs v2 · fixed threshold 0.50 · 20 scenarios")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=170, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def _run_one(
    *,
    manifest_path: Path,
    root_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    expected_counts: Mapping[int, int],
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = evaluate_checkpoint(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        split="test",
        output_dir=output_dir,
        root_dir=root_dir,
        scenario_mode="full",
        batch_size=batch_size,
        num_workers=num_workers,
        device_name=device,
        image_size=IMAGE_SIZE,
        threshold=FIXED_THRESHOLD,
        seed=FIXED_SEED,
    )
    summary = summarize_evaluation(
        evaluation,
        predictions_path=output_dir / "predictions.csv",
        expected_counts=expected_counts,
    )
    return evaluation, summary


def run_evaluation(
    *,
    genimage_manifest_path: Path,
    genimage_root: Path,
    sid_manifest_path: Path,
    sid_root: Path,
    v1_checkpoint: Path,
    v2_checkpoint: Path,
    v2_metadata_path: Path,
    audit_root: Path,
    public_json: Path,
    public_report: Path,
    public_figure: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    started = time.monotonic()
    for path, description in (
        (genimage_manifest_path, "GenImage manifest"),
        (genimage_root, "GenImage root"),
        (sid_manifest_path, "SID manifest"),
        (sid_root, "SID root"),
    ):
        _reject_wildfake_input(path, description=description)

    print("Validating locked GenImage and SID test manifests...", flush=True)
    genimage_audit = validate_test_manifest(
        genimage_manifest_path,
        name="GenImage",
        expected_dataset_id=GENIMAGE_DATASET_ID,
        expected_counts=GENIMAGE_EXPECTED_COUNTS,
    )
    sid_audit = validate_test_manifest(
        sid_manifest_path,
        name="SID",
        expected_dataset_id=SID_DATASET_ID,
        expected_counts=SID_EXPECTED_COUNTS,
    )
    ensure_disjoint_test_content(genimage_audit, sid_audit)
    model_identity = verify_checkpoint_lineage(
        v1_checkpoint,
        v2_checkpoint,
        v2_metadata_path,
        expected_v1_sha256=FROZEN_V1_SHA256,
    )

    jobs = (
        (
            "v1_genimage",
            "genimage",
            "v1",
            genimage_manifest_path,
            genimage_root,
            v1_checkpoint,
            GENIMAGE_EXPECTED_COUNTS,
        ),
        (
            "v2_genimage",
            "genimage",
            "v2",
            genimage_manifest_path,
            genimage_root,
            v2_checkpoint,
            GENIMAGE_EXPECTED_COUNTS,
        ),
        (
            "v1_sid",
            "sid",
            "v1",
            sid_manifest_path,
            sid_root,
            v1_checkpoint,
            SID_EXPECTED_COUNTS,
        ),
        (
            "v2_sid",
            "sid",
            "v2",
            sid_manifest_path,
            sid_root,
            v2_checkpoint,
            SID_EXPECTED_COUNTS,
        ),
    )
    results: dict[str, dict[str, dict[str, Any]]] = {
        "genimage": {},
        "sid": {},
    }
    audit_identities: dict[str, dict[str, Any]] = {}
    for job_name, dataset_key, model_key, manifest, root, checkpoint, counts in jobs:
        output_dir = audit_root / job_name
        print(f"Evaluating {job_name}: 20 scenarios at threshold 0.50...", flush=True)
        evaluation, result = _run_one(
            manifest_path=manifest,
            root_dir=root,
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            expected_counts=counts,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        results[dataset_key][model_key] = result
        audit_identities[job_name] = {
            "metrics_sha256": sha256_file(output_dir / "metrics.json"),
            "predictions_sha256": sha256_file(output_dir / "predictions.csv"),
            "scenario_count": len(evaluation["scenarios"]),
        }

    public_summary = build_public_summary(
        model_identity=model_identity,
        genimage_manifest=genimage_audit,
        sid_manifest=sid_audit,
        results=results,
    )
    _write_json(public_json, public_summary)
    write_public_report(public_report, public_summary)
    save_public_figure(public_summary, public_figure)

    execution = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "threshold": FIXED_THRESHOLD,
        "scenario_count": EXPECTED_SCENARIO_COUNT,
        "inputs": {
            "genimage_manifest_path": str(genimage_manifest_path.resolve()),
            "genimage_root": str(genimage_root.resolve()),
            "sid_manifest_path": str(sid_manifest_path.resolve()),
            "sid_root": str(sid_root.resolve()),
            "v1_checkpoint": str(v1_checkpoint.resolve()),
            "v2_checkpoint": str(v2_checkpoint.resolve()),
            "v2_metadata_path": str(v2_metadata_path.resolve()),
        },
        "model_identity": {
            "v1_sha256": model_identity.v1_sha256,
            "v2_sha256": model_identity.v2_sha256,
            "parent_sha256": model_identity.parent_sha256,
        },
        "audit_artifacts": audit_identities,
        "public_artifacts": {
            "summary": str(public_json.resolve()),
            "report": str(public_report.resolve()),
            "figure": str(public_figure.resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _write_json(audit_root / "execution_metadata.json", execution)
    print(f"Comparison complete. Public summary: {public_json}", flush=True)
    return public_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen v1 and warm-started v2 on held-out GenImage and SID "
            "tests across the complete 20-scenario grid."
        )
    )
    parser.add_argument(
        "--genimage-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data/processed/genimage_v2_manifest.csv",
    )
    parser.add_argument("--genimage-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--sid-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data/processed/manifest.csv",
    )
    parser.add_argument("--sid-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--v1-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/checkpoints/model.safetensors",
    )
    parser.add_argument(
        "--v2-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/checkpoints/model_v2.safetensors",
    )
    parser.add_argument(
        "--v2-metadata",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/checkpoints/model_v2_metadata.json",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/metrics/genimage_v2_audit",
    )
    parser.add_argument(
        "--public-json",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/metrics/genimage_v2_summary.json",
    )
    parser.add_argument(
        "--public-report",
        type=Path,
        default=REPOSITORY_ROOT / "docs/submission/genimage-v2-report.md",
    )
    parser.add_argument(
        "--public-figure",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/figures/genimage_v2_comparison.png",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_evaluation(
            genimage_manifest_path=args.genimage_manifest,
            genimage_root=args.genimage_root,
            sid_manifest_path=args.sid_manifest,
            sid_root=args.sid_root,
            v1_checkpoint=args.v1_checkpoint,
            v2_checkpoint=args.v2_checkpoint,
            v2_metadata_path=args.v2_metadata,
            audit_root=args.audit_root,
            public_json=args.public_json,
            public_report=args.public_report,
            public_figure=args.public_figure,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
