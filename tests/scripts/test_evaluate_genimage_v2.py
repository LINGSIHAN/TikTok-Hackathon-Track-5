import csv
import hashlib
import json
from pathlib import Path

import pytest

import scripts.evaluate_genimage_v2 as v2_eval
from scripts.evaluate_genimage_v2 import (
    EXPECTED_SCENARIO_KEYS,
    ManifestAudit,
    ModelIdentity,
    assert_public_summary_sanitized,
    build_public_summary,
    ensure_disjoint_test_content,
    read_and_validate_predictions,
    run_evaluation,
    save_public_figure,
    summarize_evaluation,
    validate_evaluation_artifact,
    validate_test_manifest,
    verify_checkpoint_lineage,
    write_public_report,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_manifest(
    path: Path,
    *,
    dataset_id: str,
    real: int,
    generated: int,
    prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "split", "dataset", "source_id", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, count in ((0, real), (1, generated)):
            for index in range(count):
                identity = f"{prefix}-{label}-{index}"
                writer.writerow(
                    {
                        "path": f"images/{identity}.jpg",
                        "label": label,
                        "split": "test",
                        "dataset": dataset_id,
                        "source_id": identity,
                        "sha256": _digest(identity),
                    }
                )


def _metrics(value: float) -> dict[str, float]:
    return {
        "roc_auc": value,
        "average_precision": min(0.99, value + 0.02),
        "balanced_accuracy": value - 0.03,
        "f1": value - 0.02,
        "false_positive_rate": 1.0 - value,
        "false_negative_rate": 0.9 - value,
        "brier_score": 0.5 - value / 3.0,
    }


def _evaluation(num_samples: int, base: float = 0.90) -> dict:
    scenarios = []
    for index, (transform, severity) in enumerate(EXPECTED_SCENARIO_KEYS):
        scenarios.append(
            {
                "transform": transform,
                "severity": None if severity == "" else severity,
                "num_samples": num_samples,
                "metrics": _metrics(base - index * 0.005),
            }
        )
    return {
        "schema_version": 1,
        "scenario_mode": "full",
        "threshold": 0.5,
        "scenarios": scenarios,
    }


def _write_predictions(
    path: Path,
    *,
    real: int,
    generated: int,
    false_positive: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "transform", "severity", "pred"],
        )
        writer.writeheader()
        for transform, severity in EXPECTED_SCENARIO_KEYS:
            for label, count in ((0, real), (1, generated)):
                for index in range(count):
                    probability = 0.9 if label else 0.1
                    if false_positive and label == 0 and index == 0:
                        probability = 0.8
                    writer.writerow(
                        {
                            "image_path": f"images/{label}-{index}.jpg",
                            "label": label,
                            "transform": transform,
                            "severity": severity,
                            "pred": probability,
                        }
                    )


def _manifest_audit(name: str, prefix: str, count: int) -> ManifestAudit:
    hashes = frozenset(_digest(f"{prefix}-{index}") for index in range(count))
    return ManifestAudit(
        name=name,
        dataset_id=prefix,
        total_count=count,
        class_counts={0: count // 2, 1: count // 2},
        manifest_sha256=_digest(prefix + "-manifest"),
        test_content_digest=_digest(prefix + "-content"),
        content_hashes=hashes,
    )


def test_manifest_validation_requires_exact_balanced_unique_test_rows(tmp_path):
    manifest = tmp_path / "genimage.csv"
    _write_manifest(
        manifest,
        dataset_id=v2_eval.GENIMAGE_DATASET_ID,
        real=2,
        generated=2,
        prefix="gen",
    )

    audit = validate_test_manifest(
        manifest,
        name="GenImage",
        expected_dataset_id=v2_eval.GENIMAGE_DATASET_ID,
        expected_counts={0: 2, 1: 2},
    )

    assert audit.total_count == 4
    assert audit.class_counts == {0: 2, 1: 2}
    assert len(audit.content_hashes) == 4
    assert len(audit.manifest_sha256) == 64
    assert len(audit.test_content_digest) == 64

    with pytest.raises(RuntimeError, match="expected exactly"):
        validate_test_manifest(
            manifest,
            name="GenImage",
            expected_dataset_id=v2_eval.GENIMAGE_DATASET_ID,
            expected_counts={0: 3, 1: 2},
        )


def test_manifest_rejects_forbidden_dataset_and_cross_dataset_duplicate(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        dataset_id=v2_eval.GENIMAGE_DATASET_ID,
        real=1,
        generated=1,
        prefix="gen",
    )
    text = manifest.read_text(encoding="utf-8").replace(
        "images/gen-0-0.jpg", "data/external/wildfake/real.jpg"
    )
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError, match="references WildFake"):
        validate_test_manifest(
            manifest,
            name="GenImage",
            expected_dataset_id=v2_eval.GENIMAGE_DATASET_ID,
            expected_counts={0: 1, 1: 1},
        )

    first = _manifest_audit("one", "one", 2)
    second = ManifestAudit(
        name="two",
        dataset_id="two",
        total_count=2,
        class_counts={0: 1, 1: 1},
        manifest_sha256=_digest("two-manifest"),
        test_content_digest=_digest("two-content"),
        content_hashes=frozenset({next(iter(first.content_hashes)), _digest("new")}),
    )
    with pytest.raises(RuntimeError, match="share 1 content hash"):
        ensure_disjoint_test_content(first, second)


def test_checkpoint_lineage_verifies_file_hash_json_and_safetensors_metadata(
    monkeypatch, tmp_path
):
    v1 = tmp_path / "model.safetensors"
    v2 = tmp_path / "model_v2.safetensors"
    metadata = tmp_path / "model_v2_metadata.json"
    v1.write_bytes(b"frozen-v1")
    v2.write_bytes(b"trained-v2")
    expected = hashlib.sha256(v1.read_bytes()).hexdigest()
    metadata.write_text(
        json.dumps({"parent_checkpoint_sha256": expected}), encoding="utf-8"
    )
    monkeypatch.setattr(
        v2_eval,
        "_read_checkpoint_metadata",
        lambda _: {
            "parent_checkpoint_sha256": expected,
            "architecture": "efficientnet_b0_binary",
            "image_size": "224",
            "preprocessing_contract": v2_eval.PREPROCESSING_CONTRACT,
        },
    )

    identity = verify_checkpoint_lineage(
        v1, v2, metadata, expected_v1_sha256=expected
    )

    assert identity.v1_sha256 == expected
    assert identity.v2_sha256 == hashlib.sha256(v2.read_bytes()).hexdigest()
    assert identity.parent_sha256 == expected

    metadata.write_text(
        json.dumps({"parent_checkpoint_sha256": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="does not identify the frozen v1"):
        verify_checkpoint_lineage(v1, v2, metadata, expected_v1_sha256=expected)


def test_evaluation_requires_fixed_threshold_and_exact_published_grid():
    valid = _evaluation(4)
    assert len(validate_evaluation_artifact(valid, expected_samples=4)) == 20

    wrong_threshold = dict(valid, threshold=0.51)
    with pytest.raises(RuntimeError, match="fixed at 0.50"):
        validate_evaluation_artifact(wrong_threshold, expected_samples=4)

    wrong_grid = _evaluation(4)
    wrong_grid["scenarios"][-1]["transform"] = "invented"
    with pytest.raises(RuntimeError, match="published 20-scenario grid"):
        validate_evaluation_artifact(wrong_grid, expected_samples=4)


def test_prediction_validation_returns_clean_confusion_and_checks_every_scenario(
    tmp_path,
):
    predictions = tmp_path / "predictions.csv"
    _write_predictions(predictions, real=2, generated=2, false_positive=True)
    scenarios = _evaluation(4)["scenarios"]

    confusion = read_and_validate_predictions(
        predictions,
        scenarios=scenarios,
        expected_counts={0: 2, 1: 2},
    )

    assert confusion["raw"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 0,
        "true_positive": 2,
    }
    assert confusion["class_normalized"]["real"]["predicted_generated"] == 0.5


def test_compact_public_outputs_have_metrics_deltas_and_no_local_paths(tmp_path):
    predictions = tmp_path / "predictions.csv"
    _write_predictions(predictions, real=2, generated=2)
    v1 = summarize_evaluation(
        _evaluation(4, base=0.82),
        predictions_path=predictions,
        expected_counts={0: 2, 1: 2},
    )
    v2 = summarize_evaluation(
        _evaluation(4, base=0.88),
        predictions_path=predictions,
        expected_counts={0: 2, 1: 2},
    )
    summary = build_public_summary(
        model_identity=ModelIdentity("a" * 64, "b" * 64, "a" * 64),
        genimage_manifest=_manifest_audit("GenImage", "genimage", 4),
        sid_manifest=_manifest_audit("SID", "sid", 4),
        results={
            "genimage": {"v1": v1, "v2": v2},
            "sid": {"v1": v1, "v2": v2},
        },
    )

    assert summary["results"]["genimage"]["delta_v2_minus_v1"][
        "clean_metrics"
    ]["roc_auc"] == pytest.approx(0.06)
    serialized = json.dumps(summary)
    assert "image_path" not in serialized
    assert str(tmp_path) not in serialized
    assert "predictions.csv" not in serialized

    report = tmp_path / "report.md"
    figure = tmp_path / "comparison.png"
    write_public_report(report, summary)
    save_public_figure(summary, figure)
    assert report.is_file()
    report_text = report.read_text(encoding="utf-8")
    assert "v2 − v1" in report_text
    assert "class-normalized clean confusion" in report_text
    assert str(tmp_path) not in report_text
    assert figure.stat().st_size > 1_000

    unsafe = dict(summary)
    unsafe["image_path"] = "/kaggle/input/private.jpg"
    with pytest.raises(RuntimeError, match="forbidden key"):
        assert_public_summary_sanitized(unsafe)


def test_run_orchestrates_four_full_evaluations_and_writes_two_layers(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(v2_eval, "GENIMAGE_EXPECTED_COUNTS", {0: 1, 1: 1})
    monkeypatch.setattr(v2_eval, "SID_EXPECTED_COUNTS", {0: 1, 1: 1})
    genimage_manifest = tmp_path / "genimage.csv"
    sid_manifest = tmp_path / "sid.csv"
    _write_manifest(
        genimage_manifest,
        dataset_id=v2_eval.GENIMAGE_DATASET_ID,
        real=1,
        generated=1,
        prefix="gen",
    )
    _write_manifest(
        sid_manifest,
        dataset_id=v2_eval.SID_DATASET_ID,
        real=1,
        generated=1,
        prefix="sid",
    )
    monkeypatch.setattr(
        v2_eval,
        "verify_checkpoint_lineage",
        lambda *args, **kwargs: ModelIdentity("a" * 64, "b" * 64, "a" * 64),
    )
    calls = []

    def fake_evaluate_checkpoint(**kwargs):
        calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        base = 0.88 if "model_v2" in str(kwargs["checkpoint_path"]) else 0.82
        evaluation = _evaluation(2, base=base)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(evaluation), encoding="utf-8"
        )
        _write_predictions(output_dir / "predictions.csv", real=1, generated=1)
        return evaluation

    monkeypatch.setattr(v2_eval, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr(
        v2_eval,
        "save_public_figure",
        lambda summary, path: (
            path.parent.mkdir(parents=True, exist_ok=True),
            path.write_bytes(b"aggregate figure"),
        ),
    )
    public_json = tmp_path / "public/summary.json"
    public_report = tmp_path / "public/report.md"
    public_figure = tmp_path / "public/figure.png"
    audit_root = tmp_path / "audit"

    summary = run_evaluation(
        genimage_manifest_path=genimage_manifest,
        genimage_root=tmp_path,
        sid_manifest_path=sid_manifest,
        sid_root=tmp_path,
        v1_checkpoint=tmp_path / "model.safetensors",
        v2_checkpoint=tmp_path / "model_v2.safetensors",
        v2_metadata_path=tmp_path / "model_v2_metadata.json",
        audit_root=audit_root,
        public_json=public_json,
        public_report=public_report,
        public_figure=public_figure,
        batch_size=8,
        num_workers=0,
        device="cpu",
    )

    assert len(calls) == 4
    assert all(call["scenario_mode"] == "full" for call in calls)
    assert all(call["threshold"] == 0.5 for call in calls)
    assert all(call["seed"] == 42 for call in calls)
    assert public_json.is_file() and public_report.is_file() and public_figure.is_file()
    assert (audit_root / "execution_metadata.json").is_file()
    assert {
        path.name for path in audit_root.iterdir() if path.is_dir()
    } == {"v1_genimage", "v2_genimage", "v1_sid", "v2_sid"}
    public_text = public_json.read_text(encoding="utf-8")
    assert str(tmp_path) not in public_text
    assert "predictions.csv" not in public_text
    assert summary["scenario_count"] == 20
