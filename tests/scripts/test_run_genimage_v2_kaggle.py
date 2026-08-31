from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from scripts import run_genimage_v2_kaggle as runner


FIELDS = ["path", "label", "split", "dataset", "source_id", "sha256"]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(name: str, label: str, split: str, dataset: str, digest: str) -> dict[str, str]:
    return {
        "path": f"data/raw/{name}.jpg",
        "label": label,
        "split": split,
        "dataset": dataset,
        "source_id": f"{dataset}:{name}",
        "sha256": digest,
    }


def test_main_requires_explicit_license_confirmation(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="license-confirmed"):
        runner.main(["--input-root", str(tmp_path)])


def test_cuda_preflight_rejects_non_t4_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Tesla P100")

    with pytest.raises(RuntimeError, match="requires a T4"):
        runner.validate_cuda()


def test_validate_prepared_v2_data_reconciles_and_rejects_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid_rows = [_row("sid-real", "0", "train", "sid", "a" * 64)]
    gen_rows = [_row("gen-fake", "1", "test", "genimage", "b" * 64)]
    combined_rows = [sid_rows[0], gen_rows[0]]
    _write_manifest(tmp_path / runner.SID_MANIFEST, sid_rows)
    _write_manifest(tmp_path / runner.GENIMAGE_MANIFEST, gen_rows)
    _write_manifest(tmp_path / runner.COMBINED_MANIFEST, combined_rows)

    monkeypatch.setattr(runner, "EXPECTED_GENIMAGE_TOTAL", 1)
    monkeypatch.setattr(runner, "EXPECTED_COMBINED_TOTAL", 2)
    monkeypatch.setattr(runner, "EXPECTED_GENIMAGE_COUNTS", {("test", "1"): 1})
    monkeypatch.setattr(
        runner,
        "EXPECTED_COMBINED_COUNTS",
        {("train", "0"): 1, ("test", "1"): 1},
    )
    summary_path = tmp_path / runner.GENIMAGE_SUMMARY
    summary = {
        "dataset": runner.GENIMAGE_DATASET_ID,
        "dataset_version": runner.GENIMAGE_DATASET_VERSION,
        "seed": 42,
        "total": 1,
        "wildfake_used": False,
        "class_counts": {"0_real": 0, "1_generated": 1},
        "split_counts": {
            "train": {"0": 0, "1": 0},
            "val": {"0": 0, "1": 0},
            "test": {"0": 0, "1": 1},
        },
        "combined_counts": {
            "total": 2,
            "train": 1,
            "val": 0,
            "test": 1,
            "sid_train_rows": 1,
            "genimage_rows": 1,
        },
        "inventory": {
            "file_count": runner.GENIMAGE_INVENTORY_FILES,
            "total_bytes": runner.GENIMAGE_INVENTORY_BYTES,
            "metadata_sha256": runner.GENIMAGE_METADATA_SHA256,
            "generator_image_counts": {
                generator: 2_500 for generator in runner.GENIMAGE_GENERATORS
            },
            "nature_image_count": 5_828,
        },
        "license_confirmation": {"confirmed": True},
        "manifest_sha256": runner.file_sha256(tmp_path / runner.GENIMAGE_MANIFEST),
        "combined_manifest_sha256": runner.file_sha256(
            tmp_path / runner.COMBINED_MANIFEST
        ),
        "sid_manifest_sha256": runner.file_sha256(tmp_path / runner.SID_MANIFEST),
        "selected_digest": "d" * 64,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert runner.validate_prepared_v2_data(tmp_path)["total"] == 1

    gen_rows[0]["sha256"] = "a" * 64
    combined_rows[1]["sha256"] = "a" * 64
    _write_manifest(tmp_path / runner.GENIMAGE_MANIFEST, gen_rows)
    _write_manifest(tmp_path / runner.COMBINED_MANIFEST, combined_rows)
    with pytest.raises(RuntimeError, match="overlapping image hashes"):
        runner.validate_prepared_v2_data(tmp_path)


def test_safe_export_dir_is_scoped_to_requested_root(tmp_path: Path) -> None:
    assert runner._safe_export_dir(tmp_path) == (
        tmp_path.resolve() / "genimage_v2_export"
    )
    with pytest.raises(RuntimeError, match="filesystem root"):
        runner._safe_export_dir(Path("/"))


def test_package_export_has_explicit_audit_and_public_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    output_root = tmp_path / "output"
    files = {
        runner.V1_CHECKPOINT: b"frozen-v1",
        runner.V2_CHECKPOINT: b"candidate-v2",
        Path("artifacts/checkpoints/model_v2_metadata.json"): b"{}",
        Path("artifacts/metrics/genimage_v2_training_history.json"): b"[]",
        runner.GENIMAGE_MANIFEST: b"manifest",
        runner.GENIMAGE_SUMMARY: json.dumps(
            {
                "selected_digest": "a" * 64,
                "inventory": {"inventory_digest": "b" * 64},
            }
        ).encode("utf-8"),
        runner.COMBINED_MANIFEST: b"combined",
        runner.SID_MANIFEST: b"sid",
        runner.SID_SUMMARY: b"{}",
        runner.V2_CONFIG: b"seed: 42\n",
        runner.PUBLIC_SUMMARY: b"{}",
        runner.PUBLIC_FIGURE: b"png",
        runner.PUBLIC_REPORT: b"# Report\n",
        Path("requirements.txt"): b"Pillow\n",
        Path("requirements-train.txt"): b"-r requirements.txt\n",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    raw_metrics = root / runner.AUDIT_ROOT / "v1_genimage/metrics.json"
    raw_metrics.parent.mkdir(parents=True, exist_ok=True)
    raw_metrics.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner, "V1_SHA256", runner.file_sha256(root / runner.V1_CHECKPOINT)
    )
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "example-package==1.0\n",
    )

    archive = runner.package_export(
        root,
        output_root,
        repository={"commit": "a" * 40},
        environment={"gpu": "test"},
        license_confirmation="confirmed for test",
    )

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert {
            "README.txt",
            "public/genimage_v2_summary.json",
            "public/genimage_v2_comparison.png",
            "public/genimage-v2-report.md",
            "local_audit/checkpoints/model_v2.safetensors",
            "local_audit/manifests/genimage_v2_manifest.csv",
            "local_audit/evaluations/v1_genimage/metrics.json",
            "local_audit/execution_metadata.json",
            "local_audit/environment/pip_freeze.txt",
        }.issubset(names)
        assert {
            name for name in names if name.startswith("public/") and name != "public/"
        } == {
            "public/genimage_v2_summary.json",
            "public/genimage_v2_comparison.png",
            "public/genimage-v2-report.md",
        }
        context = json.loads(
            bundle.read("local_audit/execution_metadata.json").decode("utf-8")
        )
        assert context["wildfake_used"] is False
        assert context["automatic_promotion"] is False


def test_enrich_v2_metadata_records_lineage_without_promoting(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / runner.V2_CHECKPOINT
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"candidate-v2")
    metadata_path = tmp_path / runner.V2_METADATA
    metadata_path.write_text(
        json.dumps(
            {
                "parent_checkpoint_sha256": runner.V1_SHA256,
                "checkpoint_sha256": runner.file_sha256(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    provenance = tmp_path / runner.GENIMAGE_SUMMARY
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "selected_digest": "a" * 64,
                "inventory": {"inventory_digest": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    for relative, payload in (
        (runner.SID_MANIFEST, b"sid"),
        (runner.GENIMAGE_MANIFEST, b"genimage"),
        (runner.COMBINED_MANIFEST, b"combined"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    metadata = runner.enrich_v2_metadata(
        tmp_path,
        repository={"commit": "c" * 40},
        environment={"gpu": "T4"},
        license_confirmation="confirmed",
    )

    assert metadata["repository"]["commit"] == "c" * 40
    assert metadata["environment"]["gpu"] == "T4"
    assert metadata["dataset_lineage"]["genimage_selected_digest"] == "a" * 64
    assert metadata["wildfake_used"] is False
    assert metadata["automatic_promotion"] is False


def test_runner_contains_no_wildfake_data_path() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    forbidden = "data" + "/external/" + "wildfake_demo"
    assert forbidden not in source


def test_v2_config_matches_strict_warm_start_contract() -> None:
    from src.training.config import load_config

    runner.validate_v2_training_config(runner.REPOSITORY_ROOT)
    config = load_config(runner.REPOSITORY_ROOT / runner.V2_CONFIG)

    assert config.data.manifest_path == str(runner.COMBINED_MANIFEST)
    assert config.model.pretrained is False
    assert config.model.freeze_backbone is True
    assert config.model.unfreeze_last_blocks == 1
    assert config.initialization is not None
    assert config.initialization.checkpoint_path == str(runner.V1_CHECKPOINT)
    assert config.initialization.expected_sha256 == runner.V1_SHA256
    assert config.initialization.freeze_frozen_batchnorm is True
    assert config.output.checkpoint_path == str(runner.V2_CHECKPOINT)
    assert config.output.metadata_path == str(runner.V2_METADATA)
    assert config.output.history_path == str(runner.V2_HISTORY)
    assert config.training.epochs == 3
    assert config.training.learning_rate == pytest.approx(3e-5)
    assert config.training.weight_decay == pytest.approx(1e-4)
    assert config.training.patience == 1
    assert config.training.mixed_precision is True
    assert config.data.batch_size == 64
    assert config.robustness.clean_probability == pytest.approx(0.35)


def test_main_uses_supported_preparation_and_training_cli_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(runner.os, "chdir", lambda _path: None)
    monkeypatch.setattr(
        runner,
        "repository_context",
        lambda _root: {"repository": "test", "branch": "test", "commit": "a" * 40},
    )
    monkeypatch.setattr(runner, "validate_cuda", lambda: {"gpu": "test"})
    monkeypatch.setattr(runner, "validate_genimage_attachment", lambda _root: {})
    monkeypatch.setattr(runner, "file_sha256", lambda _path: runner.V1_SHA256)
    monkeypatch.setattr(runner, "ensure_sid_subset", lambda _root: [])
    monkeypatch.setattr(runner, "validate_prepared_v2_data", lambda _root: {})
    monkeypatch.setattr(runner, "smoke_training_data", lambda _root: None)
    monkeypatch.setattr(runner, "enrich_v2_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_validate_metrics_tree", lambda _root: None)
    monkeypatch.setattr(runner, "package_export", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(
        runner,
        "run",
        lambda *args, **_kwargs: commands.append(tuple(str(arg) for arg in args)),
    )

    assert (
        runner.main(
            [
                "--input-root",
                str(tmp_path),
                "--output-root",
                str(tmp_path),
                "--license-confirmed",
            ]
        )
        == 0
    )

    prepare_command = next(
        command
        for command in commands
        if "scripts/prepare_genimage_v2.py" in command
    )
    assert "--confirm-license" in prepare_command
    assert "--per-generator" not in prepare_command
    from scripts import prepare_genimage_v2

    script_index = prepare_command.index("scripts/prepare_genimage_v2.py")
    parsed_prepare_args = prepare_genimage_v2.parse_args(
        list(prepare_command[script_index + 1 :])
    )
    assert parsed_prepare_args.confirm_license is True
    test_command = next(command for command in commands if "pytest" in command)
    assert commands.index(test_command) > commands.index(prepare_command)

    training_command = next(
        command for command in commands if "src.training.train" in command
    )
    assert training_command[-4:] == (
        "--config",
        str(runner.V2_CONFIG),
        "--device",
        "cuda",
    )
