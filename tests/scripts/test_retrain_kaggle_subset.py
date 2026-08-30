from __future__ import annotations

import json
import subprocess
import zipfile

import pytest

from scripts.retrain_kaggle_subset import package_export, validate_worktree_clean
from src.data.preprocessing import PREPROCESSING_CONTRACT_ID


def test_package_export_records_canonical_preprocessing_contract(tmp_path) -> None:
    root = tmp_path / "repository"
    files = [
        "artifacts/checkpoints/model.safetensors",
        "artifacts/checkpoints/model_metadata.json",
        "artifacts/runs/clean/model.safetensors",
        "artifacts/runs/clean/model_metadata.json",
        "artifacts/runs/clean/history.json",
        "artifacts/metrics/training_history.json",
        "artifacts/metrics/metrics.json",
        "artifacts/metrics/predictions.csv",
        "artifacts/metrics/robustness.png",
        "artifacts/metrics/clean_baseline/metrics.json",
        "artifacts/metrics/clean_baseline/predictions.csv",
        "artifacts/metrics/clean_baseline/robustness.png",
        "data/processed/manifest.csv",
        "data/processed/manifest_summary.json",
        "configs/train_clean.yaml",
        "configs/train_robust.yaml",
        "requirements.txt",
        "requirements-train.txt",
    ]
    for relative in files:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("test\n", encoding="utf-8")
    (root / "data/processed/manifest_summary.json").write_text(
        "{}\n", encoding="utf-8"
    )
    secret = root / "artifacts/metrics/debug-secret.txt"
    secret.write_text("do not package\n", encoding="utf-8")

    archive = package_export(
        root,
        tmp_path / "output",
        {
            "repository": "https://example.test/repository.git",
            "branch": "master",
            "commit": "a" * 40,
        },
        {
            "pytorch": "test",
            "required_cuda_arch": "sm_75",
            "cuda_probe_checksum": 1.0,
        },
    )

    assert archive.is_file()
    with zipfile.ZipFile(archive) as bundle:
        context = json.loads(bundle.read("artifacts/metrics/run_context.json"))
        names = set(bundle.namelist())
    assert context["preprocessing_contract"] == PREPROCESSING_CONTRACT_ID
    assert context["dataset_revision"] is None
    assert context["dataset_revision_status"] == "legacy_unrecorded"
    assert "artifacts/metrics/debug-secret.txt" not in names


def test_worktree_allows_generated_manifests_but_rejects_untracked_code(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / ".gitignore").write_text(
        "data/raw/**\nartifacts/runs/**\n", encoding="utf-8"
    )
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    for relative in (
        "data/processed/manifest.csv",
        "data/processed/manifest_summary.json",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")

    validate_worktree_clean(repository)

    (repository / "unexpected.py").write_text("print('not committed')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected.py"):
        validate_worktree_clean(repository)
