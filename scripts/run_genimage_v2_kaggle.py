"""Run the guarded SID + Unbiased Tiny GenImage v2 Kaggle workflow.

This entrypoint deliberately keeps data acquisition, preparation, training,
evaluation, and packaging in fresh subprocesses.  It is intended to be called
by ``notebooks/train_genimage_v2_kaggle.ipynb`` after the read-only Kaggle
dataset input has been attached.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT_TEXT = str(REPOSITORY_ROOT)
if REPOSITORY_ROOT_TEXT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT_TEXT)

from scripts.prepare_sid_subset import DEFAULT_DATASET_REVISION


V1_CHECKPOINT = Path("artifacts/checkpoints/model.safetensors")
V1_SHA256 = "806fbabc5ecae8394369d08738cbf0c993568137d323a8133167e4557d04eed2"
V2_CHECKPOINT = Path("artifacts/checkpoints/model_v2.safetensors")
V2_METADATA = Path("artifacts/checkpoints/model_v2_metadata.json")
V2_HISTORY = Path("artifacts/metrics/genimage_v2_training_history.json")
SID_MANIFEST = Path("data/processed/manifest.csv")
SID_SUMMARY = Path("data/processed/manifest_summary.json")
GENIMAGE_MANIFEST = Path("data/processed/genimage_v2_manifest.csv")
GENIMAGE_SUMMARY = Path("data/processed/genimage_v2_manifest_summary.json")
COMBINED_MANIFEST = Path("data/processed/train_v2_manifest.csv")
V2_CONFIG = Path("configs/train_genimage_v2.yaml")
PUBLIC_SUMMARY = Path("artifacts/metrics/genimage_v2_summary.json")
PUBLIC_FIGURE = Path("artifacts/figures/genimage_v2_comparison.png")
PUBLIC_REPORT = Path("docs/submission/genimage-v2-report.md")
AUDIT_ROOT = Path("artifacts/metrics/genimage_v2_audit")

EXPECTED_SID_MANIFEST_SHA256 = (
    "6267a8d7e7749c1870601e196fe7ce1cc0fc2542a9975fa939832817e7fd3d9d"
)
EXPECTED_SID_COUNTS = {
    ("train", "0"): 2_400,
    ("train", "1"): 2_400,
    ("val", "0"): 300,
    ("val", "1"): 300,
    ("test", "0"): 300,
    ("test", "1"): 300,
}
EXPECTED_GENIMAGE_COUNTS = {
    ("train", "0"): 4_480,
    ("train", "1"): 4_480,
    ("val", "0"): 560,
    ("val", "1"): 560,
    ("test", "0"): 560,
    ("test", "1"): 560,
}
EXPECTED_COMBINED_COUNTS = {
    ("train", "0"): 6_880,
    ("train", "1"): 6_880,
    ("val", "0"): 560,
    ("val", "1"): 560,
    ("test", "0"): 560,
    ("test", "1"): 560,
}
EXPECTED_SCENARIOS = 20
EXPECTED_SID_TOTAL = 6_000
EXPECTED_GENIMAGE_TOTAL = 11_200
EXPECTED_COMBINED_TOTAL = 16_000
GENIMAGE_DATASET_ID = "cartografia/unbiased-tiny-genimage"
GENIMAGE_DATASET_VERSION = 1
GENIMAGE_METADATA_SHA256 = (
    "5f9a46e53e624339f6db8cc4d4a4fe5e54a0371e4b07a7da278300f6ed699e91"
)
GENIMAGE_INVENTORY_FILES = 23_329
GENIMAGE_INVENTORY_BYTES = 2_528_629_592
GENIMAGE_GENERATORS = (
    "ADM",
    "BigGAN",
    "Midjourney",
    "VQDM",
    "glide",
    "stable_diffusion_v_1_5",
    "wukong",
)
LICENSE_CONFIRMATION_TEXT = (
    "Participant confirmed organizer/licensor permission on 2026-08-31; "
    "private correspondence is not included in public artifacts."
)


def run(*args: object, cwd: Path = REPOSITORY_ROOT) -> None:
    """Run one visible fail-fast subprocess."""

    command = [str(arg) for arg in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=cwd)


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def validate_cuda() -> dict[str, Any]:
    """Require a real CUDA kernel before downloading or training."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available. In Kaggle Settings choose a T4 GPU, "
            "restart the session, and run all cells again."
        )
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(
            f"The locked Kaggle workflow requires a T4 GPU; found {gpu_name}. "
            "Change the accelerator in Kaggle Settings and restart the session."
        )
    capability = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = torch.cuda.get_arch_list()
    if required_arch not in compiled_arches:
        raise RuntimeError(
            f"PyTorch {torch.__version__} does not contain kernels for "
            f"{gpu_name} ({required_arch}); choose Kaggle T4 instead."
        )
    probe = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    checksum = float((probe @ probe.T).sum().item())
    torch.cuda.synchronize()
    print("CUDA preflight passed on", gpu_name, "checksum:", checksum)
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "torchvision": _installed_version("torchvision"),
        "pillow": _installed_version("Pillow"),
        "numpy": _installed_version("numpy"),
        "pandas": _installed_version("pandas"),
        "datasets": _installed_version("datasets"),
        "huggingface_hub": _installed_version("huggingface_hub"),
        "safetensors": _installed_version("safetensors"),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_name,
        "gpu_capability": list(capability),
        "required_cuda_arch": required_arch,
        "compiled_cuda_arches": compiled_arches,
        "cuda_probe_checksum": checksum,
    }


def validate_genimage_attachment(input_root: Path) -> dict[str, Any]:
    """Fail early unless the read-only Kaggle input is the pinned inventory."""

    from scripts.prepare_genimage_v2 import validate_inventory

    dataset_root, _sources, inventory = validate_inventory(input_root)
    print(
        "Pinned GenImage attachment passed:",
        dataset_root,
        f"({inventory['file_count']:,} files, {inventory['total_bytes']:,} bytes)",
    )
    return inventory


def validate_v2_training_config(root: Path) -> None:
    """Require every fixed warm-start setting and every isolated output path."""

    from src.training.config import load_config

    actual = load_config(root / V2_CONFIG).to_dict()
    expected = {
        "seed": 42,
        "data": {
            "manifest_path": COMBINED_MANIFEST.as_posix(),
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "image_size": 224,
            "batch_size": 64,
            "num_workers": 2,
        },
        "model": {
            "pretrained": False,
            "freeze_backbone": True,
            "unfreeze_last_blocks": 1,
        },
        "training": {
            "epochs": 3,
            "learning_rate": 3e-5,
            "weight_decay": 1e-4,
            "patience": 1,
            "mixed_precision": True,
        },
        "robustness": {"enabled": True, "clean_probability": 0.35},
        "output": {
            "checkpoint_path": V2_CHECKPOINT.as_posix(),
            "metadata_path": V2_METADATA.as_posix(),
            "history_path": V2_HISTORY.as_posix(),
        },
        "initialization": {
            "checkpoint_path": V1_CHECKPOINT.as_posix(),
            "expected_sha256": V1_SHA256,
            "freeze_frozen_batchnorm": True,
        },
    }
    if actual != expected:
        raise RuntimeError(
            "configs/train_genimage_v2.yaml differs from the locked v2 settings"
        )


def repository_context(root: Path) -> dict[str, str]:
    """Return reproducibility context and reject modified tracked source."""

    if not (root / ".git").is_dir():
        raise RuntimeError(f"Repository checkout not found: {root}")
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        text=True,
        cwd=root,
    ).strip()
    if tracked:
        raise RuntimeError(
            "Tracked files differ from the checked-out commit; start from the "
            "pushed repository version before running v2."
        )
    return {
        "repository": subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, cwd=root
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True, cwd=root
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=root
        ).strip(),
    }


def _read_manifest_counts(path: Path) -> tuple[list[dict[str, str]], Counter[tuple[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "split", "source_id", "sha256", "dataset"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Manifest schema is incomplete: {path}")
        rows = list(reader)
    return rows, Counter((row["split"], row["label"]) for row in rows)


def validate_sid_subset(root: Path) -> list[dict[str, str]]:
    """Require the exact v1 SID manifest and all 6,000 normalized images."""

    manifest = root / SID_MANIFEST
    summary_path = root / SID_SUMMARY
    if not manifest.is_file() or not summary_path.is_file():
        raise RuntimeError("SID manifest or summary is missing")
    if file_sha256(manifest) != EXPECTED_SID_MANIFEST_SHA256:
        raise RuntimeError(
            "Recreated SID manifest differs from the frozen v1 manifest; "
            "refusing to rehearse on a different sample."
        )
    rows, counts = _read_manifest_counts(manifest)
    if len(rows) != EXPECTED_SID_TOTAL or dict(counts) != EXPECTED_SID_COUNTS:
        raise RuntimeError(f"Unexpected SID split/class counts: {dict(counts)}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset_revision") != DEFAULT_DATASET_REVISION:
        raise RuntimeError("SID summary does not contain the pinned dataset revision")
    root_resolved = root.resolve()
    for row in rows:
        image_path = (root / row["path"]).resolve()
        try:
            image_path.relative_to(root_resolved)
        except ValueError as error:
            raise RuntimeError(f"SID path escapes the repository: {row['path']}") from error
        if not image_path.is_file() or file_sha256(image_path) != row["sha256"]:
            raise RuntimeError(f"SID image is missing or changed: {row['path']}")
    return rows


def ensure_sid_subset(root: Path) -> list[dict[str, str]]:
    """Reuse an exact SID subset or deterministically recreate it."""

    try:
        return validate_sid_subset(root)
    except (FileNotFoundError, RuntimeError) as error:
        print("SID subset is not reusable:", error)
    run(
        sys.executable,
        "scripts/prepare_sid_subset.py",
        "--total",
        "6000",
        "--seed",
        "42",
        "--revision",
        DEFAULT_DATASET_REVISION,
        "--shuffle-buffer",
        "512",
    )
    return validate_sid_subset(root)


def validate_prepared_v2_data(root: Path) -> dict[str, Any]:
    """Validate exact GenImage and combined-manifest counts and provenance."""

    genimage_rows, genimage_counts = _read_manifest_counts(root / GENIMAGE_MANIFEST)
    if (
        len(genimage_rows) != EXPECTED_GENIMAGE_TOTAL
        or dict(genimage_counts) != EXPECTED_GENIMAGE_COUNTS
    ):
        raise RuntimeError(
            f"Unexpected GenImage split/class counts: {dict(genimage_counts)}"
        )
    combined_rows, combined_counts = _read_manifest_counts(root / COMBINED_MANIFEST)
    if (
        len(combined_rows) != EXPECTED_COMBINED_TOTAL
        or dict(combined_counts) != EXPECTED_COMBINED_COUNTS
    ):
        raise RuntimeError(
            "Combined manifest must contain 13,760 train, 1,120 validation, "
            f"and 1,120 test rows; found {len(combined_rows)} rows and {dict(combined_counts)}"
        )
    genimage_hashes = {row["sha256"] for row in genimage_rows}
    sid_rows, _ = _read_manifest_counts(root / SID_MANIFEST)
    sid_hashes = {row["sha256"] for row in sid_rows}
    if len(genimage_hashes) != len(genimage_rows):
        raise RuntimeError("GenImage manifest contains duplicate normalized hashes")
    if genimage_hashes & sid_hashes:
        raise RuntimeError("GenImage and SID manifests contain overlapping image hashes")

    summary = json.loads((root / GENIMAGE_SUMMARY).read_text(encoding="utf-8"))
    expected_genimage_split_counts = {
        split: {
            label: EXPECTED_GENIMAGE_COUNTS.get((split, label), 0)
            for label in ("0", "1")
        }
        for split in ("train", "val", "test")
    }
    expected_combined_split_totals = {
        split: sum(
            count
            for (row_split, _label), count in EXPECTED_COMBINED_COUNTS.items()
            if row_split == split
        )
        for split in ("train", "val", "test")
    }
    expected_genimage_train = sum(
        count
        for (split, _label), count in EXPECTED_GENIMAGE_COUNTS.items()
        if split == "train"
    )
    expected_summary_values = {
        "dataset": GENIMAGE_DATASET_ID,
        "dataset_version": GENIMAGE_DATASET_VERSION,
        "seed": 42,
        "total": EXPECTED_GENIMAGE_TOTAL,
        "wildfake_used": False,
        "class_counts": {
            "0_real": sum(
                count
                for (_split, label), count in EXPECTED_GENIMAGE_COUNTS.items()
                if label == "0"
            ),
            "1_generated": sum(
                count
                for (_split, label), count in EXPECTED_GENIMAGE_COUNTS.items()
                if label == "1"
            ),
        },
        "split_counts": expected_genimage_split_counts,
        "combined_counts": {
            "total": EXPECTED_COMBINED_TOTAL,
            **expected_combined_split_totals,
            "sid_train_rows": expected_combined_split_totals["train"]
            - expected_genimage_train,
            "genimage_rows": EXPECTED_GENIMAGE_TOTAL,
        },
    }
    for key, expected in expected_summary_values.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"GenImage preparation summary has unexpected {key}")
    inventory = summary.get("inventory")
    if not isinstance(inventory, Mapping) or any(
        inventory.get(key) != expected
        for key, expected in {
            "file_count": GENIMAGE_INVENTORY_FILES,
            "total_bytes": GENIMAGE_INVENTORY_BYTES,
            "metadata_sha256": GENIMAGE_METADATA_SHA256,
            "generator_image_counts": {
                generator: 2_500 for generator in GENIMAGE_GENERATORS
            },
            "nature_image_count": 5_828,
        }.items()
    ):
        raise RuntimeError("GenImage preparation summary has unexpected inventory")
    license_confirmation = summary.get("license_confirmation")
    if not isinstance(license_confirmation, Mapping) or (
        license_confirmation.get("confirmed") is not True
    ):
        raise RuntimeError("GenImage preparation summary lacks licence confirmation")
    for key, path in (
        ("manifest_sha256", root / GENIMAGE_MANIFEST),
        ("combined_manifest_sha256", root / COMBINED_MANIFEST),
        ("sid_manifest_sha256", root / SID_MANIFEST),
    ):
        if summary.get(key) != file_sha256(path):
            raise RuntimeError(f"GenImage preparation summary has changed {key}")
    selected_digest = summary.get("selected_digest")
    if not isinstance(selected_digest, str) or len(selected_digest) != 64:
        raise RuntimeError("GenImage preparation summary lacks a dataset digest")
    return summary


def smoke_training_data(root: Path) -> None:
    """Load one combined batch and run a v1 forward/backward pass."""

    import torch
    from torch.nn import functional as torch_functional

    from src.data.dataset import ImageManifestDataset
    from src.models.checkpoints import load_checkpoint
    from src.models.efficientnet import build_model
    from src.training.train import TrainingImageTransform

    dataset = ImageManifestDataset(
        root / COMBINED_MANIFEST,
        "train",
        transform=TrainingImageTransform(224, robust=True, clean_probability=0.35),
        root_dir=root,
    )
    images = []
    labels = []
    for index in range(min(2, len(dataset))):
        image, label = dataset[index]
        images.append(image)
        labels.append(float(label))
    batch = torch.stack(images).to("cuda")
    targets = torch.tensor(labels, dtype=torch.float32, device="cuda")
    model = build_model(
        pretrained=False, freeze_backbone=True, unfreeze_last_blocks=1
    ).to("cuda")
    load_checkpoint(model, root / V1_CHECKPOINT, device="cuda")
    logits = model(batch).reshape(-1)
    loss = torch_functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    if not math.isfinite(float(loss.item())):
        raise RuntimeError("Training smoke check produced a non-finite loss")
    print("Training smoke check passed; loss:", float(loss.item()))


def _validate_metrics_tree(root: Path) -> None:
    expected_samples = {
        "v1_genimage": 1_120,
        "v2_genimage": 1_120,
        "v1_sid": 600,
        "v2_sid": 600,
    }
    for name, sample_count in expected_samples.items():
        path = root / AUDIT_ROOT / name / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, list) or len(scenarios) != EXPECTED_SCENARIOS:
            raise RuntimeError(f"{path} does not contain all 20 scenarios")
        for scenario in scenarios:
            if scenario.get("num_samples") != sample_count:
                raise RuntimeError(f"Unexpected sample count in {path}")
            for metric_name, value in scenario.get("metrics", {}).items():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise RuntimeError(
                        f"Non-finite metric {metric_name} in {path}"
                    )


def _safe_export_dir(output_root: Path) -> Path:
    resolved_root = output_root.resolve()
    if resolved_root == Path(resolved_root.anchor):
        raise RuntimeError("Refusing to use a filesystem root as output-root")
    export_dir = (resolved_root / "genimage_v2_export").resolve()
    if export_dir.parent != resolved_root:
        raise RuntimeError("Export directory escaped output-root")
    return export_dir


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def enrich_v2_metadata(
    root: Path,
    *,
    repository: Mapping[str, str],
    environment: Mapping[str, Any],
    license_confirmation: str,
) -> dict[str, Any]:
    """Attach review lineage to the separate v2 JSON metadata artifact."""

    metadata_path = root / V2_METADATA
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("v2 metadata must be a JSON object")
    v2_hash = file_sha256(root / V2_CHECKPOINT)
    if metadata.get("parent_checkpoint_sha256") != V1_SHA256:
        raise RuntimeError("v2 metadata does not identify the frozen v1 parent")
    if metadata.get("checkpoint_sha256") != v2_hash:
        raise RuntimeError("v2 metadata checkpoint hash does not match model_v2")
    provenance = json.loads((root / GENIMAGE_SUMMARY).read_text(encoding="utf-8"))
    inventory = provenance.get("inventory")
    if not isinstance(inventory, Mapping):
        raise RuntimeError("GenImage provenance lacks its source inventory")
    metadata.update(
        {
            "repository": dict(repository),
            "environment": dict(environment),
            "license_confirmation": {
                "confirmed_on": "2026-08-31",
                "statement": license_confirmation,
            },
            "dataset_lineage": {
                "sid_manifest_sha256": file_sha256(root / SID_MANIFEST),
                "genimage_manifest_sha256": file_sha256(root / GENIMAGE_MANIFEST),
                "combined_manifest_sha256": file_sha256(root / COMBINED_MANIFEST),
                "genimage_selected_digest": provenance.get("selected_digest"),
                "genimage_inventory_digest": inventory.get("inventory_digest"),
            },
            "wildfake_used": False,
            "automatic_promotion": False,
        }
    )
    _write_json(metadata_path, metadata)
    return metadata


def package_export(
    root: Path,
    output_root: Path,
    *,
    repository: Mapping[str, str],
    environment: Mapping[str, Any],
    license_confirmation: str,
) -> Path:
    """Build the local-audit and compact-public v2 export archive."""

    export_dir = _safe_export_dir(output_root)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    audit_files = {
        V2_CHECKPOINT: Path("local_audit/checkpoints/model_v2.safetensors"),
        V2_METADATA: Path(
            "local_audit/checkpoints/model_v2_metadata.json"
        ),
        V2_HISTORY: Path(
            "local_audit/training/genimage_v2_training_history.json"
        ),
        GENIMAGE_MANIFEST: Path("local_audit/manifests/genimage_v2_manifest.csv"),
        GENIMAGE_SUMMARY: Path(
            "local_audit/provenance/genimage_v2_manifest_summary.json"
        ),
        COMBINED_MANIFEST: Path("local_audit/manifests/train_v2_manifest.csv"),
        SID_MANIFEST: Path("local_audit/manifests/sid_manifest.csv"),
        SID_SUMMARY: Path("local_audit/provenance/sid_manifest_summary.json"),
        V2_CONFIG: Path("local_audit/config/train_genimage_v2.yaml"),
        Path("requirements.txt"): Path("local_audit/environment/requirements.txt"),
        Path("requirements-train.txt"): Path(
            "local_audit/environment/requirements-train.txt"
        ),
    }
    public_files = {
        PUBLIC_SUMMARY: Path("public/genimage_v2_summary.json"),
        PUBLIC_FIGURE: Path("public/genimage_v2_comparison.png"),
        PUBLIC_REPORT: Path("public/genimage-v2-report.md"),
    }
    for relative, destination_relative in {**audit_files, **public_files}.items():
        source = root / relative
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f"Required v2 artifact is missing or empty: {relative}")
        destination = export_dir / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    audit_source = root / AUDIT_ROOT
    if not audit_source.is_dir():
        raise RuntimeError("GenImage v2 audit directory is missing")
    shutil.copytree(audit_source, export_dir / "local_audit/evaluations")

    v1_hash = file_sha256(root / V1_CHECKPOINT)
    if v1_hash != V1_SHA256:
        raise RuntimeError("Frozen v1 checkpoint changed before packaging")
    genimage_provenance = json.loads(
        (root / GENIMAGE_SUMMARY).read_text(encoding="utf-8")
    )
    selected_digest = genimage_provenance.get("selected_digest")
    inventory = genimage_provenance.get("inventory")
    inventory_digest = (
        inventory.get("inventory_digest") if isinstance(inventory, Mapping) else None
    )
    if not isinstance(selected_digest, str) or len(selected_digest) != 64:
        raise RuntimeError("GenImage provenance lacks the selected dataset digest")
    if not isinstance(inventory_digest, str) or len(inventory_digest) != 64:
        raise RuntimeError("GenImage provenance lacks the source inventory digest")
    context = {
        "schema_version": 1,
        **dict(repository),
        **dict(environment),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_confirmation": license_confirmation,
        "license_confirmation_date": "2026-08-31",
        "threshold": 0.5,
        "seed": 42,
        "parent_checkpoint_sha256": v1_hash,
        "v2_checkpoint_sha256": file_sha256(root / V2_CHECKPOINT),
        "sid_manifest_sha256": file_sha256(root / SID_MANIFEST),
        "genimage_manifest_sha256": file_sha256(root / GENIMAGE_MANIFEST),
        "combined_manifest_sha256": file_sha256(root / COMBINED_MANIFEST),
        "genimage_selected_digest": selected_digest,
        "genimage_inventory_digest": inventory_digest,
        "public_summary_sha256": file_sha256(root / PUBLIC_SUMMARY),
        "public_report_sha256": file_sha256(root / PUBLIC_REPORT),
        "public_figure_sha256": file_sha256(root / PUBLIC_FIGURE),
        "wildfake_used": False,
        "automatic_promotion": False,
    }
    _write_json(export_dir / "local_audit/execution_metadata.json", context)
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    freeze_path = export_dir / "local_audit/environment/pip_freeze.txt"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(freeze, encoding="utf-8")
    (export_dir / "README.txt").write_text(
        "RealityCheck GenImage v2 review export\n\n"
        "public/ contains only sanitized aggregate evidence suitable for review.\n"
        "local_audit/ contains the candidate checkpoint, detailed manifests, "
        "per-image predictions, raw scenario metrics, hashes, and environment.\n"
        "The candidate is not deployed automatically; v1 remains the model of record.\n",
        encoding="utf-8",
    )

    archive_base = output_root.resolve() / "genimage_v2_export"
    archive = Path(shutil.make_archive(str(archive_base), "zip", export_dir))
    print("Validated GenImage v2 export ready:", archive)
    return archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, warm-start, evaluate, and export GenImage v2 on Kaggle."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Attached read-only cartografia/unbiased-tiny-genimage root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working"),
        help="Directory receiving genimage_v2_export.zip.",
    )
    parser.add_argument(
        "--license-confirmed",
        action="store_true",
        help="Required acknowledgement that dataset use was confirmed for this event.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest only after it passed in this same Kaggle session.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.license_confirmed:
        raise RuntimeError(
            "Training is blocked until --license-confirmed is supplied after "
            "the participant verifies permission for this hackathon."
        )
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise RuntimeError(
            f"Attached dataset not found at {input_root}. In Kaggle choose Add Input "
            "and attach cartografia/unbiased-tiny-genimage."
        )

    os.chdir(REPOSITORY_ROOT)
    repository = repository_context(REPOSITORY_ROOT)
    environment = validate_cuda()
    validate_genimage_attachment(input_root)
    validate_v2_training_config(REPOSITORY_ROOT)
    if file_sha256(REPOSITORY_ROOT / V1_CHECKPOINT) != V1_SHA256:
        raise RuntimeError("Frozen v1 checkpoint hash mismatch before v2 preparation")

    ensure_sid_subset(REPOSITORY_ROOT)
    run(
        sys.executable,
        "scripts/prepare_genimage_v2.py",
        "--input-root",
        input_root,
        "--sid-manifest",
        SID_MANIFEST,
        "--output-root",
        REPOSITORY_ROOT,
        "--seed",
        "42",
        "--confirm-license",
    )
    validate_prepared_v2_data(REPOSITORY_ROOT)
    if not args.skip_tests:
        run(sys.executable, "-m", "pytest", "-q")
    smoke_training_data(REPOSITORY_ROOT)
    run(
        sys.executable,
        "-m",
        "src.training.train",
        "--config",
        V2_CONFIG,
        "--device",
        "cuda",
    )
    enrich_v2_metadata(
        REPOSITORY_ROOT,
        repository=repository,
        environment=environment,
        license_confirmation=LICENSE_CONFIRMATION_TEXT,
    )
    if file_sha256(REPOSITORY_ROOT / V1_CHECKPOINT) != V1_SHA256:
        raise RuntimeError("Frozen v1 checkpoint changed during v2 training")
    run(
        sys.executable,
        "scripts/evaluate_genimage_v2.py",
        "--genimage-manifest",
        GENIMAGE_MANIFEST,
        "--sid-manifest",
        SID_MANIFEST,
        "--v1-checkpoint",
        V1_CHECKPOINT,
        "--v2-checkpoint",
        V2_CHECKPOINT,
        "--device",
        "cuda",
    )
    _validate_metrics_tree(REPOSITORY_ROOT)
    package_export(
        REPOSITORY_ROOT,
        args.output_root,
        repository=repository,
        environment=environment,
        license_confirmation=LICENSE_CONFIRMATION_TEXT,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
