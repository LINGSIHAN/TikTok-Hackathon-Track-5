"""Validate and evaluate the locked WildFake demonstration-only benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT_TEXT = str(REPOSITORY_ROOT)
if REPOSITORY_ROOT_TEXT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT_TEXT)

from src.evaluation.evaluate import evaluate_checkpoint
from src.evaluation.plotting import save_robustness_plot


WILDFAKE_REPOSITORY = "https://www.modelscope.cn/datasets/hy2628982280/WildFake"
WILDFAKE_REVISION = "18f53ff36ad9da60644039f0452b0e7b3907af6f"
EXPECTED_COUNTS = {"real": 4_998, "generated": 8_843}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
FIXED_THRESHOLD = 0.50
SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
DISCLAIMER = (
    "WildFake was evaluated only after the model checkpoint and 0.50 threshold "
    "were locked. It is demonstration-only and was never used for training, "
    "threshold selection, or model selection."
)


@dataclass(frozen=True)
class ValidatedSample:
    path: str
    label: int
    sha256: str
    size: int
    source: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _decode_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise OSError("image has invalid dimensions")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unreadable image in WildFake subset: {path}") from error


def _class_files(class_root: Path) -> list[Path]:
    if not class_root.is_dir():
        raise RuntimeError(f"Missing WildFake class directory: {class_root}")
    files = sorted(path for path in class_root.rglob("*") if path.is_file())
    unexpected = [
        path for path in files if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected non-image file in class directory: {unexpected[0]}")
    return files


def dataset_digest(samples: Iterable[ValidatedSample]) -> str:
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda value: value.path):
        digest.update(sample.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_dataset(
    dataset_root: Path,
    *,
    duplicate_groups: list[dict[str, Any]] | None = None,
) -> tuple[list[ValidatedSample], str]:
    dataset_root = dataset_root.resolve()
    samples: list[ValidatedSample] = []
    hash_labels: dict[str, tuple[int, list[str]]] = {}
    for class_name, label in (("real", 0), ("generated", 1)):
        files = _class_files(dataset_root / class_name)
        expected_count = EXPECTED_COUNTS[class_name]
        if len(files) != expected_count:
            raise RuntimeError(
                f"WildFake {class_name} count is {len(files):,}; expected exactly "
                f"{expected_count:,}"
            )
        for index, path in enumerate(files, start=1):
            _decode_image(path)
            digest = sha256_file(path)
            previous = hash_labels.get(digest)
            relative = path.relative_to(dataset_root).as_posix()
            if previous is not None:
                previous_label, previous_paths = previous
                if previous_label != label:
                    raise RuntimeError(
                        "Conflicting labels for identical WildFake image bytes: "
                        f"{previous_paths[0]} and {relative}"
                    )
                previous_paths.append(relative)
            else:
                hash_labels[digest] = (label, [relative])
            samples.append(
                ValidatedSample(
                    path=relative,
                    label=label,
                    sha256=digest,
                    size=path.stat().st_size,
                    source=("COCO val2017" if label == 0 else "Advanced DALL-E 3"),
                )
            )
            if index % 1_000 == 0 or index == len(files):
                print(
                    f"  validated {index:,}/{len(files):,} {class_name} images",
                    flush=True,
                )
    if len(samples) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"WildFake total is {len(samples):,}; expected exactly {EXPECTED_TOTAL:,}"
        )
    samples.sort(key=lambda sample: sample.path)
    if duplicate_groups is not None:
        duplicate_groups.extend(
            {
                "sha256": digest,
                "label": label,
                "paths": paths,
            }
            for digest, (label, paths) in sorted(hash_labels.items())
            if len(paths) > 1
        )
    return samples, dataset_digest(samples)


def verify_download_manifest(dataset_root: Path, digest: str) -> dict[str, Any]:
    manifest_path = dataset_root / "download_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "Missing download_manifest.json; run scripts/download_wildfake_demo.py first"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("revision") != WILDFAKE_REVISION:
        raise RuntimeError("WildFake download manifest revision is not the locked revision")
    if payload.get("dataset_digest") != digest:
        raise RuntimeError(
            "WildFake files do not match the verified aggregate download digest"
        )
    if payload.get("total_count") != EXPECTED_TOTAL:
        raise RuntimeError("WildFake download manifest has an unexpected total count")
    if payload.get("class_counts") != EXPECTED_COUNTS:
        raise RuntimeError("WildFake download manifest has unexpected class counts")
    return payload


def write_evaluation_manifest(path: Path, samples: Sequence[ValidatedSample]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [
        "path",
        "label",
        "split",
        "content_sha256",
        "source_id",
        "dataset",
        "dataset_revision",
        "provenance",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in sorted(samples, key=lambda value: value.path):
            writer.writerow(
                {
                    "path": sample.path,
                    "label": sample.label,
                    "split": "test",
                    "content_sha256": sample.sha256,
                    "source_id": f"wildfake:{sample.path}",
                    "dataset": "WildFake",
                    "dataset_revision": WILDFAKE_REVISION,
                    "provenance": sample.source,
                }
            )
    temporary.replace(path)
    return sha256_file(path)


def verify_frozen_checkpoint(checkpoint: Path, run_context: Path) -> str:
    if not checkpoint.is_file():
        raise RuntimeError(f"Frozen checkpoint not found: {checkpoint}")
    if not run_context.is_file():
        raise RuntimeError(f"Run context not found: {run_context}")
    context = json.loads(run_context.read_text(encoding="utf-8"))
    expected = context.get("checkpoint_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError("run_context.json has no valid frozen checkpoint_sha256")
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise RuntimeError(
            f"Frozen checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def confusion_summary(
    rows: Sequence[dict[str, str]], *, threshold: float
) -> dict[str, Any]:
    counts = {"true_negative": 0, "false_positive": 0, "false_negative": 0, "true_positive": 0}
    for row in rows:
        if row["transform"] != "clean":
            continue
        label = int(row["label"])
        probability = float(row["pred"])
        predicted = int(probability >= threshold)
        if label == 0 and predicted == 0:
            counts["true_negative"] += 1
        elif label == 0:
            counts["false_positive"] += 1
        elif predicted == 0:
            counts["false_negative"] += 1
        else:
            counts["true_positive"] += 1
    real_count = counts["true_negative"] + counts["false_positive"]
    generated_count = counts["false_negative"] + counts["true_positive"]
    if (real_count, generated_count) != (
        EXPECTED_COUNTS["real"],
        EXPECTED_COUNTS["generated"],
    ):
        raise RuntimeError("Clean predictions do not contain the exact locked class counts")
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


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_external_device(requested: str) -> str:
    """Prefer available GPU acceleration for this large external benchmark."""

    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_public_summary(
    *,
    evaluation: dict[str, Any],
    checkpoint_sha256: str,
    digest: str,
    manifest_sha256: str,
    confusion: dict[str, Any],
    duplicate_groups: Sequence[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    clean = evaluation["scenarios"][0]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "scope": "post-lock demonstration-only external evaluation",
        "disclaimer": DISCLAIMER,
        "dataset": {
            "name": "WildFake",
            "repository": WILDFAKE_REPOSITORY,
            "revision": WILDFAKE_REVISION,
            "dataset_digest": digest,
            "evaluation_manifest_sha256": manifest_sha256,
            "total_count": EXPECTED_TOTAL,
            "class_counts": EXPECTED_COUNTS,
            "class_provenance": {
                "real": "COCO val2017",
                "generated": "Advanced DALL-E 3 (IsAdvanced=1, IsFake=1)",
            },
            "duplicate_content": {
                "groups": len(duplicate_groups),
                "additional_files": sum(
                    len(group["paths"]) - 1 for group in duplicate_groups
                ),
                "unique_content_hashes": EXPECTED_TOTAL
                - sum(len(group["paths"]) - 1 for group in duplicate_groups),
                "conflicting_label_groups": 0,
                "policy": (
                    "same-label duplicates retained to preserve the exact organizer subset"
                ),
            },
        },
        "model": {
            "checkpoint_sha256": checkpoint_sha256,
            "decision_threshold": FIXED_THRESHOLD,
        },
        "scenario_mode": mode,
        "scenario_count": len(evaluation["scenarios"]),
        "clean": {
            "num_samples": clean["num_samples"],
            "metrics": clean["metrics"],
            "confusion": confusion,
        },
    }
    if mode == "full":
        summary["robustness"] = {
            "mean_transformed": evaluation["summary"]["mean_transformed"],
            "worst_case": evaluation["summary"]["worst_case"],
            "clean_to_worst_roc_auc_drop": evaluation["summary"][
                "clean_to_worst_roc_auc_drop"
            ],
        }
    return summary


def _format_metric(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.4f}"


def write_public_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["clean"]["metrics"]
    confusion = summary["clean"]["confusion"]
    class_normalized = confusion["class_normalized"]
    lines = [
        "# WildFake Demonstration Evaluation",
        "",
        f"> **Scope:** {summary['disclaimer']}",
        "",
        "This external benchmark measures the frozen RealityCheck detector on the exact "
        "organizer subset. It is evidence of behavior on this subset, not proof of image "
        "provenance or a claim of universal detector performance.",
        "",
        "## Locked evaluation identity",
        "",
        f"- WildFake revision: `{summary['dataset']['revision']}`",
        f"- Dataset digest: `{summary['dataset']['dataset_digest']}`",
        f"- Checkpoint SHA-256: `{summary['model']['checkpoint_sha256']}`",
        f"- Fixed threshold: `{summary['model']['decision_threshold']:.2f}`",
        f"- Scenario mode: `{summary['scenario_mode']}`",
        f"- Images: {summary['dataset']['total_count']:,} total "
        f"({summary['dataset']['class_counts']['real']:,} COCO val2017 real; "
        f"{summary['dataset']['class_counts']['generated']:,} Advanced DALL-E 3 generated)",
        f"- Duplicate-content audit: "
        f"{summary['dataset']['duplicate_content']['groups']:,} same-label group(s), "
        f"{summary['dataset']['duplicate_content']['additional_files']:,} additional "
        f"file(s), and {summary['dataset']['duplicate_content']['unique_content_hashes']:,} "
        "unique byte hashes; duplicates retained to preserve the exact organizer subset; "
        "no conflicting labels",
        "",
        "## Clean results",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    labels = (
        ("ROC-AUC", "roc_auc"),
        ("Average precision", "average_precision"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("F1", "f1"),
        ("False-positive rate", "false_positive_rate"),
        ("False-negative rate", "false_negative_rate"),
        ("Brier score", "brier_score"),
    )
    lines.extend(f"| {label} | {_format_metric(metrics[key])} |" for label, key in labels)
    raw = confusion["raw"]
    lines.extend(
        [
            "",
            "## Confusion counts at 0.50",
            "",
            "| Actual class | Predicted real | Predicted generated |",
            "|---|---:|---:|",
            f"| Real | {raw['true_negative']:,} "
            f"({class_normalized['real']['predicted_real']:.4f}) | "
            f"{raw['false_positive']:,} "
            f"({class_normalized['real']['predicted_generated']:.4f}) |",
            f"| Generated | {raw['false_negative']:,} "
            f"({class_normalized['generated']['predicted_real']:.4f}) | "
            f"{raw['true_positive']:,} "
            f"({class_normalized['generated']['predicted_generated']:.4f}) |",
        ]
    )
    if summary["scenario_mode"] == "full":
        robustness = summary["robustness"]
        worst = robustness["worst_case"]
        lines.extend(
            [
                "",
                "## Transformation robustness",
                "",
                f"- Mean transformed ROC-AUC: "
                f"{_format_metric(robustness['mean_transformed']['roc_auc'])}",
                f"- Worst scenario: `{worst['transform']}:{worst['severity']}` "
                f"with ROC-AUC {_format_metric(worst['value'])}",
                f"- Clean-to-worst ROC-AUC degradation: "
                f"{_format_metric(robustness['clean_to_worst_roc_auc_drop'])}",
            ]
        )
    lines.extend(
        [
            "",
            "Metrics weight every organizer row, including the disclosed same-label "
            "duplicate-content rows. They should not be interpreted as an estimate from "
            f"{summary['dataset']['total_count']:,} independent images.",
            "",
            "The compact JSON and figure beside this report contain only aggregate "
            "evidence. Raw predictions, the deterministic sample manifest, execution "
            "metadata, and all source images remain local and ignored by Git.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def save_clean_public_figure(summary: dict[str, Any], path: Path) -> None:
    """Render an aggregate-only clean benchmark figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = summary["clean"]["metrics"]
    labels = [
        "ROC-AUC",
        "Avg precision",
        "Balanced acc.",
        "F1",
        "True-negative rate",
        "True-positive rate",
        "Brier (lower better)",
    ]
    values = [
        metrics["roc_auc"],
        metrics["average_precision"],
        metrics["balanced_accuracy"],
        metrics["f1"],
        1.0 - metrics["false_positive_rate"],
        1.0 - metrics["false_negative_rate"],
        metrics["brier_score"],
    ]
    colors = ["#2563eb"] * 6 + ["#f59e0b"]
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    bars = axis.bar(range(len(values)), values, color=colors, width=0.72)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Metric value")
    axis.set_title("WildFake clean demonstration metrics · fixed threshold 0.50")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    duplicate_content = summary["dataset"]["duplicate_content"]
    figure.text(
        0.5,
        0.01,
        f"{summary['dataset']['total_count']:,} organizer rows · "
        f"{duplicate_content['unique_content_hashes']:,} unique byte hashes · "
        "post-lock demonstration only",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_evaluation(
    *,
    dataset_root: Path,
    checkpoint: Path,
    run_context: Path,
    mode: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict[str, Any]:
    if mode not in {"clean", "full"}:
        raise ValueError("mode must be either 'clean' or 'full'")
    started = time.monotonic()
    dataset_root = dataset_root.resolve()
    checkpoint = checkpoint.resolve()
    run_context = run_context.resolve()
    print("Validating exact WildFake class counts, decodes, and hashes...", flush=True)
    duplicate_groups: list[dict[str, Any]] = []
    samples, digest = validate_dataset(
        dataset_root, duplicate_groups=duplicate_groups
    )
    verify_download_manifest(dataset_root, digest)
    checkpoint_sha256 = verify_frozen_checkpoint(checkpoint, run_context)
    manifest_path = dataset_root / "evaluation_manifest.csv"
    manifest_sha256 = write_evaluation_manifest(manifest_path, samples)
    resolved_device = resolve_external_device(device)

    audit_root = REPOSITORY_ROOT / "artifacts/runs/wildfake_demo" / mode
    print(
        f"Evaluating frozen checkpoint in {mode} mode at threshold "
        f"{FIXED_THRESHOLD:.2f} on {resolved_device}...",
        flush=True,
    )
    evaluation = evaluate_checkpoint(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint,
        split="test",
        output_dir=audit_root,
        root_dir=dataset_root,
        scenario_mode=mode,
        batch_size=batch_size,
        num_workers=num_workers,
        device_name=resolved_device,
        image_size=224,
        threshold=FIXED_THRESHOLD,
        seed=42,
    )
    predictions = _read_predictions(audit_root / "predictions.csv")
    confusion = confusion_summary(predictions, threshold=FIXED_THRESHOLD)
    public_summary = build_public_summary(
        evaluation=evaluation,
        checkpoint_sha256=checkpoint_sha256,
        digest=digest,
        manifest_sha256=manifest_sha256,
        confusion=confusion,
        duplicate_groups=duplicate_groups,
        mode=mode,
    )
    public_json = REPOSITORY_ROOT / "artifacts/metrics/wildfake_demo_summary.json"
    public_figure = REPOSITORY_ROOT / "artifacts/figures/wildfake_demo.png"
    public_report = REPOSITORY_ROOT / "docs/submission/wildfake-demo-report.md"
    _write_json(public_json, public_summary)
    if mode == "clean":
        save_clean_public_figure(public_summary, public_figure)
    else:
        save_robustness_plot(evaluation["scenarios"], public_figure)
    write_public_report(public_report, public_summary)

    execution = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "mode": mode,
        "threshold": FIXED_THRESHOLD,
        "device": resolved_device,
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint),
        "run_context_path": str(run_context),
        "dataset_digest": digest,
        "checkpoint_sha256": checkpoint_sha256,
        "duplicate_content_groups": duplicate_groups,
        "python": platform.python_version(),
        "packages": {
            name: _version(name)
            for name in (
                "numpy",
                "pandas",
                "Pillow",
                "safetensors",
                "torch",
                "torchvision",
            )
        },
        "public_artifacts": {
            "summary": str(public_json),
            "figure": str(public_figure),
            "report": str(public_report),
        },
    }
    _write_json(audit_root / "execution_metadata.json", execution)
    print(
        f"External benchmark complete. Public summary: "
        f"{public_json.relative_to(REPOSITORY_ROOT)}",
        flush=True,
    )
    return public_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen detector on the exact WildFake demo subset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPOSITORY_ROOT / "data/external/wildfake_demo",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/checkpoints/model.safetensors",
    )
    parser.add_argument(
        "--run-context",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/metrics/run_context.json",
    )
    parser.add_argument("--mode", choices=("clean", "full"), default="clean")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_evaluation(
            dataset_root=args.dataset_root,
            checkpoint=args.checkpoint,
            run_context=args.run_context,
            mode=args.mode,
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
