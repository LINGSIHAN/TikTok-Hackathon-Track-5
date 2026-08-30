import csv
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluate import (
    Scenario,
    _manifest_paths,
    _write_json,
    _write_predictions,
    build_scenarios,
    predict_loader,
    summarize_scenarios,
)


class IdentityLogitModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs[:, :1]


def test_build_scenarios_is_clean_first_and_stable():
    scenarios = build_scenarios({"jpeg": [90, 50], "blur": [0.5]})

    assert scenarios == [
        Scenario("clean", None),
        Scenario("jpeg", 90),
        Scenario("jpeg", 50),
        Scenario("blur", 0.5),
    ]


def test_predict_loader_uses_manifest_fallback_paths():
    loader = DataLoader(
        TensorDataset(torch.tensor([[-2.0], [2.0]]), torch.tensor([0, 1])),
        batch_size=2,
    )

    labels, probabilities, paths = predict_loader(
        IdentityLogitModel(),
        loader,
        device=torch.device("cpu"),
        fallback_paths=["real.jpg", "fake.jpg"],
    )

    assert labels == [0, 1]
    assert probabilities == pytest.approx([0.1192029, 0.8807971])
    assert paths == ["real.jpg", "fake.jpg"]


def test_summary_reports_worst_transformation_and_drop():
    results = [
        {
            "transform": "clean",
            "severity": None,
            "metrics": {"roc_auc": 0.9, "brier_score": 0.1},
        },
        {
            "transform": "jpeg",
            "severity": 30,
            "metrics": {"roc_auc": 0.7, "brier_score": 0.2},
        },
        {
            "transform": "blur",
            "severity": 2.0,
            "metrics": {"roc_auc": 0.6, "brier_score": 0.3},
        },
    ]

    summary = summarize_scenarios(results)

    assert summary["mean_transformed"]["roc_auc"] == pytest.approx(0.65)
    assert summary["worst_case"]["transform"] == "blur"
    assert summary["clean_to_worst_roc_auc_drop"] == pytest.approx(0.3)


def test_manifest_filter_and_artifact_writers(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,label,split\ntrain.jpg,0,train\ntest.jpg,1,test\n",
        encoding="utf-8",
    )
    assert _manifest_paths(manifest, "test") == ["test.jpg"]

    predictions_path = tmp_path / "predictions.csv"
    _write_predictions(
        predictions_path,
        [
            {
                "image_path": "test.jpg",
                "label": 1,
                "transform": "clean",
                "severity": "",
                "pred": 0.8,
            }
        ],
    )
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["image_path"] == "test.jpg"

    metrics_path = tmp_path / "metrics.json"
    _write_json(metrics_path, {"undefined": float("nan")})
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {"undefined": None}
