import pytest

from scripts.generate_submission_evidence import confusion_counts


def test_confusion_counts_covers_each_outcome() -> None:
    rows = [
        {"label": "0", "pred": "0.1"},
        {"label": "0", "pred": "0.9"},
        {"label": "1", "pred": "0.1"},
        {"label": "1", "pred": "0.9"},
    ]

    assert confusion_counts(rows, 0.5) == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_confusion_counts_rejects_boundary_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        confusion_counts([{"label": "0", "pred": "0.1"}], threshold)


def test_confusion_counts_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="invalid probability"):
        confusion_counts([{"label": "1", "pred": "1.5"}], 0.5)
