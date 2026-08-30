"""Pure presentation calculations for the Streamlit demo.

Keeping these functions independent of Streamlit makes the model's displayed
interpretation and robustness summary straightforward to unit test.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


DEFAULT_THRESHOLD = 0.5
DEFAULT_UNCERTAINTY_MARGIN = 0.1


@dataclass(frozen=True)
class ScoreInterpretation:
    """Plain-language interpretation of one model probability."""

    label: str
    uncertainty: str
    explanation: str
    generated_side: bool | None


@dataclass(frozen=True)
class StressResult:
    """Validated form of one transformed-image prediction."""

    transform: str
    severity: str
    probability: float

    @property
    def display_name(self) -> str:
        return format_transform_name(self.transform, self.severity)


@dataclass(frozen=True)
class RobustnessSummary:
    """Compact robustness-passport statistics."""

    case_count: int
    consistent_count: int
    family_count: int
    label_stability: float
    score_consistency: float
    average_probability: float
    mean_absolute_shift: float
    minimum_probability: float
    maximum_probability: float
    largest_shift: float
    largest_shift_delta: float
    largest_shift_transform: str
    largest_shift_name: str
    largest_shift_severity: str
    largest_shift_probability: float
    largest_shift_flipped: bool


def validate_probability(value: Any, *, field: str = "probability") -> float:
    """Return ``value`` as a finite probability or raise a useful error."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a number between 0 and 1")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be a finite number between 0 and 1")
    return probability


def interpret_probability(
    probability: float,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> ScoreInterpretation:
    """Describe a score without presenting it as a forensic verdict."""

    score = validate_probability(probability)
    boundary = validate_probability(threshold, field="threshold")
    if boundary in (0.0, 1.0):
        raise ValueError("threshold must be strictly between 0 and 1")
    if not math.isfinite(uncertainty_margin) or uncertainty_margin < 0:
        raise ValueError("uncertainty_margin must be finite and non-negative")
    if uncertainty_margin > min(boundary, 1.0 - boundary):
        raise ValueError("uncertainty_margin extends beyond the probability range")

    distance = abs(score - boundary)
    if distance <= uncertainty_margin:
        return ScoreInterpretation(
            label="Inconclusive",
            uncertainty="High uncertainty",
            explanation=(
                "The score is close to the decision boundary, so small image "
                "changes could alter the label."
            ),
            generated_side=None,
        )

    generated_side = score > boundary
    label = "Likely AI-generated" if generated_side else "Likely authentic"
    if distance < 0.25:
        uncertainty = "Moderate uncertainty"
        strength = "leans toward"
    else:
        uncertainty = "Lower relative uncertainty"
        strength = "more strongly favors"
    subject = "AI-generated" if generated_side else "authentic"
    return ScoreInterpretation(
        label=label,
        uncertainty=uncertainty,
        explanation=(
            f"The model {strength} the {subject} side of its decision boundary. "
            "This is supporting evidence, not proof of origin."
        ),
        generated_side=generated_side,
    )


def format_transform_name(transform: str, severity: str = "") -> str:
    """Turn an internal transform name into a concise display label."""

    cleaned = str(transform).strip().replace("_", " ")
    if not cleaned:
        cleaned = "Unknown transform"
    label = cleaned.capitalize()
    label = label.replace("Jpeg", "JPEG")
    severity_text = str(severity).strip()
    if not severity_text:
        return label

    normalized = cleaned.casefold()
    if "jpeg" in normalized:
        detail = f"quality {severity_text}"
    elif "blur" in normalized or "noise" in normalized:
        detail = f"σ={severity_text}"
    elif "resize" in normalized:
        detail = f"scale {severity_text}×"
    elif "crop" in normalized:
        try:
            retain = float(severity_text)
            detail = f"retain {retain:.0%}" if 0.0 < retain <= 1.0 else severity_text
        except ValueError:
            detail = severity_text
    elif normalized == "color jitter":
        try:
            amount = float(severity_text)
            detail = f"±{amount:.0%}" if 0.0 < amount <= 1.0 else severity_text
        except ValueError:
            detail = severity_text
    else:
        detail = severity_text
    return f"{label} · {detail}"


def normalize_stress_results(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: Sequence[tuple[str, str]] | None = None,
) -> list[StressResult]:
    """Validate Predictor.stress_test output for display calculations."""

    normalized: list[StressResult] = []
    seen: set[tuple[str, str]] = set()
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise TypeError(f"stress result {index} must be a mapping")
        if "transform" not in result or "severity" not in result or "pred" not in result:
            raise ValueError(
                f"stress result {index} must contain transform, severity, and pred"
            )
        if not isinstance(result["transform"], str):
            raise TypeError(f"stress result {index} transform must be a string")
        if not isinstance(result["severity"], str):
            raise TypeError(f"stress result {index} severity must be a string")
        transform = result["transform"].strip()
        if not transform:
            raise ValueError(f"stress result {index} has an empty transform name")
        if transform.casefold() == "clean":
            raise ValueError("stress results must not include the clean baseline")
        severity = result["severity"].strip()
        if not severity:
            raise ValueError(f"stress result {index} has an empty severity")
        descriptor = (transform.casefold(), severity.casefold())
        if descriptor in seen:
            raise ValueError(
                f"stress result {index} duplicates transform/severity {descriptor!r}"
            )
        seen.add(descriptor)
        normalized.append(
            StressResult(
                transform=transform,
                severity=severity,
                probability=validate_probability(
                    result["pred"], field=f"stress result {index} pred"
                ),
            )
        )
    if expected_pairs is not None:
        expected = [
            (str(transform).strip().casefold(), str(severity).strip().casefold())
            for transform, severity in expected_pairs
        ]
        actual = [
            (result.transform.casefold(), result.severity.casefold())
            for result in normalized
        ]
        if actual != expected:
            raise ValueError(
                "stress results do not match the configured transform grid"
            )
    return normalized


def _transform_family(transform: str) -> str:
    """Map related colour operations to one equally weighted family."""

    normalized = transform.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"brightness", "contrast", "saturation", "color_jitter"}:
        return "color_jitter"
    if normalized.startswith(("brightness_", "contrast_", "saturation_")):
        return "color_jitter"
    return normalized


def summarize_robustness(
    base_probability: float,
    results: Sequence[Mapping[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    expected_pairs: Sequence[tuple[str, str]] | None = None,
) -> RobustnessSummary:
    """Build a robustness passport from clean and transformed predictions.

    Summary averages are balanced across transform families, rather than raw
    row counts, so a family with extra severity levels cannot dominate.
    """

    base = validate_probability(base_probability, field="base probability")
    boundary = validate_probability(threshold, field="threshold")
    if boundary in (0.0, 1.0):
        raise ValueError("threshold must be strictly between 0 and 1")
    normalized = normalize_stress_results(results, expected_pairs=expected_pairs)
    if not normalized:
        raise ValueError("at least one stress-test result is required")

    base_generated = base >= boundary
    consistent_count = sum(
        (result.probability >= boundary) == base_generated for result in normalized
    )
    largest = max(normalized, key=lambda result: abs(result.probability - base))

    families: OrderedDict[str, list[StressResult]] = OrderedDict()
    for result in normalized:
        families.setdefault(_transform_family(result.transform), []).append(result)

    family_mean_probabilities: list[float] = []
    family_mean_shifts: list[float] = []
    family_label_agreements: list[float] = []
    for family_results in families.values():
        family_mean_probabilities.append(
            sum(item.probability for item in family_results) / len(family_results)
        )
        family_mean_shifts.append(
            sum(abs(item.probability - base) for item in family_results)
            / len(family_results)
        )
        family_label_agreements.append(
            sum(
                (item.probability >= boundary) == base_generated
                for item in family_results
            )
            / len(family_results)
        )

    largest_delta = largest.probability - base
    mean_absolute_shift = sum(family_mean_shifts) / len(family_mean_shifts)
    return RobustnessSummary(
        case_count=len(normalized),
        consistent_count=consistent_count,
        family_count=len(families),
        label_stability=sum(family_label_agreements)
        / len(family_label_agreements),
        score_consistency=1.0 - mean_absolute_shift,
        average_probability=sum(family_mean_probabilities)
        / len(family_mean_probabilities),
        mean_absolute_shift=mean_absolute_shift,
        minimum_probability=min(item.probability for item in normalized),
        maximum_probability=max(item.probability for item in normalized),
        largest_shift=abs(largest_delta),
        largest_shift_delta=largest_delta,
        largest_shift_transform=largest.display_name,
        largest_shift_name=largest.transform,
        largest_shift_severity=largest.severity,
        largest_shift_probability=largest.probability,
        largest_shift_flipped=(largest.probability >= boundary) != base_generated,
    )


def aggregate_transform_scores(
    base_probability: float,
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, str | float]]:
    """Return a compact clean-plus-family-average chart dataset."""

    base = validate_probability(base_probability, field="base probability")
    normalized = normalize_stress_results(results)
    grouped: OrderedDict[str, list[float]] = OrderedDict()
    for result in normalized:
        grouped.setdefault(_transform_family(result.transform), []).append(
            result.probability
        )

    rows: list[dict[str, str | float]] = [
        {"Scenario": "Clean image", "AIGC score": base}
    ]
    rows.extend(
        {
            "Scenario": format_transform_name(transform),
            "AIGC score": sum(scores) / len(scores),
        }
        for transform, scores in grouped.items()
    )
    return rows


def build_stress_table(
    base_probability: float,
    results: Sequence[Mapping[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, str | float]]:
    """Format detailed per-severity rows for Streamlit's native dataframe."""

    base = validate_probability(base_probability, field="base probability")
    boundary = validate_probability(threshold, field="threshold")
    if boundary in (0.0, 1.0):
        raise ValueError("threshold must be strictly between 0 and 1")
    base_generated = base >= boundary
    return [
        {
            "Transform": format_transform_name(result.transform),
            "Severity": result.severity,
            "AIGC score (%)": result.probability * 100.0,
            "Shift (pp)": (result.probability - base) * 100.0,
            "Label flipped": (
                "No"
                if (result.probability >= boundary) == base_generated
                else "Yes"
            ),
        }
        for result in normalize_stress_results(results)
    ]
