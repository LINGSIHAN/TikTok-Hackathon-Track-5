"""Streamlit entrypoint for the transformation-aware AIGC detector demo."""

from __future__ import annotations

import hashlib
import importlib
import io
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Streamlit Cloud adds the entrypoint directory, rather than necessarily adding
# the repository root, to ``sys.path``. Resolve imports from this trusted local
# checkout so the app behaves the same from the repo root and from ``app/``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from app.ui_logic import (
    DEFAULT_THRESHOLD,
    aggregate_transform_scores,
    build_stress_table,
    interpret_probability,
    summarize_robustness,
    validate_probability,
)
from src.data.preprocessing import normalize_pil_image
from src.inference.predictor import (
    Predictor,
    prepare_stress_image,
)


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_EDGE = 8192
MAX_ASPECT_RATIO = 8.0
DEPLOYED_CHECKPOINT_PATH = Path("artifacts/checkpoints/model_v2.safetensors")
DEPLOYED_MODEL_LABEL = "GenImage v2"
LOGGER = logging.getLogger(__name__)


st.set_page_config(
    page_title="RealityCheck — AIGC image detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_predictor(checkpoint_path: str, checkpoint_version: int) -> Predictor:
    """Load one CPU predictor per checkpoint version for free-tier hosting."""

    del checkpoint_version  # Its value is part of the Streamlit cache key.
    return Predictor.from_checkpoint(checkpoint_path, device="cpu")


def _checkpoint_path() -> Path:
    configured = os.environ.get("AIGC_CHECKPOINT_PATH")
    path = Path(configured) if configured else DEPLOYED_CHECKPOINT_PATH
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_uploaded_image(uploaded_file: Any) -> tuple[Image.Image, bytes]:
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded file exceeds the 10 MB demo limit.")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "This image is too large for the demo. Please use an image "
                    "under 25 megapixels."
                )
            longest_edge = max(opened.size)
            shortest_edge = min(opened.size)
            if (
                longest_edge > MAX_IMAGE_EDGE
                or longest_edge / shortest_edge > MAX_ASPECT_RATIO
            ):
                raise ValueError(
                    "This image's dimensions are too extreme for the free-tier "
                    "demo. Use an image no wider or taller than 8192 pixels and "
                    "with an aspect ratio no greater than 8:1."
                )
            if getattr(opened, "is_animated", False):
                raise ValueError("Animated images are not supported by this demo.")
            opened.load()
            image = normalize_pil_image(opened)
    except Image.DecompressionBombError as error:
        raise ValueError("The image dimensions exceed the safe demo limit.") from error
    except UnidentifiedImageError as error:
        raise ValueError("The uploaded file is not a readable JPG or PNG image.") from error
    except OSError as error:
        raise ValueError("The image could not be decoded. Try exporting it again.") from error
    return image, raw


def _reset_results_for(upload_key: str) -> None:
    if st.session_state.get("active_upload_key") == upload_key:
        return
    st.session_state["active_upload_key"] = upload_key
    st.session_state.pop("base_probability", None)
    st.session_state.pop("stress_base_probability", None)
    st.session_state.pop("stress_results", None)


def _show_inference_error(action: str, error: Exception) -> None:
    LOGGER.exception("The Streamlit detector could not %s", action)
    st.error(
        f"The detector could not {action}. No prediction was produced. "
        "Please retry or check the deployment logs."
    )
    st.caption(f"Error category: `{type(error).__name__}`")


def _load_transform_suite() -> tuple[
    Any | None, list[tuple[str, str]], str | None
]:
    """Preflight the transform module before enabling the stress-test action."""

    try:
        module = importlib.import_module("src.transforms.robustness")
    except (ImportError, AttributeError):
        return (
            None,
            [],
            "The robustness transformations have not been integrated yet.",
        )
    except Exception:
        LOGGER.exception("The robustness transformation suite could not load")
        return None, [], "The robustness transformation suite could not be loaded."

    grid = getattr(module, "TRANSFORM_GRID", None)
    apply_transform = getattr(module, "apply_transform", None)
    if not isinstance(grid, Mapping) or not grid or not callable(apply_transform):
        return None, [], "The robustness transformation suite is incomplete."

    descriptors: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    try:
        for transform, severities in grid.items():
            if isinstance(severities, (str, bytes)):
                raise TypeError
            severity_values = list(severities)
            if not str(transform).strip() or not severity_values:
                raise ValueError
            for severity in severity_values:
                descriptor = (str(transform).casefold(), str(severity).casefold())
                if descriptor in seen:
                    raise ValueError
                seen.add(descriptor)
                descriptors.append((str(transform), str(severity)))
    except (TypeError, ValueError):
        return None, [], "The robustness transformation grid is invalid."
    return module, descriptors, None


def _show_empty_state() -> None:
    st.subheader("What this demo checks")
    first, second, third = st.columns(3)
    first.markdown(
        "**1 · Screen**\n\nGet the model's AIGC score for one JPG or PNG image."
    )
    second.markdown(
        "**2 · Stress test**\n\nRepeat the prediction after compression, blur, resize, noise, color, and crop changes."
    )
    third.markdown(
        "**3 · Compare**\n\nSee whether the label remains stable and which transformation changes it most."
    )


def _render_base_result(probability: float) -> None:
    interpretation = interpret_probability(probability)
    st.subheader("Screening result")
    score_column, label_column = st.columns([1, 1])
    score_column.metric("AIGC model score", f"{probability:.1%}")
    label_column.metric("Plain-language reading", interpretation.label)
    st.progress(
        probability,
        text=f"Model score: {probability:.1%} · boundary: {DEFAULT_THRESHOLD:.0%}",
    )
    if interpretation.generated_side is None:
        st.warning(f"**{interpretation.uncertainty}.** {interpretation.explanation}")
    else:
        st.info(f"**{interpretation.uncertainty}.** {interpretation.explanation}")


def _render_robustness_passport(
    base_probability: float,
    stress_results: list[dict[str, str | float]],
    transform_module: Any,
    expected_pairs: list[tuple[str, str]],
    image: Image.Image,
) -> None:
    summary = summarize_robustness(
        base_probability, stress_results, expected_pairs=expected_pairs
    )
    st.divider()
    st.header("Robustness passport")
    st.caption(
        "The same image was re-scored after common real-world transformations. "
        "This reports single-image stress stability—not dataset accuracy. "
        "Family bars average the tested severity levels equally. Large uploads "
        "use an aspect-preserving 1536 px working copy for both the passport's "
        "clean baseline and transformed scores."
    )
    if interpret_probability(base_probability).generated_side is None:
        st.warning(
            "The clean score is inconclusive. Boundary-side stability below is "
            "a mathematical sensitivity check, not a confident origin label."
        )

    stability, consistency, shift, score_range = st.columns(4)
    stability.metric(
        "Boundary-side stability",
        f"{summary.label_stability:.0%}",
        help=(
            f"Displayed value is balanced equally across {summary.family_count} "
            "transform families. Raw case count: "
            f"{summary.consistent_count} of {summary.case_count} stayed on the "
            "same side of the project-wide 50% boundary."
        ),
    )
    consistency.metric(
        "Score consistency",
        f"{summary.score_consistency:.0%}",
        help="One minus the family-balanced mean absolute score shift.",
    )
    shift.metric(
        "Largest score shift",
        f"{summary.largest_shift:.1%}",
        delta=f"{summary.largest_shift_delta:+.1%}",
        delta_color="off",
        help=summary.largest_shift_transform,
    )
    score_range.metric(
        "Transformed range",
        f"{summary.minimum_probability:.0%}–{summary.maximum_probability:.0%}",
        help=f"Family-balanced mean score: {summary.average_probability:.1%}",
    )
    st.caption(
        f"Most destabilizing case: **{summary.largest_shift_transform}** → "
        f"**{summary.largest_shift_probability:.1%}**"
        + (" · crossed the 50% boundary" if summary.largest_shift_flipped else "")
    )

    try:
        original_severity = next(
            severity
            for severity in transform_module.TRANSFORM_GRID[
                summary.largest_shift_name
            ]
            if str(severity) == summary.largest_shift_severity
        )
        stress_source = prepare_stress_image(image)
        worst_preview = transform_module.apply_transform(
            stress_source.copy(),
            summary.largest_shift_name,
            original_severity,
            42,
        )
        original_column, transformed_column = st.columns(2)
        original_column.image(
            stress_source,
            caption="Stress-test working image",
            use_container_width=True,
        )
        transformed_column.image(
            worst_preview,
            caption=f"Most destabilizing · {summary.largest_shift_transform}",
            use_container_width=True,
        )
    except Exception:
        LOGGER.exception("The most-destabilizing transform preview could not render")
        st.caption("A transformed preview is unavailable for this result.")

    chart_rows = aggregate_transform_scores(base_probability, stress_results)
    chart_frame = pd.DataFrame(chart_rows)
    st.vega_lite_chart(
        chart_frame,
        {
            "height": 330,
            "layer": [
                {
                    "mark": {
                        "type": "bar",
                        "color": "#25F4EE",
                        "cornerRadiusTopLeft": 4,
                        "cornerRadiusTopRight": 4,
                    },
                    "encoding": {
                        "x": {
                            "field": "Scenario",
                            "type": "nominal",
                            "sort": None,
                            "axis": {"title": None, "labelAngle": -25},
                        },
                        "y": {
                            "field": "AIGC score",
                            "type": "quantitative",
                            "scale": {"domain": [0, 1]},
                            "axis": {
                                "title": "AIGC model score",
                                "format": ".0%",
                            },
                        },
                        "tooltip": [
                            {"field": "Scenario", "type": "nominal"},
                            {
                                "field": "AIGC score",
                                "type": "quantitative",
                                "format": ".1%",
                            },
                        ],
                    },
                },
                {
                    "mark": {"type": "text", "dy": -9, "color": "#F5F7FA"},
                    "encoding": {
                        "x": {
                            "field": "Scenario",
                            "type": "nominal",
                            "sort": None,
                        },
                        "y": {
                            "field": "AIGC score",
                            "type": "quantitative",
                            "scale": {"domain": [0, 1]},
                        },
                        "text": {
                            "field": "AIGC score",
                            "type": "quantitative",
                            "format": ".0%",
                        },
                    },
                },
                {
                    "data": {"values": [{"boundary": DEFAULT_THRESHOLD}]},
                    "mark": {
                        "type": "rule",
                        "color": "#FE2C55",
                        "strokeDash": [6, 4],
                        "strokeWidth": 2,
                    },
                    "encoding": {
                        "y": {
                            "field": "boundary",
                            "type": "quantitative",
                            "scale": {"domain": [0, 1]},
                        }
                    },
                },
            ],
            "config": {"view": {"stroke": None}},
        },
        use_container_width=True,
    )
    st.caption("Dashed line: project-wide 50% decision boundary.")

    with st.expander("View every tested transformation and severity"):
        detail_frame = pd.DataFrame(
            build_stress_table(base_probability, stress_results)
        )
        st.dataframe(
            detail_frame,
            hide_index=True,
            use_container_width=True,
            column_config={
                "AIGC score (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Shift (pp)": st.column_config.NumberColumn(format="%+.1f"),
            },
        )
        st.download_button(
            "Download stress results (CSV)",
            data=detail_frame.to_csv(index=False).encode("utf-8"),
            file_name="realitycheck_stress_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    st.title("🛡️ RealityCheck")
    st.markdown(
        "#### Transformation-aware screening for AI-generated images"
    )
    st.write(
        "Upload one image to inspect its AIGC model score, then test whether that "
        "score survives everyday edits such as JPEG compression and cropping."
    )
    st.info(
        "**Use this as evidence, not proof.** The score is a model estimate, "
        "not a verified statement about who or what created an image."
    )

    checkpoint = _checkpoint_path()
    checkpoint_ready = checkpoint.is_file()
    checkpoint_version = checkpoint.stat().st_mtime_ns if checkpoint_ready else -1

    with st.sidebar:
        st.header("About this demo")
        st.write(
            "The detector uses a compact EfficientNet-B0 model and runs on CPU "
            "to stay within free hosting limits."
        )
        if checkpoint_ready:
            if checkpoint.name == DEPLOYED_CHECKPOINT_PATH.name:
                st.success(f"Model checkpoint available · {DEPLOYED_MODEL_LABEL}")
            else:
                st.success("Custom model checkpoint available")
        else:
            st.warning("Model checkpoint not deployed")
        st.divider()
        st.subheader("Privacy")
        st.caption(
            "Uploaded pixels are processed in memory and are not intentionally "
            "saved by this app. The hosting provider may retain operational logs."
        )
        st.subheader("Known limitations")
        st.caption(
            "Results may be unreliable for unseen generators, screenshots, "
            "heavy editing, illustrations, or images unlike the training data."
        )

    if not checkpoint_ready:
        st.warning(
            "The app interface is ready, but its trained checkpoint is missing. "
            f"Expected `{checkpoint.as_posix()}`. Predictions remain disabled."
        )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="JPG or PNG, up to 10 MB and 25 megapixels.",
    )
    if uploaded_file is None:
        _show_empty_state()
        return

    try:
        image, raw = _read_uploaded_image(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        return

    upload_key = ":".join(
        (
            hashlib.sha256(raw).hexdigest(),
            str(checkpoint.resolve()),
            str(checkpoint_version),
        )
    )
    _reset_results_for(upload_key)

    preview_column, result_column = st.columns([1, 1], gap="large")
    with preview_column:
        st.subheader("Uploaded image")
        st.image(
            prepare_stress_image(image),
            caption=f"{image.width} × {image.height} pixels",
        )

    with result_column:
        analyze = st.button(
            "Analyze image",
            type="primary",
            use_container_width=True,
            disabled=not checkpoint_ready,
        )
        if analyze:
            try:
                with st.spinner("Loading the detector and screening the image…"):
                    predictor = load_predictor(
                        str(checkpoint.resolve()), checkpoint_version
                    )
                    probability = validate_probability(predictor.predict_pil(image))
                st.session_state["base_probability"] = probability
                st.session_state.pop("stress_base_probability", None)
                st.session_state.pop("stress_results", None)
            except Exception as error:  # Streamlit must fail visibly, not fabricate.
                st.session_state.pop("base_probability", None)
                st.session_state.pop("stress_base_probability", None)
                st.session_state.pop("stress_results", None)
                _show_inference_error("analyze this image", error)

        base_probability = st.session_state.get("base_probability")
        if base_probability is not None:
            _render_base_result(float(base_probability))
            transform_module, expected_pairs, transform_error = _load_transform_suite()
            if transform_error:
                st.caption(transform_error)
            run_stress_test = st.button(
                "Run robustness stress test",
                use_container_width=True,
                help="Tests every configured transform and severity on CPU.",
                disabled=transform_module is None,
            )
            if run_stress_test:
                try:
                    with st.spinner("Testing real-world transformations…"):
                        predictor = load_predictor(
                            str(checkpoint.resolve()), checkpoint_version
                        )
                        stress_source = prepare_stress_image(image)
                        stress_base_probability = validate_probability(
                            predictor.predict_pil(stress_source)
                        )
                        stress_results = predictor.stress_test(image)
                        # Validate the complete contract before saving UI state.
                        summarize_robustness(
                            stress_base_probability,
                            stress_results,
                            expected_pairs=expected_pairs,
                        )
                    st.session_state["stress_base_probability"] = (
                        stress_base_probability
                    )
                    st.session_state["stress_results"] = stress_results
                except Exception as error:  # See comment above.
                    st.session_state.pop("stress_base_probability", None)
                    st.session_state.pop("stress_results", None)
                    _show_inference_error("complete the stress test", error)
        elif not analyze:
            st.caption("Select **Analyze image** to generate a model score.")

    stress_results = st.session_state.get("stress_results")
    stress_base_probability = st.session_state.get("stress_base_probability")
    if stress_results is not None and stress_base_probability is not None:
        transform_module, expected_pairs, transform_error = _load_transform_suite()
        if transform_module is not None:
            _render_robustness_passport(
                float(stress_base_probability),
                stress_results,
                transform_module,
                expected_pairs,
                image,
            )
        elif transform_error:
            st.warning(transform_error)

    st.divider()
    st.caption(
        "RealityCheck is a hackathon prototype. Do not use it as the sole basis "
        "for moderation, attribution, disciplinary, legal, or safety decisions."
    )


if __name__ == "__main__":
    main()
