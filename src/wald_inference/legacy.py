"""Frozen direct-call behavior for migration from :mod:`confcurve.core`.

This module intentionally preserves historical coercion, warning, and
nonfinite-output behavior. New consumers should use the strict canonical
functions exported from :mod:`wald_inference`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .compatibility import ObservedResult, _legacy_confidence_curve, _legacy_summaries
from .effects import (
    EffectResult,
    EffectValues,
    _legacy_from_working_scale,
    _legacy_to_working_scale,
)
from .grid import (
    DEFAULT_GRID_POINTS,
    DEFAULT_SPAN_MULTIPLIER,
    _legacy_build_grid,
)
from .likelihood import (
    LikelihoodValues,
    _legacy_log_relative_likelihood,
    _legacy_relative_likelihood,
)
from .reconstruction import _legacy_estimate_se

__all__ = [
    "build_grid",
    "confidence_curve",
    "estimate_se",
    "from_working_scale",
    "log_relative_likelihood",
    "relative_likelihood",
    "summaries",
    "to_working_scale",
]


def to_working_scale(
    effect_type: str,
    values: EffectValues,
) -> EffectResult:
    """Preserve the frozen natural-to-working transformation contract."""

    return _legacy_to_working_scale(effect_type, values)


def from_working_scale(
    effect_type: str,
    values: EffectValues,
) -> EffectResult:
    """Preserve the frozen working-to-natural transformation contract."""

    return _legacy_from_working_scale(effect_type, values)


def estimate_se(theta_hat: float, lower: float, upper: float) -> float:
    """Preserve the frozen direct standard-error reconstruction contract."""

    return _legacy_estimate_se(theta_hat, lower, upper)


def build_grid(
    theta_hat: float,
    se: float,
    span_multiplier: float = DEFAULT_SPAN_MULTIPLIER,
    n: int = DEFAULT_GRID_POINTS,
    include_values: Sequence[float] | None = None,
    max_span: float | None = None,
) -> np.ndarray:
    """Preserve the frozen grid coercion and nonfinite-output contract."""

    return _legacy_build_grid(
        theta_hat,
        se,
        span_multiplier,
        n,
        include_values,
        max_span,
    )


def confidence_curve(
    theta: float | np.ndarray,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    """Preserve the frozen two-sided confidence-curve contract."""

    return _legacy_confidence_curve(theta, theta_hat, se)


def relative_likelihood(
    theta: LikelihoodValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    """Preserve the frozen normalized relative-likelihood contract."""

    return _legacy_relative_likelihood(theta, theta_hat, se)


def log_relative_likelihood(
    theta: LikelihoodValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    """Preserve the frozen log-relative-likelihood contract."""

    return _legacy_log_relative_likelihood(theta, theta_hat, se)


def summaries(theta_hat: float, se: float, null_value: float) -> dict[str, float | None]:
    """Preserve the frozen null-summary mapping, including sentinel ``None`` values."""

    return _legacy_summaries(theta_hat, se, null_value)
