"""Finite working-scale grid utilities for downstream curve consumers."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import numpy as np

from .compatibility import MAX_FINITE_ABS_Z, MAX_FLOAT
from .effects import _require_finite, _to_array_kernel
from .errors import ValidationError

DEFAULT_GRID_POINTS = 801
DEFAULT_SPAN_MULTIPLIER = 4.5
GRID_EXPANSION_PADDING_MULTIPLIER = 0.25
MAX_FINITE_SPAN = float(np.finfo(float).max / 4.0)


def _coerce_finite_float(value: object, *, label: str) -> float:
    try:
        float_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} must be finite.") from exc
    if not isfinite(float_value):
        raise ValidationError(f"{label} must be finite.")
    return float_value


def max_safe_grid_span(
    theta_hat: float,
    se: float,
    *,
    natural_axis_upper_bound: float | None = None,
) -> float:
    """Return the largest span keeping endpoints and standardized distances finite."""

    center = _coerce_finite_float(theta_hat, label="Point estimate")
    standard_error = _coerce_finite_float(se, label="Standard error")
    if standard_error <= 0:
        raise ValidationError("Standard error must be positive.")

    natural_upper: float | None = None
    if natural_axis_upper_bound is not None:
        natural_upper = _coerce_finite_float(
            natural_axis_upper_bound,
            label="Natural-axis upper bound",
        )

    with np.errstate(over="ignore", invalid="ignore"):
        z_safe_span = float(MAX_FINITE_ABS_Z * standard_error)
    endpoint_headroom = max(MAX_FLOAT - abs(center), 0.0)
    span_limit = min(MAX_FINITE_SPAN, z_safe_span, endpoint_headroom)
    if natural_upper is not None and natural_upper > center:
        with np.errstate(over="ignore", invalid="ignore"):
            natural_headroom = natural_upper - center
        span_limit = min(span_limit, natural_headroom)
    return max(span_limit, 0.0)


def _build_grid_kernel(
    theta_hat: float,
    se: float,
    span_multiplier: float = DEFAULT_SPAN_MULTIPLIER,
    n: int = DEFAULT_GRID_POINTS,
    include_values: Sequence[float] | np.ndarray | None = None,
    max_span: float | None = None,
    *,
    strict_finite_output: bool,
) -> np.ndarray:
    if se <= 0:
        raise ValidationError("Standard error must be positive.")
    if span_multiplier <= 0:
        raise ValidationError("Span multiplier must be positive.")
    points = int(n)
    if points < 5:
        raise ValidationError("Grid must contain at least 5 points.")
    if points % 2 == 0:
        points += 1

    span = span_multiplier * se
    if include_values is not None:
        values = _to_array_kernel(include_values)
        if values.size:
            _require_finite(values, "Included grid values")
            with np.errstate(over="ignore"):
                required_span = float(np.max(np.abs(values - theta_hat)))
            if required_span > span:
                span = required_span + (GRID_EXPANSION_PADDING_MULTIPLIER * se)
    if max_span is not None:
        if max_span < 0:
            raise ValidationError("Maximum span must not be negative.")
        span = min(span, max_span)
    if strict_finite_output and not isfinite(span):
        raise ValidationError("Grid span exceeds the finite floating-point range.")
    if span == 0:
        return np.full(points, theta_hat, dtype=float)

    lower = theta_hat - span
    upper = theta_hat + span
    if strict_finite_output and (not isfinite(lower) or not isfinite(upper)):
        raise ValidationError("Grid endpoints exceed the finite floating-point range.")
    grid = np.linspace(lower, upper, num=points, dtype=float)
    if strict_finite_output and not np.isfinite(grid).all():
        raise ValidationError("Grid values exceed the finite floating-point range.")
    return grid


def build_grid(
    theta_hat: float,
    se: float,
    span_multiplier: float = DEFAULT_SPAN_MULTIPLIER,
    n: int = DEFAULT_GRID_POINTS,
    include_values: Sequence[float] | np.ndarray | None = None,
    max_span: float | None = None,
) -> np.ndarray:
    """Build a finite symmetric working-scale grid around a point estimate."""

    center = _coerce_finite_float(theta_hat, label="Point estimate")
    standard_error = _coerce_finite_float(se, label="Standard error")
    span_factor = _coerce_finite_float(span_multiplier, label="Span multiplier")
    try:
        points = int(n)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Grid points must be an integer.") from exc

    span_cap: float | None = None
    if max_span is not None:
        span_cap = _coerce_finite_float(max_span, label="Maximum span")

    included_values: np.ndarray | None = None
    if include_values is not None:
        try:
            included_values = _to_array_kernel(include_values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("Included grid values must be finite.") from exc

    with np.errstate(over="ignore", invalid="ignore"):
        return _build_grid_kernel(
            center,
            standard_error,
            span_factor,
            points,
            included_values,
            span_cap,
            strict_finite_output=True,
        )


def _legacy_build_grid(
    theta_hat: float,
    se: float,
    span_multiplier: float = DEFAULT_SPAN_MULTIPLIER,
    n: int = DEFAULT_GRID_POINTS,
    include_values: Sequence[float] | np.ndarray | None = None,
    max_span: float | None = None,
) -> np.ndarray:
    return _build_grid_kernel(
        theta_hat,
        se,
        span_multiplier,
        n,
        include_values,
        max_span,
        strict_finite_output=False,
    )
