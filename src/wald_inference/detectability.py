"""Legacy closed-form critical-effect benchmark utilities."""

from __future__ import annotations

from math import isfinite

import numpy as np
from scipy.stats import norm

from .errors import ValidationError

Z975 = float(norm.ppf(0.975))
Z80 = float(norm.ppf(0.80))
LEGACY_ALPHA = 0.05
LEGACY_POWER = 0.80


def legacy_critical_effect_distance(se: float) -> float:
    """Return the preserved z-sum benchmark distance for alpha=.05/power=.80."""

    try:
        standard_error = float(se)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Standard error must be positive.") from exc
    if not isfinite(standard_error) or standard_error <= 0:
        raise ValidationError("Standard error must be positive.")
    with np.errstate(over="ignore", invalid="ignore"):
        distance = float((Z975 + Z80) * standard_error)
    if not isfinite(distance):
        raise ValidationError(
            "Legacy critical-effect distance exceeds the finite floating-point range."
        )
    return distance


def legacy_critical_effect_markers(
    null_value: float,
    se: float,
) -> tuple[float, float]:
    """Return symmetric legacy z-sum benchmark markers around the null."""

    try:
        null = float(null_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Null value must be finite.") from exc
    if not isfinite(null):
        raise ValidationError("Null value must be finite.")
    distance = legacy_critical_effect_distance(se)
    with np.errstate(over="ignore", invalid="ignore"):
        lower = null - distance
        upper = null + distance
    if not isfinite(lower) or not isfinite(upper):
        raise ValidationError(
            "Legacy critical-effect markers exceed the finite floating-point range."
        )
    return float(lower), float(upper)


def critical_effect_distance(se: float) -> float:
    """Backward-compatible alias for the legacy z-sum benchmark distance."""

    return legacy_critical_effect_distance(se)


def critical_effect_markers(null_value: float, se: float) -> tuple[float, float]:
    """Backward-compatible alias for the legacy z-sum benchmark markers."""

    return legacy_critical_effect_markers(null_value, se)
