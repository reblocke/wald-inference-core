"""Supported effect measures and natural/working-scale transformations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .errors import ValidationError
from .types import EffectSpec

DEFAULT_EFFECT_TYPE = "odds_ratio"

EFFECT_SPECS: dict[str, EffectSpec] = {
    "odds_ratio": EffectSpec(
        key="odds_ratio",
        label="Odds ratio",
        family="ratio",
        working_scale="log",
        default_null=1.0,
        positive_only=True,
    ),
    "risk_ratio": EffectSpec(
        key="risk_ratio",
        label="Risk ratio",
        family="ratio",
        working_scale="log",
        default_null=1.0,
        positive_only=True,
    ),
    "hazard_ratio": EffectSpec(
        key="hazard_ratio",
        label="Hazard ratio",
        family="ratio",
        working_scale="log",
        default_null=1.0,
        positive_only=True,
    ),
    "incidence_rate_ratio": EffectSpec(
        key="incidence_rate_ratio",
        label="Incidence rate ratio",
        family="ratio",
        working_scale="log",
        default_null=1.0,
        positive_only=True,
    ),
    "ratio_of_means": EffectSpec(
        key="ratio_of_means",
        label="Ratio of means",
        family="ratio",
        working_scale="log",
        default_null=1.0,
        positive_only=True,
    ),
    "mean_difference": EffectSpec(
        key="mean_difference",
        label="Mean difference",
        family="additive",
        working_scale="identity",
        default_null=0.0,
        positive_only=False,
    ),
    "risk_difference": EffectSpec(
        key="risk_difference",
        label="Risk difference",
        family="additive",
        working_scale="identity",
        default_null=0.0,
        positive_only=False,
    ),
    "rate_difference": EffectSpec(
        key="rate_difference",
        label="Rate difference",
        family="additive",
        working_scale="identity",
        default_null=0.0,
        positive_only=False,
    ),
    "regression_coefficient": EffectSpec(
        key="regression_coefficient",
        label="Regression coefficient",
        family="additive",
        working_scale="identity",
        default_null=0.0,
        positive_only=False,
    ),
}

EffectValues = float | Sequence[float] | np.ndarray
EffectResult = float | np.float64 | np.ndarray


def _lookup_effect_spec(effect_type: str) -> EffectSpec:
    return EFFECT_SPECS[effect_type]


def _unsupported_effect_error(effect_type: object) -> ValidationError:
    valid = ", ".join(sorted(EFFECT_SPECS))
    return ValidationError(f"Unsupported effect type {effect_type!r}. Expected one of: {valid}.")


def get_effect_spec(effect_type: str) -> EffectSpec:
    """Return the exact registry entry for a supported effect type."""

    try:
        return _lookup_effect_spec(effect_type)
    except (KeyError, TypeError) as exc:
        raise _unsupported_effect_error(effect_type) from exc


def _legacy_get_effect_spec(effect_type: str) -> EffectSpec:
    try:
        return _lookup_effect_spec(effect_type)
    except KeyError as exc:
        raise _unsupported_effect_error(effect_type) from exc


def _to_array_kernel(values: EffectValues) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _strict_to_array(values: EffectValues) -> np.ndarray:
    try:
        return _to_array_kernel(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Values must be numeric and finite.") from exc


def _maybe_scalar(original: EffectValues, values: np.ndarray) -> EffectResult:
    if np.isscalar(original):
        return float(values.reshape(-1)[0])
    return values


def _require_finite(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise ValidationError(f"{label} must be finite.")


def _to_working_scale_kernel(
    spec: EffectSpec,
    original: EffectValues,
    array: np.ndarray,
) -> EffectResult:
    _require_finite(array, "Values")

    if spec.working_scale == "log":
        if np.any(array <= 0):
            raise ValidationError(
                f"{spec.label} values must be strictly positive on the natural scale."
            )
        array = np.log(array)

    return _maybe_scalar(original, array)


def to_working_scale(effect_type: str, values: EffectValues) -> EffectResult:
    """Convert natural-scale values to the effect measure's Wald working scale."""

    spec = get_effect_spec(effect_type)
    return _to_working_scale_kernel(spec, values, _strict_to_array(values))


def _legacy_to_working_scale(
    effect_type: str,
    values: EffectValues,
) -> EffectResult:
    spec = _legacy_get_effect_spec(effect_type)
    return _to_working_scale_kernel(spec, values, _to_array_kernel(values))


def _from_working_scale_kernel(
    spec: EffectSpec,
    original: EffectValues,
    array: np.ndarray,
) -> EffectResult:
    _require_finite(array, "Working-scale values")

    if spec.working_scale == "log":
        array = np.exp(array)

    return _maybe_scalar(original, array)


def from_working_scale(effect_type: str, values: EffectValues) -> EffectResult:
    """Convert working-scale values to the effect measure's natural scale."""

    spec = get_effect_spec(effect_type)
    array = _strict_to_array(values)
    with np.errstate(over="ignore", invalid="ignore"):
        result = _from_working_scale_kernel(spec, values, array)
    _require_finite(np.asarray(result, dtype=float), "Transformed values")
    return result


def _legacy_from_working_scale(
    effect_type: str,
    values: EffectValues,
) -> EffectResult:
    spec = _legacy_get_effect_spec(effect_type)
    return _from_working_scale_kernel(spec, values, _to_array_kernel(values))
