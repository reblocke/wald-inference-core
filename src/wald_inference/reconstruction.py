"""Reconstruct a Wald estimate and standard error from a reported 95% CI."""

from __future__ import annotations

from contextlib import nullcontext
from math import isfinite

import numpy as np
from scipy.stats import norm

from .effects import (
    DEFAULT_EFFECT_TYPE,
    from_working_scale,
    get_effect_spec,
    to_working_scale,
)
from .errors import ValidationError
from .types import (
    EffectSpec,
    EstimateSource,
    StandardErrorEstimate,
    WaldReconstruction,
)

Z975 = float(norm.ppf(0.975))
ASYMMETRY_RELATIVE_TOLERANCE = 0.02
ESTIMATE_MATCH_RELATIVE_TOLERANCE = 0.02
ESTIMATE_MATCH_ABSOLUTE_TOLERANCE = 1e-12


def _coerce_finite_float(value: object, *, label: str) -> float:
    try:
        float_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} must be finite.") from exc
    if not isfinite(float_value):
        raise ValidationError(f"{label} must be finite.")
    return float_value


def _working_scale_midpoint_and_half_width(
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Return a finite midpoint and half-width without overflowing."""

    midpoint = (lower * 0.5) + (upper * 0.5)
    half_width = (upper * 0.5) - (lower * 0.5)
    if not isfinite(midpoint):
        raise ValidationError("The inferred CI midpoint must be finite on the working scale.")
    if not isfinite(half_width) or half_width <= 0:
        raise ValidationError(
            "The supplied 95% confidence interval must have positive width on the working scale."
        )
    return midpoint, half_width


def _working_scale_difference(
    minuend: float,
    subtrahend: float,
    *,
    label: str,
) -> float:
    """Return a finite difference without overflowing on opposite signs."""

    difference = 2.0 * ((minuend * 0.5) - (subtrahend * 0.5))
    if not isfinite(difference):
        raise ValidationError(f"{label} must be finite on the working scale.")
    return difference


def _estimate_se_details_kernel(
    theta_hat: float,
    lower: float,
    upper: float,
    *,
    finite_mean_fallback: bool,
) -> StandardErrorEstimate:
    _, ci_half_width = _working_scale_midpoint_and_half_width(lower, upper)
    se_width = ci_half_width / Z975
    se_lower = (
        _working_scale_difference(
            theta_hat,
            lower,
            label="Lower-side CI width",
        )
        / Z975
    )
    se_upper = (
        _working_scale_difference(
            upper,
            theta_hat,
            label="Upper-side CI width",
        )
        / Z975
    )
    warning_context = (
        np.errstate(over="ignore", invalid="ignore") if finite_mean_fallback else nullcontext()
    )
    with warning_context:
        mean_side_se = float(np.mean([se_lower, se_upper]))
    if finite_mean_fallback and not isfinite(mean_side_se):
        mean_side_se = (se_lower * 0.5) + (se_upper * 0.5)
    if finite_mean_fallback and (not isfinite(mean_side_se) or mean_side_se <= 0):
        raise ValidationError(
            "Reconstructed standard error must be finite and positive on the working scale."
        )

    relative_asymmetry = abs(se_upper - se_lower) / max(
        abs(mean_side_se),
        np.finfo(float).eps,
    )
    if relative_asymmetry > ASYMMETRY_RELATIVE_TOLERANCE:
        return StandardErrorEstimate(
            se=mean_side_se,
            method="mean_side_se",
            se_lower=se_lower,
            se_upper=se_upper,
            se_width=se_width,
            relative_asymmetry=relative_asymmetry,
        )

    return StandardErrorEstimate(
        se=se_width,
        method="ci_width",
        se_lower=se_lower,
        se_upper=se_upper,
        se_width=se_width,
        relative_asymmetry=relative_asymmetry,
    )


def estimate_se_details(
    theta_hat: float,
    lower: float,
    upper: float,
) -> StandardErrorEstimate:
    """Reconstruct working-scale SE details from a nominal 95% Wald CI."""

    theta_value = _coerce_finite_float(theta_hat, label="Estimate")
    lower_value = _coerce_finite_float(lower, label="Lower confidence limit")
    upper_value = _coerce_finite_float(upper, label="Upper confidence limit")
    return _estimate_se_details_kernel(
        theta_value,
        lower_value,
        upper_value,
        finite_mean_fallback=True,
    )


def estimate_se(theta_hat: float, lower: float, upper: float) -> float:
    """Return only the strictly reconstructed working-scale SE."""

    return estimate_se_details(theta_hat, lower, upper).se


def _legacy_estimate_se(theta_hat: float, lower: float, upper: float) -> float:
    return _estimate_se_details_kernel(
        theta_hat,
        lower,
        upper,
        finite_mean_fallback=False,
    ).se


def asymmetry_warning(spec: EffectSpec, relative_asymmetry: float) -> str | None:
    """Return the preserved non-Wald/rounding warning when asymmetry is material."""

    if relative_asymmetry <= ASYMMETRY_RELATIVE_TOLERANCE:
        return None

    if spec.family == "ratio":
        return (
            "CI is not symmetric on the log scale; "
            "this may reflect rounding or a non-Wald interval. "
            "The plotted curves are a Wald approximation."
        )
    return (
        "CI is not symmetric on the working scale; "
        "this may reflect rounding or a non-Wald interval. "
        "The plotted curves are a Wald approximation."
    )


def reconstruct_wald(
    effect_type: str = DEFAULT_EFFECT_TYPE,
    estimate: float | int | None = None,
    lower: float | int | None = None,
    upper: float | int | None = None,
    null_value: float | int | None = None,
) -> WaldReconstruction:
    """Create the preserved Wald reconstruction from legacy reported inputs."""

    spec = get_effect_spec(effect_type)
    if lower is None or upper is None:
        raise ValidationError("Lower and upper confidence limits are required.")

    lower_display = _coerce_finite_float(lower, label="Lower confidence limit")
    upper_display = _coerce_finite_float(upper, label="Upper confidence limit")
    provided_estimate_display = (
        None if estimate is None else _coerce_finite_float(estimate, label="Estimate")
    )
    default_null_applied = null_value is None
    null_display = _coerce_finite_float(
        spec.default_null if null_value is None else null_value,
        label="Null value",
    )

    if lower_display >= upper_display:
        raise ValidationError(
            "The lower confidence limit must be less than the upper confidence limit."
        )

    if spec.positive_only:
        positive_values = [lower_display, upper_display, null_display]
        if provided_estimate_display is not None:
            positive_values.append(provided_estimate_display)
        if any(value <= 0 for value in positive_values):
            raise ValidationError(
                f"{spec.label} inputs must be strictly positive on the natural scale."
            )

    lower_working = float(to_working_scale(effect_type, lower_display))
    upper_working = float(to_working_scale(effect_type, upper_display))
    estimate_working, ci_half_width_working = _working_scale_midpoint_and_half_width(
        lower_working,
        upper_working,
    )
    estimate_display = float(from_working_scale(effect_type, estimate_working))
    null_working = float(to_working_scale(effect_type, null_display))
    estimate_match_tolerance = max(
        ESTIMATE_MATCH_ABSOLUTE_TOLERANCE,
        ESTIMATE_MATCH_RELATIVE_TOLERANCE * ci_half_width_working,
    )

    provided_estimate_working: float | None = None
    estimate_source: EstimateSource
    if provided_estimate_display is None:
        estimate_source = "inferred_from_ci"
    else:
        provided_estimate_working = float(to_working_scale(effect_type, provided_estimate_display))
        if abs(provided_estimate_working - estimate_working) > estimate_match_tolerance:
            raise ValidationError(
                "Provided estimate is inconsistent with the supplied 95% confidence "
                "interval on the working scale beyond the rounding tolerance."
            )
        estimate_source = "provided_validated"

    reconstructed_se = estimate_se_details(
        estimate_working,
        lower_working,
        upper_working,
    )
    observed_estimate_working = (
        estimate_working if provided_estimate_working is None else provided_estimate_working
    )
    observed_se = estimate_se_details(
        observed_estimate_working,
        lower_working,
        upper_working,
    )
    warning = asymmetry_warning(spec, observed_se.relative_asymmetry)
    warnings = () if warning is None else (warning,)

    return WaldReconstruction(
        effect_spec=spec,
        estimate_display=estimate_display,
        estimate_working=estimate_working,
        estimate_source=estimate_source,
        provided_estimate_display=provided_estimate_display,
        provided_estimate_working=provided_estimate_working,
        lower_display=lower_display,
        upper_display=upper_display,
        lower_working=lower_working,
        upper_working=upper_working,
        null_display=null_display,
        null_working=null_working,
        default_null_applied=default_null_applied,
        standard_error=reconstructed_se.se,
        se_method=reconstructed_se.method,
        se_lower=reconstructed_se.se_lower,
        se_upper=reconstructed_se.se_upper,
        se_width=reconstructed_se.se_width,
        relative_asymmetry=observed_se.relative_asymmetry,
        warnings=warnings,
    )


def reconstruct_wald_from_95_ci(
    effect_type: str = DEFAULT_EFFECT_TYPE,
    estimate: float | int | None = None,
    lower: float | int | None = None,
    upper: float | int | None = None,
    null_value: float | int | None = None,
) -> WaldReconstruction:
    """Canonical explicit name for :func:`reconstruct_wald`."""

    return reconstruct_wald(
        effect_type=effect_type,
        estimate=estimate,
        lower=lower,
        upper=upper,
        null_value=null_value,
    )
