"""Observed Wald standardized distances and compatibility curves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite

import numpy as np
from scipy.stats import norm

from .errors import ValidationError
from .types import WaldPointSummary

MAX_FLOAT = float(np.finfo(float).max)
LOG_MAX_FLOAT = float(np.log(np.finfo(float).max))
MAX_FINITE_ABS_Z = float(np.sqrt(np.finfo(float).max))

ObservedValues = float | Sequence[float] | np.ndarray
ObservedResult = np.float64 | np.ndarray
ObservedKernel = Callable[[ObservedValues, float, float], ObservedResult]


def _to_array_kernel(values: ObservedValues) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _strict_to_array(values: ObservedValues) -> np.ndarray:
    try:
        return _to_array_kernel(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Evaluation points must be numeric and finite.") from exc


def _validate_center_and_se(theta_hat: float, se: float) -> tuple[float, float]:
    try:
        center = float(theta_hat)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Point estimate must be finite.") from exc
    try:
        standard_error = float(se)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Standard error must be positive.") from exc
    if not isfinite(center):
        raise ValidationError("Point estimate must be finite.")
    if not isfinite(standard_error) or standard_error <= 0:
        raise ValidationError("Standard error must be positive.")
    return center, standard_error


def _standardized_distance_kernel(
    theta: ObservedValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    values = _to_array_kernel(theta)
    if se <= 0:
        raise ValidationError("Standard error must be positive.")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        differences = values - theta_hat
        z_values = differences / se
        overflowed_differences = ~np.isfinite(differences)
        if np.any(overflowed_differences):
            scaled_z_values = ((0.5 * values) - (0.5 * theta_hat)) / (0.5 * se)
            z_values = np.where(overflowed_differences, scaled_z_values, z_values)

    if not np.isfinite(z_values).all():
        raise ValidationError("Standardized distance exceeds the finite floating-point range.")
    return z_values


def standardized_distance(
    theta: ObservedValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    """Compute finite Wald standardized distances from the point estimate."""

    values = _strict_to_array(theta)
    if not np.isfinite(values).all():
        raise ValidationError("Evaluation points must be finite.")
    center, standard_error = _validate_center_and_se(theta_hat, se)
    return _standardized_distance_kernel(values, center, standard_error)


def _compatibility_curve_kernel(z_values: ObservedResult) -> ObservedResult:
    return 2.0 * norm.sf(np.abs(z_values))


def compatibility_curve(
    theta: ObservedValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    """Map standardized distance to two-sided Wald compatibility."""

    z_values = standardized_distance(theta, theta_hat=theta_hat, se=se)
    return _compatibility_curve_kernel(z_values)


def confidence_curve(
    theta: ObservedValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    """Backward-compatible name for :func:`compatibility_curve`."""

    return compatibility_curve(theta, theta_hat=theta_hat, se=se)


def _legacy_confidence_curve(
    theta: ObservedValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    return _compatibility_curve_kernel(_standardized_distance_kernel(theta, theta_hat, se))


def _wald_point_summary_kernel(
    theta_hat: float,
    se: float,
    candidate_working: float,
    *,
    standardized_distance_function: ObservedKernel,
    log_relative_likelihood_function: ObservedKernel,
) -> WaldPointSummary:
    try:
        z_value = float(
            standardized_distance_function(
                candidate_working,
                theta_hat,
                se,
            ).reshape(-1)[0]
        )
    except ValidationError:
        z_value = None

    if z_value is None or abs(z_value) > MAX_FINITE_ABS_Z:
        return WaldPointSummary(
            candidate_working=candidate_working,
            relative_likelihood=0.0,
            log_relative_likelihood=None,
            likelihood_ratio_mle_to_candidate=None,
            log_likelihood_ratio_mle_to_candidate=None,
            two_sided_wald_p_value=0.0,
            z_value=None,
        )

    log_relative = float(
        log_relative_likelihood_function(
            candidate_working,
            theta_hat,
            se,
        ).reshape(-1)[0]
    )
    relative = float(np.exp(log_relative))
    log_mle_to_candidate = -log_relative
    mle_to_candidate = (
        None if log_mle_to_candidate > LOG_MAX_FLOAT else float(np.exp(log_mle_to_candidate))
    )
    two_sided_p = float(
        _compatibility_curve_kernel(np.asarray(z_value, dtype=np.float64)).reshape(-1)[0]
    )

    return WaldPointSummary(
        candidate_working=candidate_working,
        relative_likelihood=relative,
        log_relative_likelihood=log_relative,
        likelihood_ratio_mle_to_candidate=mle_to_candidate,
        log_likelihood_ratio_mle_to_candidate=log_mle_to_candidate,
        two_sided_wald_p_value=two_sided_p,
        z_value=z_value,
    )


def _point_summary_mapping(summary: WaldPointSummary) -> dict[str, float | None]:
    return {
        "null_relative_likelihood": summary.relative_likelihood,
        "log_null_relative_likelihood": summary.log_relative_likelihood,
        "likelihood_ratio_mle_to_null": summary.likelihood_ratio_mle_to_candidate,
        "log_likelihood_ratio_mle_to_null": (summary.log_likelihood_ratio_mle_to_candidate),
        "two_sided_wald_p_value": summary.two_sided_wald_p_value,
        "null_z_value": summary.z_value,
    }


def wald_point_summary(
    theta_hat: float,
    se: float,
    candidate_working: float,
) -> WaldPointSummary:
    """Summarize observed compatibility and support at one candidate value."""

    center, standard_error = _validate_center_and_se(theta_hat, se)
    try:
        candidate = float(candidate_working)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Candidate value must be finite.") from exc
    if not isfinite(candidate):
        raise ValidationError("Candidate value must be finite.")

    from .likelihood import log_relative_likelihood

    return _wald_point_summary_kernel(
        center,
        standard_error,
        candidate,
        standardized_distance_function=standardized_distance,
        log_relative_likelihood_function=log_relative_likelihood,
    )


def summaries(theta_hat: float, se: float, null_value: float) -> dict[str, float | None]:
    """Return the strict canonical null-summary mapping."""

    return _point_summary_mapping(
        wald_point_summary(
            theta_hat=theta_hat,
            se=se,
            candidate_working=null_value,
        )
    )


def _legacy_summaries(
    theta_hat: float,
    se: float,
    null_value: float,
) -> dict[str, float | None]:
    from .likelihood import _legacy_log_relative_likelihood

    return _point_summary_mapping(
        _wald_point_summary_kernel(
            theta_hat,
            se,
            null_value,
            standardized_distance_function=_standardized_distance_kernel,
            log_relative_likelihood_function=_legacy_log_relative_likelihood,
        )
    )
