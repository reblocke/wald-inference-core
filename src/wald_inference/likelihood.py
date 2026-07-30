"""Normalized Wald relative likelihood and support comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import numpy as np

from .compatibility import (
    LOG_MAX_FLOAT,
    ObservedResult,
    _standardized_distance_kernel,
    standardized_distance,
)
from .compatibility import (
    wald_point_summary as wald_point_summary,
)
from .errors import ValidationError
from .types import SupportComparison, SupportInterval

MAX_FLOAT = float(np.finfo(float).max)
S_MINUS_2_SUPPORT_CUTOFF = -2.0
S_MINUS_2_DISTANCE = 2.0

LikelihoodValues = float | Sequence[float] | np.ndarray


def _relative_likelihood_kernel(z_values: ObservedResult) -> ObservedResult:
    return np.exp(-0.5 * np.square(z_values))


def relative_likelihood(
    theta: LikelihoodValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    """Map standardized distance to normalized Wald relative likelihood."""

    z_values = standardized_distance(theta, theta_hat=theta_hat, se=se)
    with np.errstate(over="ignore", under="ignore"):
        return _relative_likelihood_kernel(z_values)


def _legacy_relative_likelihood(
    theta: LikelihoodValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    return _relative_likelihood_kernel(_standardized_distance_kernel(theta, theta_hat, se))


def _log_relative_likelihood_kernel(z_values: ObservedResult) -> ObservedResult:
    return -0.5 * np.square(z_values)


def log_relative_likelihood(
    theta: LikelihoodValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    """Return finite log relative likelihood on the Wald working scale."""

    z_values = standardized_distance(theta, theta_hat=theta_hat, se=se)
    with np.errstate(over="ignore", invalid="ignore"):
        log_likelihood = _log_relative_likelihood_kernel(z_values)
    if not np.isfinite(log_likelihood).all():
        raise ValidationError("Log relative likelihood exceeds the finite floating-point range.")
    return log_likelihood


def _legacy_log_relative_likelihood(
    theta: LikelihoodValues,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    return _log_relative_likelihood_kernel(_standardized_distance_kernel(theta, theta_hat, se))


def _exp_or_none(log_value: float | None) -> float | None:
    if log_value is None or log_value > LOG_MAX_FLOAT:
        return None
    return float(np.exp(log_value))


def _log_support_ratio_kernel(
    log_candidate_a: ObservedResult,
    log_candidate_b: ObservedResult,
) -> ObservedResult:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            log_ratio = np.subtract(log_candidate_a, log_candidate_b)
    except ValueError as exc:
        raise ValidationError("Support-ratio values must be broadcast-compatible.") from exc
    if not np.isfinite(log_ratio).all():
        raise ValidationError("Log support ratio exceeds the finite floating-point range.")
    return log_ratio


def log_support_ratio(
    candidate_a_working: LikelihoodValues,
    candidate_b_working: LikelihoodValues,
    *,
    theta_hat: float,
    se: float,
) -> ObservedResult:
    """Return finite log L(A)/L(B) under the normalized Wald reconstruction."""

    log_candidate_a = log_relative_likelihood(
        candidate_a_working,
        theta_hat=theta_hat,
        se=se,
    )
    log_candidate_b = log_relative_likelihood(
        candidate_b_working,
        theta_hat=theta_hat,
        se=se,
    )
    return _log_support_ratio_kernel(log_candidate_a, log_candidate_b)


def _coerce_support_pair(
    candidate_a_working: float,
    candidate_b_working: float,
) -> tuple[float, float]:
    try:
        candidate_a = float(candidate_a_working)
        candidate_b = float(candidate_b_working)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Support comparison values must be finite.") from exc
    if not isfinite(candidate_a) or not isfinite(candidate_b):
        raise ValidationError("Support comparison values must be finite.")
    return candidate_a, candidate_b


def support_ratio(
    candidate_a_working: float,
    candidate_b_working: float,
    *,
    theta_hat: float,
    se: float,
) -> float | None:
    """Return L(A)/L(B), or ``None`` when exponentiation would overflow."""

    candidate_a, candidate_b = _coerce_support_pair(
        candidate_a_working,
        candidate_b_working,
    )
    log_ratio = float(
        log_support_ratio(
            candidate_a,
            candidate_b,
            theta_hat=theta_hat,
            se=se,
        )
    )
    return _exp_or_none(log_ratio)


def support_comparison(
    candidate_working: float,
    reference_working: float,
    *,
    theta_hat: float,
    se: float,
) -> SupportComparison:
    """Compare candidate support with the MLE and with a reference value."""

    candidate, reference = _coerce_support_pair(candidate_working, reference_working)

    log_values = log_relative_likelihood(
        np.asarray([candidate, reference]),
        theta_hat=theta_hat,
        se=se,
    )
    log_candidate = float(log_values[0])
    log_reference = float(log_values[1])
    candidate_relative = float(np.exp(log_candidate))
    log_mle_to_candidate = -log_candidate
    log_candidate_to_reference = log_candidate - log_reference

    return SupportComparison(
        candidate_working=candidate,
        reference_working=reference,
        relative_likelihood=candidate_relative,
        log_relative_likelihood=log_candidate,
        likelihood_ratio_mle_to_candidate=_exp_or_none(log_mle_to_candidate),
        log_likelihood_ratio_mle_to_candidate=log_mle_to_candidate,
        likelihood_ratio_candidate_to_reference=_exp_or_none(log_candidate_to_reference),
        log_likelihood_ratio_candidate_to_reference=log_candidate_to_reference,
    )


def _finite_support_endpoint(
    theta_hat: float,
    half_distance: float,
    direction: float,
) -> tuple[float, bool]:
    half_endpoint = (theta_hat * 0.5) + (direction * half_distance)
    if not np.isfinite(half_endpoint) or abs(half_endpoint) > (MAX_FLOAT * 0.5):
        return (MAX_FLOAT if direction > 0 else -MAX_FLOAT), True
    return float(half_endpoint * 2.0), False


def support_interval(
    theta_hat: float,
    se: float,
    *,
    log_relative_likelihood_cutoff: float = S_MINUS_2_SUPPORT_CUTOFF,
) -> SupportInterval:
    """Return the working-scale interval at a log-relative-likelihood cutoff."""

    try:
        center = float(theta_hat)
        standard_error = float(se)
        cutoff = float(log_relative_likelihood_cutoff)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Support interval inputs must be finite.") from exc
    if not all(isfinite(value) for value in (center, standard_error, cutoff)):
        raise ValidationError("Support interval inputs must be finite.")
    if standard_error <= 0:
        raise ValidationError("Standard error must be positive.")
    if cutoff > 0:
        raise ValidationError("Log-relative-likelihood cutoff must be less than or equal to 0.")

    with np.errstate(over="ignore", invalid="ignore"):
        distance = float(np.sqrt(-2.0 * cutoff))
    if not isfinite(distance):
        distance = float(np.sqrt(-cutoff) * np.sqrt(2.0))
    with np.errstate(over="ignore", invalid="ignore"):
        half_distance = (distance * 0.5) * standard_error

    lower_working, lower_clipped = _finite_support_endpoint(
        center,
        half_distance,
        -1.0,
    )
    upper_working, upper_clipped = _finite_support_endpoint(
        center,
        half_distance,
        1.0,
    )
    return SupportInterval(
        support_cutoff=cutoff,
        relative_likelihood_cutoff=float(np.exp(cutoff)),
        likelihood_ratio_mle_to_bound=_exp_or_none(-cutoff),
        lower_working=lower_working,
        upper_working=upper_working,
        lower_clipped=lower_clipped,
        upper_clipped=upper_clipped,
    )


def support_interval_for_ratio(
    theta_hat: float,
    se: float,
    *,
    mle_to_bound_ratio: float,
) -> SupportInterval:
    """Return effects no more than a chosen ratio less supported than the MLE."""

    try:
        ratio = float(mle_to_bound_ratio)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            "MLE-to-bound support ratio must be finite and greater than 1."
        ) from exc
    if not isfinite(ratio) or ratio <= 1.0:
        raise ValidationError("MLE-to-bound support ratio must be finite and greater than 1.")

    return support_interval(
        theta_hat,
        se,
        log_relative_likelihood_cutoff=-float(np.log(ratio)),
    )
