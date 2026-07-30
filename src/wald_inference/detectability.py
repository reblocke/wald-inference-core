"""Exact Wald detectability and legacy closed-form benchmark utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import fsum, isfinite
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.stats import norm

from .errors import ValidationError
from .selection import (
    DEFAULT_CLAIM_DIRECTION,
    DEFAULT_SELECTION_RULE,
    _coerce_finite_float,
    _finite_standardized_distance,
    _interval_probability,
    _pdf_shifted,
    _probability,
    _selected_probability,
    _validate_claim_direction,
    _validate_selection_rule,
    selection_rule_spec,
)
from .types import CriticalEffectResult, SelectionRuleSpec

Z975 = float(norm.ppf(0.975))
Z80 = float(norm.ppf(0.80))
LEGACY_ALPHA = 0.05
LEGACY_POWER = 0.80
MAX_SOLVER_DELTA = 1e6
MAX_BRACKET_STEPS = 80
MAX_BISECTION_STEPS = 200
STABLE_INCREMENT_MAX_ABS_DELTA = 1e-2
STABLE_INCREMENT_MAX_RELATIVE_TARGET = 1e-8
STABLE_COMPLEMENT_MAX_RELATIVE_TARGET = 1e-8
INTEGRATION_ABSOLUTE_TOLERANCE = 1e-18
INTEGRATION_RELATIVE_TOLERANCE = 1e-12

_EXACT_NULL_SELECTION_RULES = {
    "two_sided_p_lt_alpha",
    "one_sided_positive_p_lt_alpha",
    "one_sided_negative_p_lt_alpha",
}

TrueEffectValues = float | Sequence[float] | np.ndarray


def _coerce_true_effect_values(values: object) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValidationError("Detectability true effects must be numeric and finite.")
    try:
        true_effects = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Detectability true effects must be numeric and finite.") from exc
    if true_effects.size == 0:
        raise ValidationError("Detectability true effects must contain at least one value.")
    if not np.isfinite(true_effects).all():
        raise ValidationError("Detectability true effects must be numeric and finite.")
    return true_effects


def _probabilities_for_deltas(spec: SelectionRuleSpec, deltas: np.ndarray) -> np.ndarray:
    probabilities = np.fromiter(
        (_probability_at_delta(spec, float(delta)) for delta in deltas.flat),
        dtype=float,
        count=deltas.size,
    ).reshape(deltas.shape)
    if not np.isfinite(probabilities).all():
        raise ValidationError("Selected-claim probability exceeds the finite floating-point range.")
    return probabilities


def selected_claim_probability(
    true_effect_working: TrueEffectValues,
    *,
    null_working: float,
    standard_error: float,
    alpha: float = 0.05,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    claim_direction: str = DEFAULT_CLAIM_DIRECTION,
    threshold_working: float | None = None,
) -> float | np.ndarray:
    """Return exact selected-claim probability under a future Wald model.

    Scalar input returns a Python ``float``. Array-like input returns an array
    with the same shape. All six canonical selection rules are supported.
    """

    true_effects = _coerce_true_effect_values(true_effect_working)
    null_value = _coerce_finite_float(null_working, label="Detectability null value")
    se_value = _coerce_finite_float(
        standard_error,
        label="Detectability standard error",
    )
    spec = selection_rule_spec(
        selection_rule=selection_rule,
        alpha=alpha,
        null_working=null_value,
        se=se_value,
        claim_direction=claim_direction,
        threshold_working=threshold_working,
    )
    deltas = _finite_standardized_distance(
        true_effects,
        center=null_value,
        scale=se_value,
    )
    probabilities = _probabilities_for_deltas(spec, deltas)
    if true_effects.ndim == 0:
        return float(probabilities.item())
    return probabilities


def power_curve(
    true_effects_working: Sequence[float] | np.ndarray,
    *,
    null_working: float,
    standard_error: float,
    alpha: float = 0.05,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    claim_direction: str = DEFAULT_CLAIM_DIRECTION,
    threshold_working: float | None = None,
) -> np.ndarray:
    """Return an exact one-dimensional selected-claim probability curve."""

    true_effects = _coerce_true_effect_values(true_effects_working)
    if true_effects.ndim != 1:
        raise ValidationError(
            "Power-curve true effects must be a one-dimensional numeric sequence."
        )
    probabilities = selected_claim_probability(
        true_effects,
        null_working=null_working,
        standard_error=standard_error,
        alpha=alpha,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=threshold_working,
    )
    assert isinstance(probabilities, np.ndarray)
    return probabilities


def _probability_at_delta(spec: SelectionRuleSpec, delta: float) -> float:
    if spec.key in _EXACT_NULL_SELECTION_RULES:
        if delta == 0.0:
            return spec.alpha
        if abs(delta) <= STABLE_INCREMENT_MAX_ABS_DELTA:
            return _probability(spec.alpha + _probability_increment_from_null(spec, delta))
    return _selected_probability(spec.intervals, delta)


def _scaled_probability_derivative(spec: SelectionRuleSpec, delta: float) -> float:
    return float(
        sum(
            (_pdf_shifted(lower, delta) / spec.alpha) - (_pdf_shifted(upper, delta) / spec.alpha)
            for lower, upper in spec.intervals
        )
    )


def _probability_increment_from_null(
    spec: SelectionRuleSpec,
    delta: float,
) -> float:
    if spec.key not in _EXACT_NULL_SELECTION_RULES:
        raise ValidationError("Stable probability increments require a p-value selection rule.")
    if delta == 0.0:
        return 0.0
    if abs(delta) > STABLE_INCREMENT_MAX_ABS_DELTA:
        raise ValidationError("Stable probability increment requested outside the near-null range.")
    try:
        with catch_warnings():
            simplefilter("error", IntegrationWarning)
            scaled_increment, _ = quad(
                lambda shifted_delta: _scaled_probability_derivative(
                    spec,
                    shifted_delta,
                ),
                0.0,
                delta,
                epsabs=INTEGRATION_ABSOLUTE_TOLERANCE,
                epsrel=INTEGRATION_RELATIVE_TOLERANCE,
                limit=100,
            )
    except (IntegrationWarning, OverflowError, ValueError) as exc:
        raise ValidationError(
            "Could not evaluate a stable near-null probability increment."
        ) from exc
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        increment = float(np.float64(scaled_increment) * np.float64(spec.alpha))
    if not isfinite(increment):
        raise ValidationError(
            "Near-null probability increment exceeds the finite floating-point range."
        )
    return increment


def _bracket_and_bisect_magnitude(
    *,
    is_satisfied: Callable[[float], bool],
    initial_high: float = 1.0,
) -> float:
    low = 0.0
    high = initial_high
    for _ in range(MAX_BRACKET_STEPS):
        if is_satisfied(high):
            break
        high *= 2.0
        if not np.isfinite(high) or high > MAX_SOLVER_DELTA:
            raise ValidationError(
                "Could not bracket a finite critical effect for the requested probability."
            )
    else:
        raise ValidationError(
            "Could not bracket a finite critical effect for the requested probability."
        )

    for _ in range(MAX_BISECTION_STEPS):
        midpoint = low + ((high - low) / 2.0)
        if midpoint == low or midpoint == high:
            break
        if is_satisfied(midpoint):
            high = midpoint
        else:
            low = midpoint
    if not is_satisfied(high):
        raise ValidationError(
            "Could not solve a finite critical effect for the requested probability."
        )
    preceding = float(np.nextafter(high, 0.0))
    if is_satisfied(preceding):
        raise ValidationError(
            "Could not identify the smallest finite critical effect at floating-point precision."
        )
    return high


def _is_near_null_target(
    *,
    spec: SelectionRuleSpec,
    target_probability: float,
) -> bool:
    target_increment = target_probability - spec.alpha
    return target_increment <= (STABLE_INCREMENT_MAX_RELATIVE_TARGET * spec.alpha)


def _is_near_one_target(
    *,
    spec: SelectionRuleSpec,
    target_probability: float,
) -> bool:
    return (1.0 - target_probability) <= (
        STABLE_COMPLEMENT_MAX_RELATIVE_TARGET * (1.0 - spec.alpha)
    )


def _one_sided_critical_z(spec: SelectionRuleSpec) -> float:
    if spec.key == "one_sided_positive_p_lt_alpha":
        return spec.intervals[0][0]
    return -spec.intervals[0][1]


def _one_sided_unselected_probability(
    spec: SelectionRuleSpec,
    magnitude: float,
) -> float:
    shifted_boundary = _one_sided_critical_z(spec) - magnitude
    return _probability(float(norm.cdf(shifted_boundary)))


def _one_sided_probability_from_quantile(
    *,
    target_probability: float,
    target_quantile: float,
    shifted_boundary: float,
) -> float:
    if shifted_boundary > target_quantile:
        raise ValidationError(
            "Could not certify the one-sided critical effect in the quantile domain."
        )
    if shifted_boundary == target_quantile:
        return target_probability
    try:
        with catch_warnings():
            simplefilter("error", IntegrationWarning)
            probability_increment, _ = quad(
                norm.pdf,
                shifted_boundary,
                target_quantile,
                epsabs=INTEGRATION_ABSOLUTE_TOLERANCE,
                epsrel=INTEGRATION_RELATIVE_TOLERANCE,
                limit=100,
            )
    except (IntegrationWarning, OverflowError, ValueError) as exc:
        raise ValidationError(
            "Could not certify the one-sided critical effect probability."
        ) from exc
    achieved_probability = fsum((target_probability, probability_increment))
    if not isfinite(achieved_probability):
        raise ValidationError(
            "Critical-effect probability exceeds the finite floating-point range."
        )
    return _probability(achieved_probability)


def _one_sided_critical_solution(
    *,
    spec: SelectionRuleSpec,
    target_probability: float,
) -> tuple[float, float]:
    target_increment = target_probability - spec.alpha
    direction_sign = 1.0 if spec.claim_direction == "positive" else -1.0
    if _is_near_null_target(
        spec=spec,
        target_probability=target_probability,
    ):
        magnitude = _bracket_and_bisect_magnitude(
            is_satisfied=lambda magnitude: (
                _probability_increment_from_null(
                    spec,
                    direction_sign * magnitude,
                )
                >= target_increment
            ),
            initial_high=STABLE_INCREMENT_MAX_ABS_DELTA,
        )
        achieved_probability = _probability(
            fsum(
                (
                    spec.alpha,
                    _probability_increment_from_null(
                        spec,
                        direction_sign * magnitude,
                    ),
                )
            )
        )
        return magnitude, achieved_probability

    if _is_near_one_target(
        spec=spec,
        target_probability=target_probability,
    ):
        target_unselected_probability = 1.0 - target_probability
        magnitude = _bracket_and_bisect_magnitude(
            is_satisfied=lambda candidate: (
                _one_sided_unselected_probability(
                    spec,
                    candidate,
                )
                <= target_unselected_probability
            )
        )
        achieved_probability = _probability(
            1.0 - _one_sided_unselected_probability(spec, magnitude)
        )
        return magnitude, achieved_probability

    critical_z = _one_sided_critical_z(spec)
    target_quantile = float(norm.isf(target_probability))
    with np.errstate(over="ignore", invalid="ignore"):
        initial_high = float(critical_z - target_quantile)
    if not isfinite(initial_high) or initial_high <= 0:
        raise ValidationError(
            "Could not solve a finite critical effect for the requested probability."
        )
    magnitude = _bracket_and_bisect_magnitude(
        is_satisfied=lambda candidate: float(critical_z - candidate) <= target_quantile,
        initial_high=initial_high,
    )
    shifted_boundary = float(critical_z - magnitude)
    achieved_probability = _one_sided_probability_from_quantile(
        target_probability=target_probability,
        target_quantile=target_quantile,
        shifted_boundary=shifted_boundary,
    )
    return magnitude, achieved_probability


def _two_sided_unselected_probability(
    spec: SelectionRuleSpec,
    magnitude: float,
) -> float:
    lower = spec.intervals[0][1]
    upper = spec.intervals[1][0]
    return _probability(_interval_probability(lower, upper, magnitude))


def _solve_magnitude_for_probability(
    *,
    spec: SelectionRuleSpec,
    direction_sign: float,
    target_probability: float,
) -> tuple[float, float]:
    if spec.key != "two_sided_p_lt_alpha":
        magnitude, achieved_probability = _one_sided_critical_solution(
            spec=spec,
            target_probability=target_probability,
        )
    elif _is_near_null_target(
        spec=spec,
        target_probability=target_probability,
    ):
        target_increment = target_probability - spec.alpha
        magnitude = _bracket_and_bisect_magnitude(
            is_satisfied=lambda candidate: (
                _probability_increment_from_null(
                    spec,
                    direction_sign * candidate,
                )
                >= target_increment
            ),
            initial_high=STABLE_INCREMENT_MAX_ABS_DELTA,
        )
        achieved_probability = _probability(
            fsum(
                (
                    spec.alpha,
                    _probability_increment_from_null(
                        spec,
                        direction_sign * magnitude,
                    ),
                )
            )
        )
    elif _is_near_one_target(
        spec=spec,
        target_probability=target_probability,
    ):
        target_unselected_probability = 1.0 - target_probability
        magnitude = _bracket_and_bisect_magnitude(
            is_satisfied=lambda candidate: (
                _two_sided_unselected_probability(
                    spec,
                    candidate,
                )
                <= target_unselected_probability
            )
        )
        achieved_probability = _probability(
            1.0 - _two_sided_unselected_probability(spec, magnitude)
        )
    else:
        magnitude = _bracket_and_bisect_magnitude(
            is_satisfied=lambda candidate: (
                _probability_at_delta(
                    spec,
                    direction_sign * candidate,
                )
                >= target_probability
            )
        )
        achieved_probability = _probability_at_delta(
            spec,
            direction_sign * magnitude,
        )

    if achieved_probability < target_probability:
        raise ValidationError(
            "Could not certify the critical effect probability at floating-point precision."
        )
    return magnitude, achieved_probability


def _critical_effect_working(
    *,
    null_working: float,
    standard_error: float,
    critical_delta: float,
) -> float:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        displacement = float(np.float64(critical_delta) * np.float64(standard_error))
        critical_effect = float(np.float64(null_working) + np.float64(displacement))
    if not isfinite(displacement) or not isfinite(critical_effect):
        raise ValidationError(
            "Critical effect exceeds the finite floating-point range on the working scale."
        )
    represented_delta = float(
        _finite_standardized_distance(
            np.asarray([critical_effect]),
            center=null_working,
            scale=standard_error,
        )[0]
    )
    if not np.isclose(
        represented_delta,
        critical_delta,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValidationError(
            "Critical effect cannot be represented accurately on the working scale."
        )
    return critical_effect


def critical_effect_for_target_probability(
    *,
    null_working: float,
    standard_error: float,
    alpha: float = 0.05,
    target_probability: float = 0.80,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    claim_direction: str = DEFAULT_CLAIM_DIRECTION,
) -> CriticalEffectResult:
    """Return the smallest directed effect meeting an exact probability target.

    Inversion is limited to two-sided p-value selection and the matching
    positive or negative one-sided p-value selection rules.
    """

    null_value = _coerce_finite_float(null_working, label="Detectability null value")
    se_value = _coerce_finite_float(
        standard_error,
        label="Detectability standard error",
    )
    target = _coerce_finite_float(
        target_probability,
        label="Detectability target probability",
    )
    if target <= 0 or target >= 1:
        raise ValidationError(
            "Detectability target probability must be finite and between 0 and 1."
        )
    supported_directions = {
        "two_sided_p_lt_alpha": ("positive", "negative"),
        "one_sided_positive_p_lt_alpha": ("positive",),
        "one_sided_negative_p_lt_alpha": ("negative",),
    }
    rule = _validate_selection_rule(selection_rule)
    direction = _validate_claim_direction(claim_direction)
    allowed_directions = supported_directions.get(rule)
    if allowed_directions is None:
        raise ValidationError(
            "Critical-effect inversion supports only two-sided, one-sided positive, "
            "and one-sided negative p-value selection rules."
        )
    if direction not in allowed_directions:
        expected = allowed_directions[0]
        raise ValidationError(
            f"{rule!r} critical-effect inversion requires {expected!r} claim direction."
        )
    spec = selection_rule_spec(
        selection_rule=rule,
        alpha=alpha,
        null_working=null_value,
        se=se_value,
        claim_direction=direction,
    )

    null_probability = _probability_at_delta(spec, 0.0)
    if target <= spec.alpha:
        critical_delta = 0.0
        critical_effect = null_value
        achieved_probability = null_probability
    else:
        direction_sign = 1.0 if spec.claim_direction == "positive" else -1.0
        magnitude, achieved_probability = _solve_magnitude_for_probability(
            spec=spec,
            direction_sign=direction_sign,
            target_probability=target,
        )
        critical_delta = direction_sign * magnitude
        critical_effect = _critical_effect_working(
            null_working=null_value,
            standard_error=se_value,
            critical_delta=critical_delta,
        )

    return CriticalEffectResult(
        selection_rule=spec.key,
        claim_direction=spec.claim_direction,
        alpha=spec.alpha,
        target_probability=target,
        null_working=null_value,
        standard_error=se_value,
        critical_delta=critical_delta,
        critical_effect_working=critical_effect,
        achieved_probability=achieved_probability,
    )


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
