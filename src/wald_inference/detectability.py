"""Exact Wald detectability and legacy closed-form benchmark utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from math import exp, isfinite, log1p, nextafter, sinh, sqrt

import numpy as np
from scipy.stats import norm

from .errors import ValidationError
from .selection import (
    DEFAULT_CLAIM_DIRECTION,
    DEFAULT_SELECTION_RULE,
    _coerce_finite_float,
    _finite_standardized_distance,
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
STABLE_INCREMENT_MAX_ABS_DELTA = 0.125
NUMERICAL_GUARD_ULPS = 8
_INVERSE_SQRT_TWO_PI = 1.0 / sqrt(2.0 * np.pi)
_MAX_FLOAT_EXACT = Fraction.from_float(float(np.finfo(float).max))
_GAUSS_16_NODES, _GAUSS_16_WEIGHTS = np.polynomial.legendre.leggauss(16)
_GAUSS_32_NODES, _GAUSS_32_WEIGHTS = np.polynomial.legendre.leggauss(32)

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
    if spec.key == "two_sided_p_lt_alpha":
        return _two_sided_probability(spec, delta)
    if spec.key in {
        "one_sided_positive_p_lt_alpha",
        "one_sided_negative_p_lt_alpha",
    }:
        return _one_sided_probability(spec, delta)
    return _selected_probability(spec.intervals, delta)


def _step_toward(value: float, direction: float, *, steps: int) -> float:
    stepped = value
    for _ in range(steps):
        stepped = nextafter(stepped, direction)
    return stepped


def _floor_probability(exact_probability: Fraction) -> float:
    if exact_probability <= 0:
        return 0.0
    if exact_probability >= 1:
        return 1.0
    probability = float(exact_probability)
    if Fraction.from_float(probability) > exact_probability:
        probability = nextafter(probability, 0.0)
    return probability


def _probability_from_null(
    alpha: float,
    increment: float,
    *,
    increasing: bool,
) -> float:
    alpha_exact = Fraction.from_float(alpha)
    increment_exact = Fraction.from_float(increment)
    exact_probability = (
        alpha_exact + increment_exact if increasing else alpha_exact - increment_exact
    )
    return _floor_probability(exact_probability)


def _lower_probability_increment(first: float, second: float) -> float:
    increment = min(first, second)
    if increment <= 0.0:
        return 0.0
    guarded = increment * (1.0 - (NUMERICAL_GUARD_ULPS * np.finfo(float).eps))
    return _step_toward(guarded, 0.0, steps=NUMERICAL_GUARD_ULPS)


def _normal_pdf(value: float) -> float:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return _INVERSE_SQRT_TWO_PI * exp(-0.5 * value * value)


def _one_sided_increment_quadrature(
    critical_z: float,
    magnitude: float,
    *,
    increasing: bool,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    half_magnitude = 0.5 * magnitude
    weighted_density = 0.0
    for node, weight in zip(nodes, weights, strict=True):
        offset = half_magnitude * (float(node) + 1.0)
        argument = critical_z - offset if increasing else critical_z + offset
        weighted_density += float(weight) * _normal_pdf(argument)
    return half_magnitude * weighted_density


def _two_sided_increment_quadrature(
    critical_z: float,
    magnitude: float,
    *,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    half_magnitude = 0.5 * magnitude
    density_at_boundary = _normal_pdf(critical_z)
    weighted_derivative = 0.0
    for node, weight in zip(nodes, weights, strict=True):
        offset = half_magnitude * (float(node) + 1.0)
        derivative = (
            2.0 * density_at_boundary * exp(-0.5 * offset * offset) * sinh(critical_z * offset)
        )
        weighted_derivative += float(weight) * derivative
    return half_magnitude * weighted_derivative


def _stable_one_sided_increment(
    critical_z: float,
    magnitude: float,
    *,
    increasing: bool,
) -> float:
    first = _one_sided_increment_quadrature(
        critical_z,
        magnitude,
        increasing=increasing,
        nodes=_GAUSS_16_NODES,
        weights=_GAUSS_16_WEIGHTS,
    )
    second = _one_sided_increment_quadrature(
        critical_z,
        magnitude,
        increasing=increasing,
        nodes=_GAUSS_32_NODES,
        weights=_GAUSS_32_WEIGHTS,
    )
    return _lower_probability_increment(first, second)


def _stable_two_sided_increment(
    critical_z: float,
    magnitude: float,
) -> float:
    first = _two_sided_increment_quadrature(
        critical_z,
        magnitude,
        nodes=_GAUSS_16_NODES,
        weights=_GAUSS_16_WEIGHTS,
    )
    second = _two_sided_increment_quadrature(
        critical_z,
        magnitude,
        nodes=_GAUSS_32_NODES,
        weights=_GAUSS_32_WEIGHTS,
    )
    return _lower_probability_increment(first, second)


def _lower_probability_from_complement(complement: float) -> float:
    if complement <= 0.0:
        return nextafter(1.0, 0.0)
    conservative_complement = _step_toward(
        complement,
        float("inf"),
        steps=NUMERICAL_GUARD_ULPS,
    )
    return _floor_probability(Fraction(1, 1) - Fraction.from_float(conservative_complement))


def _direct_one_sided_probability(critical_z: float, directed_delta: float) -> float:
    selected_z = directed_delta - critical_z
    if selected_z <= 0.0:
        probability = float(norm.cdf(selected_z))
        if probability <= 0.0:
            return 0.0
        return _step_toward(
            probability,
            0.0,
            steps=NUMERICAL_GUARD_ULPS,
        )
    complement = float(norm.cdf(-selected_z))
    return _lower_probability_from_complement(complement)


def _one_sided_probability(spec: SelectionRuleSpec, delta: float) -> float:
    directed_delta = delta if spec.key == "one_sided_positive_p_lt_alpha" else -delta
    if directed_delta == 0.0:
        return spec.alpha

    critical_z = (
        spec.intervals[0][0]
        if spec.key == "one_sided_positive_p_lt_alpha"
        else -spec.intervals[0][1]
    )
    magnitude = abs(directed_delta)
    if magnitude <= STABLE_INCREMENT_MAX_ABS_DELTA:
        increasing = directed_delta > 0.0
        increment = _stable_one_sided_increment(
            critical_z,
            magnitude,
            increasing=increasing,
        )
        if not increasing and increment > 0.0:
            increment = _step_toward(
                increment,
                float("inf"),
                steps=NUMERICAL_GUARD_ULPS,
            )
        return _probability_from_null(
            spec.alpha,
            increment,
            increasing=increasing,
        )

    boundary_directed_delta = (
        STABLE_INCREMENT_MAX_ABS_DELTA if directed_delta > 0.0 else -STABLE_INCREMENT_MAX_ABS_DELTA
    )
    boundary_delta = (
        boundary_directed_delta
        if spec.key == "one_sided_positive_p_lt_alpha"
        else -boundary_directed_delta
    )
    boundary_probability = _one_sided_probability(spec, boundary_delta)
    direct_probability = _direct_one_sided_probability(
        critical_z,
        directed_delta,
    )
    if directed_delta > 0.0:
        return max(boundary_probability, direct_probability)
    return min(boundary_probability, direct_probability)


def _central_interval_probability(
    critical_z: float,
    magnitude: float,
) -> float:
    if critical_z <= 0.75:
        weighted_density = 0.0
        for node, weight in zip(
            _GAUSS_32_NODES,
            _GAUSS_32_WEIGHTS,
            strict=True,
        ):
            centered_value = (critical_z * float(node)) - magnitude
            weighted_density += float(weight) * _normal_pdf(centered_value)
        probability = critical_z * weighted_density
    else:
        upper_log_cdf = float(norm.logcdf(critical_z - magnitude))
        lower_log_cdf = float(norm.logcdf(-critical_z - magnitude))
        if lower_log_cdf == float("-inf"):
            probability = exp(upper_log_cdf)
        else:
            ratio = exp(lower_log_cdf - upper_log_cdf)
            if ratio >= 1.0:
                probability = 0.0
            else:
                probability = exp(upper_log_cdf + log1p(-ratio))
    return min(1.0, max(0.0, probability))


def _two_sided_probability(spec: SelectionRuleSpec, delta: float) -> float:
    magnitude = abs(delta)
    if magnitude == 0.0:
        return spec.alpha

    critical_z = spec.intervals[1][0]
    if magnitude <= STABLE_INCREMENT_MAX_ABS_DELTA:
        increment = _stable_two_sided_increment(
            critical_z,
            magnitude,
        )
        return _probability_from_null(
            spec.alpha,
            increment,
            increasing=True,
        )

    boundary_probability = _two_sided_probability(
        spec,
        STABLE_INCREMENT_MAX_ABS_DELTA,
    )
    complement = _central_interval_probability(critical_z, magnitude)
    direct_probability = _lower_probability_from_complement(complement)
    return max(boundary_probability, direct_probability)


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


def _solve_magnitude_for_probability(
    *,
    spec: SelectionRuleSpec,
    direction_sign: float,
    target_probability: float,
) -> float:
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
    preceding_probability = _probability_at_delta(
        spec,
        direction_sign * nextafter(magnitude, 0.0),
    )
    if achieved_probability < target_probability or preceding_probability >= target_probability:
        raise ValidationError(
            "Could not certify the smallest finite critical effect at floating-point precision."
        )
    return magnitude


def _compose_working_effect(
    *,
    null_working: float,
    standard_error: float,
    critical_delta: float,
) -> float:
    exact_effect = Fraction.from_float(null_working) + (
        Fraction.from_float(critical_delta) * Fraction.from_float(standard_error)
    )
    if exact_effect < -_MAX_FLOAT_EXACT or exact_effect > _MAX_FLOAT_EXACT:
        raise ValidationError(
            "Critical effect exceeds the finite floating-point range on the working scale."
        )
    return float(exact_effect)


def _probability_for_working_effect(
    effect_working: float,
    *,
    spec: SelectionRuleSpec,
    null_working: float,
    standard_error: float,
) -> tuple[float, float]:
    represented_delta = float(
        _finite_standardized_distance(
            np.asarray([effect_working]),
            center=null_working,
            scale=standard_error,
        )[0]
    )
    return represented_delta, _probability_at_delta(spec, represented_delta)


def _representable_critical_effect(
    *,
    spec: SelectionRuleSpec,
    null_working: float,
    standard_error: float,
    critical_delta: float,
    target_probability: float,
) -> tuple[float, float, float]:
    direction = float("inf") if critical_delta > 0.0 else float("-inf")
    effect_working = _compose_working_effect(
        null_working=null_working,
        standard_error=standard_error,
        critical_delta=critical_delta,
    )
    if effect_working == null_working:
        raise ValidationError(
            "Critical effect cannot be represented accurately on the working scale."
        )
    if not isfinite(effect_working):
        raise ValidationError(
            "Critical effect exceeds the finite floating-point range on the working scale."
        )

    represented_delta, achieved_probability = _probability_for_working_effect(
        effect_working,
        spec=spec,
        null_working=null_working,
        standard_error=standard_error,
    )
    for _ in range(4):
        if achieved_probability >= target_probability:
            break
        effect_working = nextafter(effect_working, direction)
        if not isfinite(effect_working):
            raise ValidationError(
                "Critical effect exceeds the finite floating-point range on the working scale."
            )
        represented_delta, achieved_probability = _probability_for_working_effect(
            effect_working,
            spec=spec,
            null_working=null_working,
            standard_error=standard_error,
        )
    else:
        raise ValidationError(
            "Critical effect cannot be represented accurately on the working scale."
        )

    for _ in range(4):
        preceding_effect = nextafter(effect_working, null_working)
        if preceding_effect == effect_working:
            break
        preceding_delta, preceding_probability = _probability_for_working_effect(
            preceding_effect,
            spec=spec,
            null_working=null_working,
            standard_error=standard_error,
        )
        if preceding_probability < target_probability:
            break
        effect_working = preceding_effect
        represented_delta = preceding_delta
        achieved_probability = preceding_probability
    else:
        raise ValidationError(
            "Critical effect cannot be represented accurately on the working scale."
        )

    if (
        effect_working == null_working
        or (critical_delta > 0.0 and represented_delta <= 0.0)
        or (critical_delta < 0.0 and represented_delta >= 0.0)
        or abs(represented_delta - critical_delta) > (abs(critical_delta) * 1e-12)
    ):
        raise ValidationError(
            "Critical effect cannot be represented accurately on the working scale."
        )
    return represented_delta, effect_working, achieved_probability


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
        magnitude = _solve_magnitude_for_probability(
            spec=spec,
            direction_sign=direction_sign,
            target_probability=target,
        )
        solved_delta = direction_sign * magnitude
        critical_delta, critical_effect, achieved_probability = _representable_critical_effect(
            spec=spec,
            null_working=null_value,
            standard_error=se_value,
            critical_delta=solved_delta,
            target_probability=target,
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
