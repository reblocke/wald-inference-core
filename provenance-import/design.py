from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, get_args

import numpy as np
from scipy.stats import norm

from .core import ValidationError, standardized_distance

SelectionRule: TypeAlias = Literal[
    "two_sided_p_lt_alpha",
    "one_sided_positive_p_lt_alpha",
    "one_sided_negative_p_lt_alpha",
    "ci_excludes_null_in_beneficial_direction",
    "estimate_exceeds_mcid_and_p_lt_alpha",
    "ci_excludes_mcid",
]
ClaimDirection: TypeAlias = Literal["positive", "negative"]
DEFAULT_SELECTION_RULE: SelectionRule = "two_sided_p_lt_alpha"
DEFAULT_CLAIM_DIRECTION: ClaimDirection = "positive"
DEFAULT_NEAR_NULL_DELTA = 1e-12
DEFAULT_SOLVER_TOLERANCE = 1e-8
MAX_INFORMATION_MULTIPLIER = 1e12

_SELECTION_RULE_LABELS: dict[SelectionRule, str] = {
    "two_sided_p_lt_alpha": "Two-sided p < alpha against the null",
    "one_sided_positive_p_lt_alpha": "One-sided positive p < alpha against the null",
    "one_sided_negative_p_lt_alpha": "One-sided negative p < alpha against the null",
    "ci_excludes_null_in_beneficial_direction": (
        "CI at selected alpha excludes the null in the selected claim direction"
    ),
    "estimate_exceeds_mcid_and_p_lt_alpha": (
        "Estimate exceeds the claim threshold and two-sided p < alpha"
    ),
    "ci_excludes_mcid": "CI at selected alpha excludes the claim threshold",
}


@dataclass(frozen=True)
class SelectionRuleSpec:
    """Selected-claim rule represented as intervals on the future Wald Z scale."""

    key: SelectionRule
    label: str
    alpha: float
    claim_direction: ClaimDirection
    threshold_working: float | None
    threshold_delta: float | None
    intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DesignMetric:
    """Repeated-study design metric for one assumed true effect."""

    true_effect_working: float
    delta: float
    power: float
    type_s: float | None
    type_m: float | None
    expected_selected_abs_z: float | None
    observed_exaggeration: float | None


@dataclass(frozen=True)
class PrecisionTargetResult:
    """Required precision for one requested design target."""

    target: str
    requested_value: float
    required_se: float | None
    required_information_multiplier: float | None
    approx_95_ci_width_working: float | None
    achieved_power: float | None
    achieved_type_s: float | None
    achieved_type_m: float | None
    note: str


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or alpha <= 0 or alpha >= 1:
        raise ValidationError("Design alpha must be finite and between 0 and 1.")


def _coerce_finite_float(value: object, *, label: str) -> float:
    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be finite.") from exc
    if not np.isfinite(float_value):
        raise ValidationError(f"{label} must be finite.")
    return float_value


def _coerce_true_effect_array(values: object) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValidationError("Design true effects must be supplied as numeric values.")
    try:
        true_effects = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Design true effects must be finite.") from exc
    if true_effects.ndim == 0:
        raise ValidationError("Design true effects must be supplied as numeric values.")
    if not np.isfinite(true_effects).all():
        raise ValidationError("Design true effects must be finite.")
    return true_effects


def _finite_standardized_distance(
    values: np.ndarray,
    *,
    center: float,
    scale: float,
) -> np.ndarray:
    """Return ``(values - center) / scale`` without emitting non-finite values."""

    try:
        return standardized_distance(values, theta_hat=center, se=scale)
    except ValidationError as exc:
        raise ValidationError(
            "Design standardized distance exceeds the finite floating-point range."
        ) from exc


def _validate_se(se: float, *, label: str = "Design standard error") -> None:
    if not np.isfinite(se) or se <= 0:
        raise ValidationError(f"{label} must be finite and positive.")


def _validate_selection_rule(selection_rule: str) -> SelectionRule:
    valid_rules = get_args(SelectionRule)
    if selection_rule not in valid_rules:
        valid = ", ".join(valid_rules)
        raise ValidationError(
            f"Unsupported design selection rule {selection_rule!r}. Expected one of: {valid}."
        )
    return selection_rule  # type: ignore[return-value]


def _validate_claim_direction(claim_direction: str) -> ClaimDirection:
    if claim_direction not in get_args(ClaimDirection):
        raise ValidationError("Design claim direction must be 'positive' or 'negative'.")
    return claim_direction  # type: ignore[return-value]


def _probability(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _critical_z_for_tail_probability(tail_probability: float) -> float:
    if tail_probability <= 0 or not np.isfinite(tail_probability):
        raise ValidationError(
            "Design alpha is too small to evaluate with finite floating-point precision."
        )
    critical_z = float(norm.isf(tail_probability))
    if not np.isfinite(critical_z) or norm.sf(critical_z) == 0.0:
        raise ValidationError(
            "Design alpha is too small to evaluate with finite floating-point precision."
        )
    return critical_z


def _two_sided_critical_z(alpha: float) -> float:
    return _critical_z_for_tail_probability(float(alpha) / 2.0)


def _one_sided_critical_z(alpha: float) -> float:
    return _critical_z_for_tail_probability(float(alpha))


def _requires_threshold(selection_rule: SelectionRule) -> bool:
    return selection_rule in {"estimate_exceeds_mcid_and_p_lt_alpha", "ci_excludes_mcid"}


def selection_rule_spec(
    *,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    alpha: float = 0.05,
    null_working: float = 0.0,
    se: float = 1.0,
    claim_direction: str = DEFAULT_CLAIM_DIRECTION,
    threshold_working: float | None = None,
) -> SelectionRuleSpec:
    """Build a deterministic selected-claim rule on the future Wald Z scale."""

    rule = _validate_selection_rule(selection_rule)
    direction = _validate_claim_direction(claim_direction)
    alpha_value = _coerce_finite_float(alpha, label="Design alpha")
    se_value = _coerce_finite_float(se, label="Design standard error")
    null_value = _coerce_finite_float(null_working, label="Design null value")
    _validate_alpha(alpha_value)
    _validate_se(se_value)

    threshold_delta: float | None = None
    threshold_value: float | None = None
    if _requires_threshold(rule):
        if threshold_working is None:
            raise ValidationError("Design claim threshold is required for this selection rule.")
        threshold_value = _coerce_finite_float(
            threshold_working,
            label="Design claim threshold",
        )
        threshold_delta = float(
            _finite_standardized_distance(
                np.asarray([threshold_value]),
                center=null_value,
                scale=se_value,
            )[0]
        )
        if direction == "positive" and threshold_delta <= 0:
            raise ValidationError(
                "Positive-claim threshold rules require a threshold above the null."
            )
        if direction == "negative" and threshold_delta >= 0:
            raise ValidationError(
                "Negative-claim threshold rules require a threshold below the null."
            )
    elif threshold_working is not None:
        threshold_value = _coerce_finite_float(
            threshold_working,
            label="Design claim threshold",
        )

    two_sided_z = _two_sided_critical_z(alpha_value)
    one_sided_z = _one_sided_critical_z(alpha_value)
    neg_inf = float("-inf")
    pos_inf = float("inf")

    if rule == "two_sided_p_lt_alpha":
        intervals = ((neg_inf, -two_sided_z), (two_sided_z, pos_inf))
    elif rule == "one_sided_positive_p_lt_alpha":
        intervals = ((one_sided_z, pos_inf),)
    elif rule == "one_sided_negative_p_lt_alpha":
        intervals = ((neg_inf, -one_sided_z),)
    elif rule == "ci_excludes_null_in_beneficial_direction":
        intervals = (
            ((two_sided_z, pos_inf),) if direction == "positive" else ((neg_inf, -two_sided_z),)
        )
    elif rule == "estimate_exceeds_mcid_and_p_lt_alpha":
        assert threshold_delta is not None
        intervals = (
            ((max(two_sided_z, threshold_delta), pos_inf),)
            if direction == "positive"
            else ((neg_inf, min(-two_sided_z, threshold_delta)),)
        )
    else:
        assert rule == "ci_excludes_mcid"
        assert threshold_delta is not None
        intervals = (
            ((threshold_delta + two_sided_z, pos_inf),)
            if direction == "positive"
            else ((neg_inf, threshold_delta - two_sided_z),)
        )

    return SelectionRuleSpec(
        key=rule,
        label=_SELECTION_RULE_LABELS[rule],
        alpha=alpha_value,
        claim_direction=direction,
        threshold_working=threshold_value,
        threshold_delta=threshold_delta,
        intervals=intervals,
    )


def _interval_probability(lower: float, upper: float, delta: float) -> float:
    if lower == float("-inf") and upper == float("inf"):
        return 1.0
    if lower == float("-inf"):
        return float(norm.cdf(upper - delta))
    if upper == float("inf"):
        return float(norm.sf(lower - delta))
    return float(norm.cdf(upper - delta) - norm.cdf(lower - delta))


def _pdf_shifted(value: float, delta: float) -> float:
    if not np.isfinite(value):
        return 0.0
    with np.errstate(over="ignore", under="ignore"):
        return float(norm.pdf(value - delta))


def _interval_z_numerator(lower: float, upper: float, delta: float) -> float:
    probability = _interval_probability(lower, upper, delta)
    return float(delta * probability + _pdf_shifted(lower, delta) - _pdf_shifted(upper, delta))


def _interval_abs_z_numerator(lower: float, upper: float, delta: float) -> float:
    if upper <= 0:
        return -_interval_z_numerator(lower, upper, delta)
    if lower >= 0:
        return _interval_z_numerator(lower, upper, delta)
    return -_interval_z_numerator(lower, 0.0, delta) + _interval_z_numerator(0.0, upper, delta)


def _intersect_interval(
    interval: tuple[float, float],
    mask: tuple[float, float],
) -> tuple[float, float] | None:
    lower = max(interval[0], mask[0])
    upper = min(interval[1], mask[1])
    if lower >= upper:
        return None
    return lower, upper


def _wrong_sign_intervals(
    spec: SelectionRuleSpec,
    delta: float,
) -> tuple[tuple[float, float], ...]:
    if delta > 0:
        mask = (float("-inf"), 0.0)
    else:
        mask = (0.0, float("inf"))
    intervals = [
        intersection
        for interval in spec.intervals
        if (intersection := _intersect_interval(interval, mask)) is not None
    ]
    return tuple(intervals)


def _selected_probability(intervals: tuple[tuple[float, float], ...], delta: float) -> float:
    return _probability(
        sum(_interval_probability(lower, upper, delta) for lower, upper in intervals)
    )


def _selected_abs_z_numerator(intervals: tuple[tuple[float, float], ...], delta: float) -> float:
    return float(sum(_interval_abs_z_numerator(lower, upper, delta) for lower, upper in intervals))


def _validate_design_inputs(
    true_effects_working: np.ndarray,
    *,
    null_working: float,
    se: float,
    estimate_working: float | None,
    alpha: float,
    selection_rule: str,
    claim_direction: str,
    threshold_working: float | None,
    near_null_delta: float,
) -> SelectionRuleSpec:
    if not np.isfinite(near_null_delta) or near_null_delta < 0:
        raise ValidationError("Design near-null delta tolerance must be finite and nonnegative.")
    if estimate_working is not None and not np.isfinite(estimate_working):
        raise ValidationError("Design estimate must be finite on the working scale.")
    if not np.isfinite(true_effects_working).all():
        raise ValidationError("Design true effects must be finite on the working scale.")
    return selection_rule_spec(
        selection_rule=selection_rule,
        alpha=alpha,
        null_working=null_working,
        se=se,
        claim_direction=claim_direction,
        threshold_working=threshold_working,
    )


def design_metrics_for_true_effects(
    true_effects_working: Sequence[float] | np.ndarray,
    *,
    null_working: float,
    se: float,
    estimate_working: float | None = None,
    alpha: float = 0.05,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    claim_direction: str = DEFAULT_CLAIM_DIRECTION,
    threshold_working: float | None = None,
    near_null_delta: float = DEFAULT_NEAR_NULL_DELTA,
) -> list[DesignMetric]:
    """Compute Type S/M design operating characteristics under a Wald model.

    These are design operating characteristics for repeated studies under an
    assumed true effect and Wald SE. They are not posterior probabilities about
    the observed estimate.
    """

    true_effects = _coerce_true_effect_array(true_effects_working)
    null_value = _coerce_finite_float(null_working, label="Design null value")
    se_value = _coerce_finite_float(se, label="Design standard error")
    estimate_value = (
        None
        if estimate_working is None
        else _coerce_finite_float(estimate_working, label="Design estimate")
    )
    alpha_value = _coerce_finite_float(alpha, label="Design alpha")
    near_null_value = _coerce_finite_float(
        near_null_delta,
        label="Design near-null delta tolerance",
    )
    threshold_value = (
        None
        if threshold_working is None
        else _coerce_finite_float(threshold_working, label="Design claim threshold")
    )
    spec = _validate_design_inputs(
        true_effects,
        null_working=null_value,
        se=se_value,
        estimate_working=estimate_value,
        alpha=alpha_value,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=threshold_value,
        near_null_delta=near_null_value,
    )

    standardized_true_effect = _finite_standardized_distance(
        true_effects,
        center=null_value,
        scale=se_value,
    )

    metrics: list[DesignMetric] = []
    for true_effect, delta in zip(true_effects, standardized_true_effect, strict=True):
        delta_float = float(delta)
        power_float = _selected_probability(spec.intervals, delta_float)
        expected_selected_abs_z: float | None
        if power_float == 0.0:
            expected_selected_abs_z = None
        else:
            selected_abs_numerator = _selected_abs_z_numerator(spec.intervals, delta_float)
            expected_selected_abs_z = max(0.0, selected_abs_numerator / power_float)

        if abs(delta_float) <= near_null_value:
            type_s = None
            type_m = None
            observed_exaggeration = None
        else:
            wrong_tail = _selected_probability(
                _wrong_sign_intervals(spec, delta_float), delta_float
            )
            type_s = None if power_float == 0.0 else _probability(wrong_tail / power_float)
            type_m = (
                None
                if expected_selected_abs_z is None
                else max(0.0, expected_selected_abs_z / abs(delta_float))
            )
            if estimate_value is None:
                observed_exaggeration = None
            else:
                estimate_distance = estimate_value - null_value
                true_effect_distance = float(true_effect) - null_value
                if np.isfinite(estimate_distance) and np.isfinite(true_effect_distance):
                    observed_exaggeration = abs(estimate_distance / true_effect_distance)
                else:
                    estimate_distance = (0.5 * estimate_value) - (0.5 * null_value)
                    true_effect_distance = (0.5 * float(true_effect)) - (0.5 * null_value)
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        observed_exaggeration = float(
                            abs(np.divide(estimate_distance, true_effect_distance))
                        )
                if not np.isfinite(observed_exaggeration):
                    raise ValidationError(
                        "Design observed exaggeration exceeds the finite floating-point range."
                    )

        metrics.append(
            DesignMetric(
                true_effect_working=float(true_effect),
                delta=delta_float,
                power=power_float,
                type_s=type_s,
                type_m=type_m,
                expected_selected_abs_z=expected_selected_abs_z,
                observed_exaggeration=observed_exaggeration,
            )
        )

    return metrics


def _metric_for_delta(delta: float, *, alpha: float) -> DesignMetric:
    return design_metrics_for_true_effects(
        [delta],
        null_working=0.0,
        se=1.0,
        alpha=alpha,
        selection_rule=DEFAULT_SELECTION_RULE,
    )[0]


def _solve_delta(
    *,
    alpha: float,
    is_satisfied: Callable[[DesignMetric], bool],
    lower: float = 0.0,
) -> float:
    alpha_value = _coerce_finite_float(alpha, label="Design alpha")
    lower_value = _coerce_finite_float(lower, label="Design solver lower bound")
    _validate_alpha(alpha_value)
    low = max(0.0, lower_value)
    high = max(1.0, low * 2.0)
    for _ in range(80):
        if is_satisfied(_metric_for_delta(high, alpha=alpha_value)):
            break
        high *= 2.0
        if high > 1e6:
            raise ValidationError("Could not bracket a finite required design delta.")
    else:
        raise ValidationError("Could not bracket a finite required design delta.")

    for _ in range(100):
        midpoint = (low + high) / 2.0
        if is_satisfied(_metric_for_delta(midpoint, alpha=alpha_value)):
            high = midpoint
        else:
            low = midpoint
    return float(high)


def solve_required_delta_for_power(alpha: float, target_power: float) -> float:
    """Required absolute true-effect delta for two-sided selected-claim power."""

    alpha_value = _coerce_finite_float(alpha, label="Design alpha")
    target = _coerce_finite_float(target_power, label="Target power")
    if not np.isfinite(target) or target <= 0 or target >= 1:
        raise ValidationError("Target power must be finite and between 0 and 1.")
    null_power = _metric_for_delta(0.0, alpha=alpha_value).power
    if target <= null_power:
        return 0.0
    return _solve_delta(alpha=alpha_value, is_satisfied=lambda metric: metric.power >= target)


def solve_required_delta_for_type_s(alpha: float, max_type_s: float) -> float:
    """Required absolute true-effect delta for two-sided selected-claim Type S risk."""

    alpha_value = _coerce_finite_float(alpha, label="Design alpha")
    target = _coerce_finite_float(max_type_s, label="Maximum Type S")
    if not np.isfinite(target) or target <= 0 or target >= 1:
        raise ValidationError("Maximum Type S must be finite and between 0 and 1.")
    return _solve_delta(
        alpha=alpha_value,
        is_satisfied=lambda metric: metric.type_s is not None and metric.type_s <= target,
        lower=DEFAULT_NEAR_NULL_DELTA,
    )


def solve_required_delta_for_type_m(alpha: float, max_type_m: float) -> float:
    """Required absolute true-effect delta for two-sided selected-claim Type M exaggeration."""

    alpha_value = _coerce_finite_float(alpha, label="Design alpha")
    target = _coerce_finite_float(max_type_m, label="Maximum Type M")
    if not np.isfinite(target) or target <= 1:
        raise ValidationError("Maximum Type M must be finite and greater than 1.")
    return _solve_delta(
        alpha=alpha_value,
        is_satisfied=lambda metric: metric.type_m is not None and metric.type_m <= target,
        lower=DEFAULT_NEAR_NULL_DELTA,
    )


def _precision_result(
    *,
    target: str,
    requested_value: float,
    required_se: float | None,
    current_se: float,
    achieved_metric: DesignMetric | None,
    z975: float,
    note: str,
) -> PrecisionTargetResult:
    if required_se is None:
        return PrecisionTargetResult(
            target=target,
            requested_value=float(requested_value),
            required_se=None,
            required_information_multiplier=None,
            approx_95_ci_width_working=None,
            achieved_power=None,
            achieved_type_s=None,
            achieved_type_m=None,
            note=note,
        )
    return PrecisionTargetResult(
        target=target,
        requested_value=float(requested_value),
        required_se=float(required_se),
        required_information_multiplier=float((current_se / required_se) ** 2),
        approx_95_ci_width_working=float(2.0 * z975 * required_se),
        achieved_power=None if achieved_metric is None else achieved_metric.power,
        achieved_type_s=None if achieved_metric is None else achieved_metric.type_s,
        achieved_type_m=None if achieved_metric is None else achieved_metric.type_m,
        note=note,
    )


def _solve_required_se_for_condition(
    *,
    true_effect_working: float,
    null_working: float,
    current_se: float,
    alpha: float,
    selection_rule: str,
    claim_direction: str,
    threshold_working: float | None,
    near_null_delta: float,
    is_satisfied: Callable[[DesignMetric], bool],
) -> tuple[float | None, DesignMetric | None, str]:
    true_distance = abs(float(true_effect_working) - float(null_working))
    if true_distance <= max(near_null_delta * current_se, 1e-300):
        return None, None, "No finite meaningful precision target is defined at or near the null."

    def metric_at(se_value: float) -> DesignMetric:
        return design_metrics_for_true_effects(
            [true_effect_working],
            null_working=null_working,
            se=se_value,
            alpha=alpha,
            selection_rule=selection_rule,
            claim_direction=claim_direction,
            threshold_working=threshold_working,
            near_null_delta=near_null_delta,
        )[0]

    current_metric = metric_at(current_se)
    if is_satisfied(current_metric):
        return current_se, current_metric, "Current CI-implied precision already meets this target."

    fail_se = current_se
    pass_se: float | None = None
    max_precision_gain = float(np.sqrt(MAX_INFORMATION_MULTIPLIER))
    min_se = current_se / max_precision_gain
    candidate_se = current_se
    for _ in range(80):
        candidate_se /= 2.0
        if candidate_se < min_se:
            candidate_se = min_se
        candidate_metric = metric_at(candidate_se)
        if is_satisfied(candidate_metric):
            pass_se = candidate_se
            break
        fail_se = candidate_se
        if candidate_se <= min_se:
            break

    if pass_se is None:
        return (
            None,
            None,
            "No finite required precision was found within the supported information range.",
        )

    for _ in range(100):
        midpoint = (pass_se + fail_se) / 2.0
        midpoint_metric = metric_at(midpoint)
        if is_satisfied(midpoint_metric):
            pass_se = midpoint
        else:
            fail_se = midpoint
        if abs(fail_se - pass_se) <= max(DEFAULT_SOLVER_TOLERANCE * pass_se, 1e-15):
            break

    achieved_metric = metric_at(pass_se)
    return pass_se, achieved_metric, "Estimated by monotonic bisection over the Wald SE."


def solve_required_precision(
    true_effect_working: float,
    *,
    null_working: float,
    current_se: float,
    alpha: float = 0.05,
    target_power: float | None = None,
    max_type_s: float | None = None,
    max_type_m: float | None = None,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    claim_direction: str = DEFAULT_CLAIM_DIRECTION,
    threshold_working: float | None = None,
    near_null_delta: float = DEFAULT_NEAR_NULL_DELTA,
    z975: float = 1.959963984540054,
) -> dict[str, float | None]:
    """Solve required precision for the strictest requested design target.

    The returned aggregate precision is the smallest required SE among requested
    targets. Per-target details are exposed through `precision_target_results`.
    """

    results = precision_target_results(
        true_effect_working,
        null_working=null_working,
        current_se=current_se,
        alpha=alpha,
        target_power=target_power,
        max_type_s=max_type_s,
        max_type_m=max_type_m,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=threshold_working,
        near_null_delta=near_null_delta,
        z975=z975,
    )
    if not results or any(result.required_se is None for result in results):
        return {
            "required_se": None,
            "required_information_multiplier": None,
            "approx_95_ci_width_working": None,
            "achieved_power": None,
            "achieved_type_s": None,
            "achieved_type_m": None,
        }
    strictest = min(results, key=lambda result: result.required_se or float("inf"))
    return {
        "required_se": strictest.required_se,
        "required_information_multiplier": strictest.required_information_multiplier,
        "approx_95_ci_width_working": strictest.approx_95_ci_width_working,
        "achieved_power": strictest.achieved_power,
        "achieved_type_s": strictest.achieved_type_s,
        "achieved_type_m": strictest.achieved_type_m,
    }


def precision_target_results(
    true_effect_working: float,
    *,
    null_working: float,
    current_se: float,
    alpha: float = 0.05,
    target_power: float | None = None,
    max_type_s: float | None = None,
    max_type_m: float | None = None,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    claim_direction: str = DEFAULT_CLAIM_DIRECTION,
    threshold_working: float | None = None,
    near_null_delta: float = DEFAULT_NEAR_NULL_DELTA,
    z975: float = 1.959963984540054,
) -> list[PrecisionTargetResult]:
    """Return per-target required precision rows for a candidate true effect."""

    true_effect_value = _coerce_finite_float(
        true_effect_working,
        label="Design precision target effect",
    )
    null_value = _coerce_finite_float(null_working, label="Design null value")
    current_se_value = _coerce_finite_float(
        current_se,
        label="Current design standard error",
    )
    alpha_value = _coerce_finite_float(alpha, label="Design alpha")
    near_null_value = _coerce_finite_float(
        near_null_delta,
        label="Design near-null delta tolerance",
    )
    threshold_value = (
        None
        if threshold_working is None
        else _coerce_finite_float(threshold_working, label="Design claim threshold")
    )
    z975_value = _coerce_finite_float(z975, label="Precision target CI multiplier")
    _validate_se(current_se_value, label="Current design standard error")
    _validate_design_inputs(
        np.asarray([true_effect_value], dtype=float),
        null_working=null_value,
        se=current_se_value,
        estimate_working=None,
        alpha=alpha_value,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=threshold_value,
        near_null_delta=near_null_value,
    )
    if z975_value <= 0:
        raise ValidationError("Precision target CI multiplier must be finite and positive.")

    target_specs: list[tuple[str, float, Callable[[DesignMetric], bool]]] = []
    if target_power is not None:
        power = _coerce_finite_float(target_power, label="Target power")
        if not np.isfinite(power) or power <= 0 or power >= 1:
            raise ValidationError("Target power must be finite and between 0 and 1.")
        target_specs.append(("Power", power, lambda metric, value=power: metric.power >= value))
    if max_type_s is not None:
        type_s = _coerce_finite_float(max_type_s, label="Maximum Type S")
        if not np.isfinite(type_s) or type_s <= 0 or type_s >= 1:
            raise ValidationError("Maximum Type S must be finite and between 0 and 1.")
        target_specs.append(
            (
                "Maximum Type S",
                type_s,
                lambda metric, value=type_s: metric.type_s is not None and metric.type_s <= value,
            )
        )
    if max_type_m is not None:
        type_m = _coerce_finite_float(max_type_m, label="Maximum Type M")
        if not np.isfinite(type_m) or type_m <= 1:
            raise ValidationError("Maximum Type M must be finite and greater than 1.")
        target_specs.append(
            (
                "Maximum Type M",
                type_m,
                lambda metric, value=type_m: metric.type_m is not None and metric.type_m <= value,
            )
        )
    if not target_specs:
        return []

    results: list[PrecisionTargetResult] = []
    for target_name, requested_value, is_satisfied in target_specs:
        required_se, achieved_metric, note = _solve_required_se_for_condition(
            true_effect_working=true_effect_value,
            null_working=null_value,
            current_se=current_se_value,
            alpha=alpha_value,
            selection_rule=selection_rule,
            claim_direction=claim_direction,
            threshold_working=threshold_value,
            near_null_delta=near_null_value,
            is_satisfied=is_satisfied,
        )
        results.append(
            _precision_result(
                target=target_name,
                requested_value=requested_value,
                required_se=required_se,
                current_se=current_se_value,
                achieved_metric=achieved_metric,
                z975=z975_value,
                note=note,
            )
        )

    return results
