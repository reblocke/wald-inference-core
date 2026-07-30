"""Selected-claim rules represented on the future Wald Z scale."""

from __future__ import annotations

from typing import get_args

import numpy as np
from scipy.stats import norm

from .compatibility import standardized_distance
from .errors import ValidationError
from .types import ClaimDirection, SelectionRule, SelectionRuleSpec

DEFAULT_SELECTION_RULE: SelectionRule = "two_sided_p_lt_alpha"
DEFAULT_CLAIM_DIRECTION: ClaimDirection = "positive"

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


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or alpha <= 0 or alpha >= 1:
        raise ValidationError("Design alpha must be finite and between 0 and 1.")


def _coerce_finite_float(value: object, *, label: str) -> float:
    try:
        float_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} must be finite.") from exc
    if not np.isfinite(float_value):
        raise ValidationError(f"{label} must be finite.")
    return float_value


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


def _selected_probability(
    intervals: tuple[tuple[float, float], ...],
    delta: float,
) -> float:
    return _probability(
        sum(_interval_probability(lower, upper, delta) for lower, upper in intervals)
    )


def _selected_abs_z_numerator(
    intervals: tuple[tuple[float, float], ...],
    delta: float,
) -> float:
    return float(sum(_interval_abs_z_numerator(lower, upper, delta) for lower, upper in intervals))
