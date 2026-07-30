"""Inverse precision planning under the preserved Wald design model."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from .errors import ValidationError
from .selection import (
    DEFAULT_CLAIM_DIRECTION,
    DEFAULT_SELECTION_RULE,
    _coerce_finite_float,
    _requires_threshold,
    _validate_alpha,
    _validate_se,
)
from .type_sm import (
    DEFAULT_NEAR_NULL_DELTA,
    _coerce_true_effect_array,
    _validate_design_inputs,
    design_metrics_for_true_effects,
)
from .types import DesignMetric, JointPrecisionResult, PrecisionTargetResult

DEFAULT_SOLVER_TOLERANCE = 1e-8
DEFAULT_BINDING_RELATIVE_TOLERANCE = 1e-8
MAX_INFORMATION_MULTIPLIER = 1e12
DEFAULT_95_CI_CRITICAL_VALUE = 1.959963984540054


def information_scaled_standard_error(
    standard_error: float,
    information_multiplier: float,
) -> float:
    """Return ``standard_error / sqrt(information_multiplier)``.

    The multiplier is relative Fisher information: a fourfold multiplier
    therefore halves the Wald standard error.
    """

    se_value = _coerce_finite_float(standard_error, label="Design standard error")
    multiplier_value = _coerce_finite_float(
        information_multiplier,
        label="Design information multiplier",
    )
    _validate_se(se_value)
    if multiplier_value <= 0:
        raise ValidationError("Design information multiplier must be finite and greater than 0.")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        scaled_se = float(se_value / np.sqrt(multiplier_value))
    if not np.isfinite(scaled_se) or scaled_se <= 0:
        raise ValidationError("Design standard error must be finite and positive.")
    return scaled_se


def approximate_wald_ci_width(
    standard_error: float,
    z975: float = DEFAULT_95_CI_CRITICAL_VALUE,
) -> float:
    """Return the approximate two-sided 95% Wald CI width ``2*z975*SE``."""

    se_value = _coerce_finite_float(standard_error, label="Design standard error")
    z975_value = _coerce_finite_float(z975, label="Precision target CI multiplier")
    _validate_se(se_value)
    if z975_value <= 0:
        raise ValidationError("Precision target CI multiplier must be finite and positive.")
    with np.errstate(invalid="ignore", over="ignore"):
        width = float(np.float64(2.0) * z975_value * se_value)
    if not np.isfinite(width):
        raise ValidationError(
            "Design confidence-interval width exceeds the finite floating-point range."
        )
    return width


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
    """Required absolute true-effect delta for two-sided selected-claim probability."""

    alpha_value = _coerce_finite_float(alpha, label="Design alpha")
    target = _coerce_finite_float(target_power, label="Target power")
    if not np.isfinite(target) or target <= 0 or target >= 1:
        raise ValidationError("Target power must be finite and between 0 and 1.")
    null_probability = _metric_for_delta(0.0, alpha=alpha_value).selected_claim_probability
    if target <= null_probability:
        return 0.0
    return _solve_delta(
        alpha=alpha_value,
        is_satisfied=lambda metric: metric.selected_claim_probability >= target,
    )


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
    """Required absolute true-effect delta for two-sided selected-claim Type M."""

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
        approx_95_ci_width_working=approximate_wald_ci_width(required_se, z975),
        achieved_power=(
            None if achieved_metric is None else achieved_metric.selected_claim_probability
        ),
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


def _validate_binding_relative_tolerance(binding_relative_tolerance: float) -> float:
    tolerance = _coerce_finite_float(
        binding_relative_tolerance,
        label="Binding relative tolerance",
    )
    if tolerance < 0 or tolerance >= 1:
        raise ValidationError("Binding relative tolerance must be finite and in [0, 1).")
    return tolerance


def _infeasible_joint_note(
    *,
    true_effect_working: float,
    selection_rule: str,
    claim_direction: str,
    threshold_working: float | None,
    near_null_delta: float,
    current_se: float,
    infeasible_results: tuple[PrecisionTargetResult, ...],
) -> str:
    target_names = ", ".join(result.target for result in infeasible_results)
    details = [
        (
            f"Mandatory target(s) infeasible: {target_names}. "
            f"The assumed true effect is {true_effect_working!r} on the working scale under "
            f"selection rule {selection_rule!r} and {claim_direction!r} claim direction."
        )
    ]
    if any("near the null" in result.note for result in infeasible_results):
        details.append(
            "The assumed effect is at or near the null under near-null tolerance "
            f"{near_null_delta!r} and current SE {current_se!r}."
        )
    if threshold_working is not None and _requires_threshold(selection_rule):
        beyond_threshold = (
            true_effect_working > threshold_working
            if claim_direction == "positive"
            else true_effect_working < threshold_working
        )
        threshold_relation = "is beyond" if beyond_threshold else "is not beyond"
        details.append(
            f"It {threshold_relation} the claim threshold {threshold_working!r} in the selected "
            "direction."
        )
    if any("supported information range" in result.note for result in infeasible_results):
        details.append(
            "No finite bracket was found within the supported maximum relative information "
            f"multiplier {MAX_INFORMATION_MULTIPLIER!r}."
        )
    details.append("Per-target results are preserved.")
    return "No finite joint solution under the selected assumptions. " + " ".join(details)


def _joint_precision_from_results(
    *,
    true_effect_working: float,
    current_se: float,
    selection_rule: str,
    claim_direction: str,
    threshold_working: float | None,
    near_null_delta: float,
    binding_relative_tolerance: float,
    results: list[PrecisionTargetResult],
) -> JointPrecisionResult:
    target_results = tuple(results)
    infeasible_results = tuple(result for result in target_results if not result.feasible)
    if infeasible_results:
        return JointPrecisionResult(
            true_effect_working=true_effect_working,
            feasible=False,
            required_se=None,
            required_information_multiplier=None,
            approx_95_ci_width_working=None,
            achieved_selected_claim_probability=None,
            achieved_type_s=None,
            achieved_type_m=None,
            binding_targets=(),
            current_precision_sufficient=False,
            target_results=target_results,
            note=_infeasible_joint_note(
                true_effect_working=true_effect_working,
                selection_rule=selection_rule,
                claim_direction=claim_direction,
                threshold_working=threshold_working,
                near_null_delta=near_null_delta,
                current_se=current_se,
                infeasible_results=infeasible_results,
            ),
        )

    strictest = min(
        target_results,
        key=lambda result: result.required_se if result.required_se is not None else float("inf"),
    )
    assert strictest.required_information_multiplier is not None
    joint_multiplier = strictest.required_information_multiplier
    current_precision_sufficient = all(
        result.current_precision_sufficient for result in target_results
    )
    if current_precision_sufficient:
        joint_multiplier = 1.0
    binding_targets = tuple(
        result.target
        for result in target_results
        if result.required_information_multiplier is not None
        and math.isclose(
            result.required_information_multiplier,
            joint_multiplier,
            rel_tol=binding_relative_tolerance,
            abs_tol=0.0,
        )
    )
    binding_names = ", ".join(binding_targets)
    if current_precision_sufficient:
        note = (
            "Current precision already satisfies all requested guardrails; the joint information "
            "multiplier is exactly 1.0. Binding target(s) within relative "
            f"information-multiplier tolerance {binding_relative_tolerance!r}: {binding_names}."
        )
    else:
        note = (
            "Finite joint solution uses the strictest per-target precision. Binding target(s) "
            "within relative information-multiplier tolerance "
            f"{binding_relative_tolerance!r}: {binding_names}."
        )
    return JointPrecisionResult(
        true_effect_working=true_effect_working,
        feasible=True,
        required_se=strictest.required_se,
        required_information_multiplier=joint_multiplier,
        approx_95_ci_width_working=strictest.approx_95_ci_width_working,
        achieved_selected_claim_probability=strictest.achieved_power,
        achieved_type_s=strictest.achieved_type_s,
        achieved_type_m=strictest.achieved_type_m,
        binding_targets=binding_targets,
        current_precision_sufficient=current_precision_sufficient,
        target_results=target_results,
        note=note,
    )


def joint_precision_result(
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
    z975: float = DEFAULT_95_CI_CRITICAL_VALUE,
    binding_relative_tolerance: float = DEFAULT_BINDING_RELATIVE_TOLERANCE,
) -> JointPrecisionResult:
    """Return the strictest joint result across mandatory precision guardrails.

    Every requested target is solved independently by
    :func:`precision_target_results`. If any target is infeasible, the joint
    result is infeasible while all target rows remain available for inspection.
    """

    tolerance = _validate_binding_relative_tolerance(binding_relative_tolerance)
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
    if not results:
        raise ValidationError("At least one precision guardrail is required.")
    return _joint_precision_from_results(
        true_effect_working=float(true_effect_working),
        current_se=float(current_se),
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=(None if threshold_working is None else float(threshold_working)),
        near_null_delta=float(near_null_delta),
        binding_relative_tolerance=tolerance,
        results=results,
    )


def precision_sensitivity(
    true_effects_working: Sequence[float] | np.ndarray,
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
    z975: float = DEFAULT_95_CI_CRITICAL_VALUE,
    binding_relative_tolerance: float = DEFAULT_BINDING_RELATIVE_TOLERANCE,
) -> list[JointPrecisionResult]:
    """Return deterministic joint precision results across assumed true effects."""

    true_effects = _coerce_true_effect_array(true_effects_working)
    if true_effects.size == 0:
        raise ValidationError("At least one design true effect is required for sensitivity.")
    return [
        joint_precision_result(
            float(true_effect),
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
            binding_relative_tolerance=binding_relative_tolerance,
        )
        for true_effect in true_effects
    ]


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
    z975: float = DEFAULT_95_CI_CRITICAL_VALUE,
) -> dict[str, float | None]:
    """Solve required precision for the strictest requested design target.

    The aggregate precision is the smallest required SE among requested
    targets. Per-target details are exposed through
    :func:`precision_target_results`.
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
    joint = _joint_precision_from_results(
        true_effect_working=float(true_effect_working),
        current_se=float(current_se),
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=(None if threshold_working is None else float(threshold_working)),
        near_null_delta=float(near_null_delta),
        binding_relative_tolerance=DEFAULT_BINDING_RELATIVE_TOLERANCE,
        results=results,
    )
    return {
        "required_se": joint.required_se,
        "required_information_multiplier": joint.required_information_multiplier,
        "approx_95_ci_width_working": joint.approx_95_ci_width_working,
        "achieved_power": joint.achieved_selected_claim_probability,
        "achieved_type_s": joint.achieved_type_s,
        "achieved_type_m": joint.achieved_type_m,
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
    z975: float = DEFAULT_95_CI_CRITICAL_VALUE,
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
        target_specs.append(
            (
                "Power",
                power,
                lambda metric, value=power: metric.selected_claim_probability >= value,
            )
        )
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
