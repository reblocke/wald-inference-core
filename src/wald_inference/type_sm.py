"""Selected-claim probability and Type S/M operating characteristics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .errors import ValidationError
from .selection import (
    DEFAULT_CLAIM_DIRECTION,
    DEFAULT_SELECTION_RULE,
    _coerce_finite_float,
    _finite_standardized_distance,
    _probability,
    _selected_abs_z_numerator,
    _selected_probability,
    _wrong_sign_intervals,
    selection_rule_spec,
)
from .types import DesignMetric

DEFAULT_NEAR_NULL_DELTA = 1e-12


def _coerce_true_effect_array(values: object) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValidationError("Design true effects must be supplied as numeric values.")
    try:
        true_effects = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Design true effects must be finite.") from exc
    if true_effects.ndim != 1:
        raise ValidationError("Design true effects must be supplied as numeric values.")
    if not np.isfinite(true_effects).all():
        raise ValidationError("Design true effects must be finite.")
    return true_effects


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
):
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
    """Compute selected-claim probability and Type S/M under a Wald model.

    These are repeated-study operating characteristics under an assumed true
    effect and Wald standard error. They are not posterior probabilities about
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
        selected_claim_probability = _selected_probability(spec.intervals, delta_float)
        expected_selected_abs_z: float | None
        if selected_claim_probability == 0.0:
            expected_selected_abs_z = None
        else:
            selected_abs_numerator = _selected_abs_z_numerator(spec.intervals, delta_float)
            expected_selected_abs_z = max(
                0.0,
                selected_abs_numerator / selected_claim_probability,
            )

        if abs(delta_float) <= near_null_value:
            type_s = None
            type_m = None
            observed_exaggeration = None
        else:
            wrong_tail = _selected_probability(
                _wrong_sign_intervals(spec, delta_float), delta_float
            )
            type_s = (
                None
                if selected_claim_probability == 0.0
                else _probability(wrong_tail / selected_claim_probability)
            )
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
                selected_claim_probability=selected_claim_probability,
                type_s=type_s,
                type_m=type_m,
                expected_selected_abs_z=expected_selected_abs_z,
                observed_exaggeration=observed_exaggeration,
            )
        )

    return metrics
