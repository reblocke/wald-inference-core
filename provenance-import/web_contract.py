from __future__ import annotations

from typing import Any

import numpy as np

from .core import (
    LOG_MAX_FLOAT,
    MAX_FLOAT,
    Z975,
    ValidationError,
    asymmetry_warning,
    build_grid,
    confidence_curve,
    critical_effect_distance,
    critical_effect_markers,
    estimate_se_details,
    from_working_scale,
    log_relative_likelihood,
    max_safe_grid_span,
    relative_likelihood,
    summaries,
    to_working_scale,
    validate_inputs,
)
from .design import (
    DEFAULT_CLAIM_DIRECTION,
    DEFAULT_NEAR_NULL_DELTA,
    DEFAULT_SELECTION_RULE,
    DesignMetric,
    design_metrics_for_true_effects,
    precision_target_results,
    selection_rule_spec,
)
from .models import (
    CurveRequest,
    CurveResponse,
    DesignGridPayload,
    DesignPayload,
    DesignPrecisionTargetPayload,
    DesignScenarioPayload,
    SMinus2IntervalPayload,
    ThresholdSupportPayload,
)

S_MINUS_2_SUPPORT_CUTOFF = -2.0
S_MINUS_2_DISTANCE = 2.0


def _float_list(values: float | np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).tolist()]


def _require_strict_json_numbers(value: object, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            _require_strict_json_numbers(nested_value, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            _require_strict_json_numbers(nested_value, path=f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValidationError(
            f"Computed response value at {path} exceeds the finite floating-point range."
        )


def _exp_or_none(log_value: float | None) -> float | None:
    if log_value is None or log_value > LOG_MAX_FLOAT:
        return None
    return float(np.exp(log_value))


def _safe_display_values(
    effect_type: str,
    working_scale: str,
    working_values: float | np.ndarray,
) -> tuple[np.ndarray, bool]:
    working_array = np.asarray(working_values, dtype=float)
    if working_scale != "log":
        return np.asarray(from_working_scale(effect_type, working_array), dtype=float), False

    clipped_mask = working_array >= LOG_MAX_FLOAT
    safe_working = np.minimum(working_array, LOG_MAX_FLOAT)
    display_values = np.asarray(from_working_scale(effect_type, safe_working), dtype=float)
    if np.any(clipped_mask):
        display_values = display_values.copy()
        display_values[clipped_mask] = MAX_FLOAT
    return display_values, bool(np.any(clipped_mask))


def _display_range_exclusion_warnings(
    display_range_working: tuple[float, float] | None,
    *,
    theta_hat: float,
    lower_working: float,
    upper_working: float,
    null_working: float,
    thresholds_working: list[float],
    critical_markers_working: tuple[float, float],
) -> list[str]:
    if display_range_working is None:
        return []

    range_lower, range_upper = display_range_working

    def outside_range(value: float) -> bool:
        return value < range_lower or value > range_upper

    warnings: list[str] = []
    if outside_range(theta_hat):
        warnings.append("The chosen display range excludes the point estimate.")
    if outside_range(lower_working):
        warnings.append("The chosen display range excludes the lower 95% CI bound.")
    if outside_range(upper_working):
        warnings.append("The chosen display range excludes the upper 95% CI bound.")
    if outside_range(null_working):
        warnings.append("The chosen display range excludes the null value.")
    if any(outside_range(value) for value in thresholds_working):
        warnings.append(
            "The chosen display range excludes one or more reference thresholds / MCIDs."
        )
    if any(outside_range(value) for value in critical_markers_working):
        warnings.append("The chosen display range excludes one or more critical-effect markers.")
    return warnings


def _direction_label(value: float, reference: float, reference_name: str) -> str:
    if np.isclose(value, reference, rtol=1e-12, atol=1e-12):
        return f"at_{reference_name}"
    if value < reference:
        return f"below_{reference_name}"
    return f"above_{reference_name}"


def _threshold_support_summaries(
    *,
    thresholds_display: list[float],
    thresholds_working: list[float],
    theta_hat: float,
    null_working: float,
    se: float,
    log_null_relative_likelihood: float | None,
) -> list[ThresholdSupportPayload]:
    summaries_payload: list[ThresholdSupportPayload] = []
    if not thresholds_working:
        return summaries_payload

    log_threshold_likelihoods = _float_list(
        log_relative_likelihood(np.asarray(thresholds_working), theta_hat=theta_hat, se=se)
    )
    threshold_likelihoods = _float_list(np.exp(np.asarray(log_threshold_likelihoods)))

    for threshold_display, threshold_working, threshold_likelihood, log_threshold in zip(
        thresholds_display,
        thresholds_working,
        threshold_likelihoods,
        log_threshold_likelihoods,
        strict=True,
    ):
        log_mle_to_threshold = -log_threshold
        if log_null_relative_likelihood is None:
            log_threshold_to_null = None
        else:
            log_threshold_to_null = log_threshold - log_null_relative_likelihood

        summaries_payload.append(
            {
                "threshold_display": float(threshold_display),
                "threshold_working": float(threshold_working),
                "relative_likelihood": float(threshold_likelihood),
                "log_relative_likelihood": float(log_threshold),
                "likelihood_ratio_mle_to_threshold": _exp_or_none(log_mle_to_threshold),
                "log_likelihood_ratio_mle_to_threshold": float(log_mle_to_threshold),
                "likelihood_ratio_threshold_to_null": _exp_or_none(log_threshold_to_null),
                "log_likelihood_ratio_threshold_to_null": (
                    None if log_threshold_to_null is None else float(log_threshold_to_null)
                ),
                "direction_from_estimate": _direction_label(
                    threshold_working, theta_hat, "estimate"
                ),
                "direction_from_null": _direction_label(threshold_working, null_working, "null"),
            }
        )
    return summaries_payload


def _finite_s_minus_2_endpoint(
    theta_hat: float,
    se: float,
    direction: float,
) -> tuple[float, bool]:
    half_endpoint = (theta_hat * 0.5) + (direction * se)
    if not np.isfinite(half_endpoint) or abs(half_endpoint) > (MAX_FLOAT * 0.5):
        return (MAX_FLOAT if half_endpoint >= 0 else -MAX_FLOAT), True
    endpoint = half_endpoint * 2.0
    return float(endpoint), False


def _s_minus_2_interval(
    *,
    effect_type: str,
    working_scale: str,
    theta_hat: float,
    se: float,
    display_natural_axis: bool,
) -> tuple[SMinus2IntervalPayload, bool, bool]:
    lower_working, lower_working_clipped = _finite_s_minus_2_endpoint(theta_hat, se, -1.0)
    upper_working, upper_working_clipped = _finite_s_minus_2_endpoint(theta_hat, se, 1.0)
    working_range = np.asarray([lower_working, upper_working], dtype=float)
    working_clipped = lower_working_clipped or upper_working_clipped

    if display_natural_axis:
        display_range, display_clipped = _safe_display_values(
            effect_type, working_scale, working_range
        )
    else:
        display_range = working_range
        display_clipped = False

    return (
        {
            "support_cutoff": S_MINUS_2_SUPPORT_CUTOFF,
            "relative_likelihood_cutoff": float(np.exp(S_MINUS_2_SUPPORT_CUTOFF)),
            "likelihood_ratio_mle_to_bound": float(np.exp(-S_MINUS_2_SUPPORT_CUTOFF)),
            "range_display": _float_list(display_range),
            "range_working": _float_list(working_range),
        },
        display_clipped,
        working_clipped,
    )


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    float_value = float(value)
    return float_value if np.isfinite(float_value) else None


def _coerce_finite_float(value: Any, *, label: str) -> float:
    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be finite.") from exc
    if not np.isfinite(float_value):
        raise ValidationError(f"{label} must be finite.")
    return float_value


def _coerce_float_sequence(values: Any, *, label: str) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{label} must be supplied as numeric values, not a string.")
    try:
        return [_coerce_finite_float(value, label=label) for value in values]
    except TypeError as exc:
        raise ValidationError(f"{label} must be supplied as numeric values.") from exc


def _coerce_design_range(
    *,
    effect_type: str,
    lower: Any,
    upper: Any,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    lower_present = lower is not None
    upper_present = upper is not None
    if not lower_present and not upper_present:
        return None, None
    if lower_present != upper_present:
        raise ValidationError(
            "Design plausible true-effect range lower and upper must be supplied together."
        )

    lower_display = _coerce_finite_float(
        lower,
        label="Design plausible true-effect range lower",
    )
    upper_display = _coerce_finite_float(
        upper,
        label="Design plausible true-effect range upper",
    )
    if lower_display >= upper_display:
        raise ValidationError(
            "Design plausible true-effect range lower must be less than the upper bound."
        )

    lower_working = float(to_working_scale(effect_type, lower_display))
    upper_working = float(to_working_scale(effect_type, upper_display))
    if lower_working >= upper_working:
        raise ValidationError(
            "Design plausible true-effect range lower must be less than the upper bound "
            "on the working scale."
        )
    return (lower_display, upper_display), (lower_working, upper_working)


def _coerce_optional_finite_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _coerce_finite_float(value, label=label)


def _coerce_positive_float(value: Any, *, label: str, default: float) -> float:
    float_value = default if value is None else _coerce_finite_float(value, label=label)
    if float_value <= 0:
        raise ValidationError(f"{label} must be finite and greater than 0.")
    return float_value


def _coerce_design_claim_threshold(
    *,
    effect_type: str,
    value: Any,
) -> tuple[float | None, float | None]:
    threshold_display = _coerce_optional_finite_float(
        value,
        label="Design claim threshold",
    )
    if threshold_display is None:
        return None, None
    threshold_working = float(to_working_scale(effect_type, threshold_display))
    return threshold_display, threshold_working


def _current_display_values(
    *,
    effect_type: str,
    working_scale: str,
    display_natural_axis: bool,
    working_values: np.ndarray,
) -> np.ndarray:
    if display_natural_axis:
        display_values, _ = _safe_display_values(effect_type, working_scale, working_values)
        return display_values
    return working_values


def _deduplicate_scenarios(
    scenarios: list[tuple[str, float, str | None]],
) -> list[tuple[str, float, str | None]]:
    deduplicated: list[tuple[str, float, str | None]] = []
    seen: list[float] = []
    for source, value, note in scenarios:
        tolerance = max(1e-12, abs(value) * 1e-10)
        if any(abs(value - previous) <= max(tolerance, abs(previous) * 1e-10) for previous in seen):
            continue
        seen.append(value)
        deduplicated.append((source, value, note))
    return deduplicated


def _scenario_label(source: str, display_value: float) -> str:
    if source == "null":
        return "Null"
    if source == "ci_implied_estimate":
        return "CI-implied estimate as truth"
    if source == "threshold":
        return f"Threshold: {display_value:.6g}"
    return f"Assumed true effect: {display_value:.6g}"


def _metric_to_scenario(
    *,
    source: str,
    metric: DesignMetric,
    display_value: float,
    note: str | None,
) -> DesignScenarioPayload:
    return {
        "label": _scenario_label(source, display_value),
        "source": source,
        "true_effect_display": float(display_value),
        "true_effect_working": metric.true_effect_working,
        "delta": metric.delta,
        "power": metric.power,
        "type_s": _optional_float(metric.type_s),
        "type_m": _optional_float(metric.type_m),
        "observed_exaggeration": _optional_float(metric.observed_exaggeration),
        "note": note,
    }


def _precision_target_payloads(
    *,
    target_effect_display: float | None,
    target_effect_working: float | None,
    null_working: float,
    current_se: float,
    alpha: float,
    selection_rule: str,
    claim_direction: str,
    threshold_working: float | None,
    target_power: float | None,
    max_type_s: float | None,
    max_type_m: float | None,
) -> list[DesignPrecisionTargetPayload]:
    if target_effect_display is None or target_effect_working is None:
        return []
    requested_target_power = 0.80 if target_power is None else target_power
    results = precision_target_results(
        target_effect_working,
        null_working=null_working,
        current_se=current_se,
        alpha=alpha,
        target_power=requested_target_power,
        max_type_s=max_type_s,
        max_type_m=max_type_m,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=threshold_working,
        near_null_delta=DEFAULT_NEAR_NULL_DELTA,
        z975=Z975,
    )
    return [
        {
            "target": result.target,
            "requested_value": result.requested_value,
            "target_effect_display": float(target_effect_display),
            "target_effect_working": float(target_effect_working),
            "required_se": _optional_float(result.required_se),
            "required_information_multiplier": _optional_float(
                result.required_information_multiplier
            ),
            "approx_95_ci_width_working": _optional_float(result.approx_95_ci_width_working),
            "achieved_power": _optional_float(result.achieved_power),
            "achieved_type_s": _optional_float(result.achieved_type_s),
            "achieved_type_m": _optional_float(result.achieved_type_m),
            "note": result.note,
        }
        for result in results
    ]


def _design_grid_payload(
    *,
    metrics: list[DesignMetric],
    grid_display: np.ndarray,
) -> DesignGridPayload:
    return {
        "true_effect_display": _float_list(grid_display),
        "true_effect_working": [metric.true_effect_working for metric in metrics],
        "delta": [metric.delta for metric in metrics],
        "power": [metric.power for metric in metrics],
        "type_s": [_optional_float(metric.type_s) for metric in metrics],
        "type_m": [_optional_float(metric.type_m) for metric in metrics],
        "expected_selected_abs_z": [
            _optional_float(metric.expected_selected_abs_z) for metric in metrics
        ],
        "observed_exaggeration": [
            _optional_float(metric.observed_exaggeration) for metric in metrics
        ],
    }


def _design_payload(
    payload: CurveRequest | dict[str, Any],
    *,
    effect_type: str,
    working_scale: str,
    effect_family: str,
    display_natural_axis: bool,
    grid_working: np.ndarray,
    grid_display: np.ndarray,
    theta_hat: float,
    null_working: float,
    se: float,
    thresholds_working: list[float],
) -> DesignPayload:
    raw_alpha = payload.get("design_alpha", 0.05)
    alpha = 0.05 if raw_alpha is None else _coerce_finite_float(raw_alpha, label="Design alpha")
    requested_selection_rule = str(payload.get("design_selection_rule", DEFAULT_SELECTION_RULE))
    raw_claim_direction = str(payload.get("design_claim_direction", DEFAULT_CLAIM_DIRECTION))
    if requested_selection_rule == "one_sided_positive_p_lt_alpha":
        claim_direction = "positive"
    elif requested_selection_rule == "one_sided_negative_p_lt_alpha":
        claim_direction = "negative"
    else:
        claim_direction = raw_claim_direction
    information_multiplier = _coerce_positive_float(
        payload.get("design_information_multiplier"),
        label="Design information multiplier",
        default=1.0,
    )
    design_se = se / float(np.sqrt(information_multiplier))
    current_ci_width = 2.0 * Z975 * se
    design_ci_width = 2.0 * Z975 * design_se
    if not np.isfinite(current_ci_width) or not np.isfinite(design_ci_width):
        raise ValidationError(
            "Design confidence-interval width exceeds the finite floating-point range."
        )
    claim_threshold_display, claim_threshold_working = _coerce_design_claim_threshold(
        effect_type=effect_type,
        value=payload.get("design_claim_threshold"),
    )
    selection_spec = selection_rule_spec(
        selection_rule=requested_selection_rule,
        alpha=alpha,
        null_working=null_working,
        se=design_se,
        claim_direction=claim_direction,
        threshold_working=claim_threshold_working,
    )

    custom_true_effects_display = _coerce_float_sequence(
        payload.get("design_true_effects"), label="Design true effects"
    )
    custom_true_effects_working = _float_list(
        to_working_scale(effect_type, custom_true_effects_display)
    )
    plausible_range_display, plausible_range_working = _coerce_design_range(
        effect_type=effect_type,
        lower=payload.get("design_plausible_range_lower"),
        upper=payload.get("design_plausible_range_upper"),
    )
    target_effect_display = _coerce_optional_finite_float(
        payload.get("design_precision_target_effect"),
        label="Design precision target effect",
    )
    target_effect_working = (
        None
        if target_effect_display is None
        else float(to_working_scale(effect_type, target_effect_display))
    )
    target_power = _coerce_optional_finite_float(
        payload.get("design_target_power"),
        label="Design target power",
    )
    max_type_s = _coerce_optional_finite_float(
        payload.get("design_max_type_s"),
        label="Design maximum Type S",
    )
    max_type_m = _coerce_optional_finite_float(
        payload.get("design_max_type_m"),
        label="Design maximum Type M",
    )

    grid_metrics = design_metrics_for_true_effects(
        grid_working,
        null_working=null_working,
        se=design_se,
        estimate_working=theta_hat,
        alpha=alpha,
        selection_rule=requested_selection_rule,
        claim_direction=claim_direction,
        threshold_working=claim_threshold_working,
        near_null_delta=DEFAULT_NEAR_NULL_DELTA,
    )

    scenario_candidates = _deduplicate_scenarios(
        [
            (
                "null",
                null_working,
                "Type S/M undefined at null; selected-claim probability is rule-dependent "
                "and shown in the Power column.",
            ),
            (
                "ci_implied_estimate",
                theta_hat,
                "Optimistic/circular: uses the observed estimate as the assumed true effect.",
            ),
            *[("threshold", threshold, None) for threshold in thresholds_working],
            *[("custom_true_effect", effect, None) for effect in custom_true_effects_working],
        ]
    )
    scenario_working = np.asarray([value for _, value, _ in scenario_candidates], dtype=float)
    scenario_display = _current_display_values(
        effect_type=effect_type,
        working_scale=working_scale,
        display_natural_axis=display_natural_axis,
        working_values=scenario_working,
    )
    scenario_metrics = design_metrics_for_true_effects(
        scenario_working,
        null_working=null_working,
        se=design_se,
        estimate_working=theta_hat,
        alpha=alpha,
        selection_rule=requested_selection_rule,
        claim_direction=claim_direction,
        threshold_working=claim_threshold_working,
        near_null_delta=DEFAULT_NEAR_NULL_DELTA,
    )
    scenario_payloads = [
        _metric_to_scenario(
            source=source,
            metric=metric,
            display_value=float(display_value),
            note=note,
        )
        for (source, _, note), metric, display_value in zip(
            scenario_candidates,
            scenario_metrics,
            scenario_display,
            strict=True,
        )
    ]
    precision_target_payloads = _precision_target_payloads(
        target_effect_display=target_effect_display,
        target_effect_working=target_effect_working,
        null_working=null_working,
        current_se=se,
        alpha=alpha,
        selection_rule=requested_selection_rule,
        claim_direction=claim_direction,
        threshold_working=claim_threshold_working,
        target_power=target_power,
        max_type_s=max_type_s,
        max_type_m=max_type_m,
    )

    design_warnings = [
        "Design calibration treats x-axis values as assumed true effects; it is not a "
        "posterior probability about the observed estimate.",
        "Type S/M are undefined at or very near the null because the true-effect direction "
        "or magnitude denominator is zero.",
        f"Selected-claim rule: {selection_spec.label}.",
    ]
    if effect_family == "ratio":
        design_warnings.append(
            "For ratio measures, Type M is computed on the log working scale away from the null."
        )
    if plausible_range_display is not None:
        design_warnings.append(
            "The plausible true-effect range is shown only in the design panels and does not "
            "change the observed reconstruction."
        )
    if information_multiplier != 1.0:
        design_warnings.append(
            "The information multiplier changes only the hypothetical design SE; observed "
            "confidence and likelihood reconstruction are unchanged."
        )
    for precision_target in precision_target_payloads:
        if precision_target["required_se"] is None:
            design_warnings.append(
                f"Precision target '{precision_target['target']}' did not have a finite "
                "solution under the selected claim rule."
            )

    return {
        "config": {
            "enabled": True,
            "alpha": alpha,
            "selection_rule": selection_spec.key,
            "selection_rule_label": selection_spec.label,
            "claim_direction": selection_spec.claim_direction,
            "claim_threshold_display": claim_threshold_display,
            "claim_threshold_working": claim_threshold_working,
            "se_working": design_se,
            "current_se_working": se,
            "design_se_working": design_se,
            "information_multiplier": information_multiplier,
            "current_ci_width_working": current_ci_width,
            "approx_design_ci_width_working": design_ci_width,
            "null_working": null_working,
            "estimate_working": theta_hat,
            "near_null_delta": DEFAULT_NEAR_NULL_DELTA,
            "type_m_scale_note": (
                "Type M is a working-scale exaggeration ratio; ratio measures use the log scale."
                if effect_family == "ratio"
                else "Type M is a working-scale exaggeration ratio on the additive scale."
            ),
            "plausible_range_display": (
                None
                if plausible_range_display is None
                else [float(value) for value in plausible_range_display]
            ),
            "plausible_range_working": (
                None
                if plausible_range_working is None
                else [float(value) for value in plausible_range_working]
            ),
        },
        "grid": _design_grid_payload(metrics=grid_metrics, grid_display=grid_display),
        "scenarios": scenario_payloads,
        "precision_targets": precision_target_payloads,
        "warnings": design_warnings,
    }


def compute_curves(payload: CurveRequest | dict[str, Any]) -> CurveResponse:
    """Compute a JSON-serializable response for the browser app."""

    validated = validate_inputs(
        effect_type=str(payload.get("effect_type", "odds_ratio")),
        estimate=payload.get("estimate"),
        lower=payload.get("lower"),
        upper=payload.get("upper"),
        null_value=payload.get("null_value"),
        thresholds=payload.get("thresholds"),
        display_range_lower=payload.get("display_range_lower"),
        display_range_upper=payload.get("display_range_upper"),
        display_natural_axis=bool(payload.get("display_natural_axis", True)),
        grid_points=int(payload.get("grid_points", 801)),
        show_cutoffs=bool(payload.get("show_cutoffs", True)),
    )

    effect_type = validated.effect_spec.key
    theta_hat = float(to_working_scale(effect_type, validated.estimate))
    lower_working = float(to_working_scale(effect_type, validated.lower))
    upper_working = float(to_working_scale(effect_type, validated.upper))
    null_working = float(to_working_scale(effect_type, validated.null_value))
    thresholds_working = _float_list(to_working_scale(effect_type, validated.thresholds))

    se_info = estimate_se_details(theta_hat=theta_hat, lower=lower_working, upper=upper_working)
    critical_distance_working = critical_effect_distance(se_info.se)
    critical_markers_working = critical_effect_markers(null_working, se=se_info.se)
    safe_span = None
    if validated.display_range_working is None:
        natural_axis_upper_bound = LOG_MAX_FLOAT if validated.display_natural_axis else None
        safe_span = max_safe_grid_span(
            theta_hat=theta_hat,
            se=se_info.se,
            natural_axis_upper_bound=natural_axis_upper_bound,
        )
        grid_working = build_grid(
            theta_hat=theta_hat,
            se=se_info.se,
            n=validated.grid_points,
            include_values=(null_working, *thresholds_working, *critical_markers_working),
            max_span=safe_span,
        )
    else:
        range_lower_working, range_upper_working = validated.display_range_working
        grid_working = np.linspace(
            range_lower_working,
            range_upper_working,
            num=validated.grid_points,
            dtype=float,
        )
        if not np.isfinite(grid_working).all():
            raise ValidationError("Plausible display range must produce a finite x-grid.")
    z_values = (grid_working - theta_hat) / se_info.se
    compatibility = confidence_curve(grid_working, theta_hat=theta_hat, se=se_info.se)
    rel_likelihood = relative_likelihood(grid_working, theta_hat=theta_hat, se=se_info.se)
    log_rel_likelihood = log_relative_likelihood(grid_working, theta_hat=theta_hat, se=se_info.se)
    if validated.display_range_working is not None and (
        not np.isfinite(z_values).all() or not np.isfinite(log_rel_likelihood).all()
    ):
        raise ValidationError(
            "Plausible display range is too far from the CI-derived estimate to plot "
            "with finite floating-point precision."
        )

    display_axis_scale = "natural" if validated.display_natural_axis else "working"
    natural_axis_clipped = False
    if validated.display_natural_axis:
        grid_display, natural_axis_clipped = _safe_display_values(
            effect_type,
            validated.effect_spec.working_scale,
            grid_working,
        )
        estimate_display = validated.estimate
        ci_display = [validated.lower, validated.upper]
        null_display = validated.null_value
        critical_markers_display, critical_markers_clipped = _safe_display_values(
            effect_type,
            validated.effect_spec.working_scale,
            np.asarray(critical_markers_working, dtype=float),
        )
        natural_axis_clipped = natural_axis_clipped or critical_markers_clipped
    else:
        grid_display = grid_working
        estimate_display = theta_hat
        ci_display = [lower_working, upper_working]
        null_display = null_working
        critical_markers_display = np.asarray(critical_markers_working, dtype=float)

    warning_messages = list(validated.warnings)
    if safe_span is not None and any(
        value is not None and abs(value - theta_hat) > safe_span
        for value in (null_working, *thresholds_working, *critical_markers_working)
    ):
        warning_messages.append(
            "Grid expansion was truncated to keep the plotted payload within finite floating-point "
            "range. Very extreme null, threshold, or critical-effect values may fall "
            "outside the plotted x-axis."
        )
    if safe_span == 0.0:
        warning_messages.append(
            "The estimate sits at the finite floating-point boundary on the working scale, "
            "so the plotted x-grid collapses to the estimate."
        )
    warning_messages.extend(
        _display_range_exclusion_warnings(
            validated.display_range_working,
            theta_hat=theta_hat,
            lower_working=lower_working,
            upper_working=upper_working,
            null_working=null_working,
            thresholds_working=thresholds_working,
            critical_markers_working=critical_markers_working,
        )
    )
    if natural_axis_clipped:
        warning_messages.append(
            "Natural-axis x-values were clipped at the largest finite floating-point value. "
            "Working-scale calculations are unchanged."
        )
    observed_estimate = (
        validated.estimate if validated.provided_estimate is None else validated.provided_estimate
    )
    observed_estimate_working = float(to_working_scale(effect_type, observed_estimate))
    observed_estimate_info = estimate_se_details(
        theta_hat=observed_estimate_working,
        lower=lower_working,
        upper=upper_working,
    )
    asymmetry_message = asymmetry_warning(
        validated.effect_spec, observed_estimate_info.relative_asymmetry
    )
    if asymmetry_message is not None:
        warning_messages.append(asymmetry_message)

    null_summary = summaries(theta_hat=theta_hat, se=se_info.se, null_value=null_working)
    if null_summary["log_null_relative_likelihood"] is None:
        warning_messages.append(
            "The null value is too far from the estimate to summarize with finite floating-point "
            "precision. Null likelihood summaries are reported as overflow."
        )
    threshold_display = (
        [float(value) for value in validated.thresholds]
        if validated.display_natural_axis
        else thresholds_working
    )
    threshold_summaries = _threshold_support_summaries(
        thresholds_display=threshold_display,
        thresholds_working=thresholds_working,
        theta_hat=theta_hat,
        null_working=null_working,
        se=se_info.se,
        log_null_relative_likelihood=null_summary["log_null_relative_likelihood"],
    )
    s_minus_2_interval, s_minus_2_display_clipped, s_minus_2_working_clipped = _s_minus_2_interval(
        effect_type=effect_type,
        working_scale=validated.effect_spec.working_scale,
        theta_hat=theta_hat,
        se=se_info.se,
        display_natural_axis=validated.display_natural_axis,
    )
    if s_minus_2_working_clipped:
        warning_messages.append(
            "Working-scale S-2 interval endpoints were clipped at the largest finite "
            "floating-point value. Wald summaries are unchanged."
        )
    if s_minus_2_display_clipped:
        warning_messages.append(
            "Natural-axis S-2 interval endpoints were clipped at the largest finite "
            "floating-point value. Working-scale calculations are unchanged."
        )

    design_payload = (
        _design_payload(
            payload,
            effect_type=effect_type,
            working_scale=validated.effect_spec.working_scale,
            effect_family=validated.effect_spec.family,
            display_natural_axis=validated.display_natural_axis,
            grid_working=grid_working,
            grid_display=grid_display,
            theta_hat=theta_hat,
            null_working=null_working,
            se=se_info.se,
            thresholds_working=thresholds_working,
        )
        if bool(payload.get("design_enabled", False))
        else None
    )

    response: CurveResponse = {
        "meta": {
            "effect_spec": {
                "key": validated.effect_spec.key,
                "label": validated.effect_spec.label,
                "family": validated.effect_spec.family,
                "working_scale": validated.effect_spec.working_scale,
                "default_null": validated.effect_spec.default_null,
                "positive_only": validated.effect_spec.positive_only,
            },
            "display_axis_scale": display_axis_scale,
            "estimate_source": validated.estimate_source,
            "default_null_applied": validated.default_null_applied,
            "grid_points": len(grid_working),
            "show_cutoffs": validated.show_cutoffs,
            "se_method": se_info.method,
            "relative_asymmetry": observed_estimate_info.relative_asymmetry,
            "thresholds_display": threshold_display,
            "thresholds_working": thresholds_working,
            "display_range_active": validated.display_range_working is not None,
            "display_range_display": (
                None
                if validated.display_range_working is None
                else [float(grid_display[0]), float(grid_display[-1])]
            ),
            "display_range_working": (
                None
                if validated.display_range_working is None
                else [float(grid_working[0]), float(grid_working[-1])]
            ),
            "threshold_support_summaries": threshold_summaries,
            "s_minus_2_interval": s_minus_2_interval,
        },
        "summary": {
            "estimate_display": float(estimate_display),
            "estimate_working": theta_hat,
            "ci_display": [float(value) for value in ci_display],
            "ci_working": [lower_working, upper_working],
            "null_display": float(null_display),
            "null_working": null_working,
            "working_scale_se": se_info.se,
            "null_relative_likelihood": null_summary["null_relative_likelihood"],
            "log_null_relative_likelihood": null_summary["log_null_relative_likelihood"],
            "likelihood_ratio_mle_to_null": null_summary["likelihood_ratio_mle_to_null"],
            "log_likelihood_ratio_mle_to_null": null_summary["log_likelihood_ratio_mle_to_null"],
            "two_sided_wald_p_value": null_summary["two_sided_wald_p_value"],
            "null_z_value": null_summary["null_z_value"],
            "critical_effect_markers_display": _float_list(critical_markers_display),
            "critical_effect_markers_working": [float(value) for value in critical_markers_working],
            "critical_effect_distance_working": critical_distance_working,
        },
        "warnings": warning_messages,
        "grid": {
            "effect_display": _float_list(grid_display),
            "effect_working": _float_list(grid_working),
            "z": _float_list(z_values),
            "compatibility": _float_list(compatibility),
            "relative_likelihood": _float_list(rel_likelihood),
            "log_relative_likelihood": _float_list(log_rel_likelihood),
        },
        "design": design_payload,
    }
    _require_strict_json_numbers(response)
    return response
