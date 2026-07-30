from __future__ import annotations

import math

import pytest

from wald_inference import (
    approximate_wald_ci_width,
    design_metrics_for_true_effects,
    from_working_scale,
    joint_precision_result,
    precision_sensitivity,
    reconstruct_wald_from_95_ci,
    to_working_scale,
)


def test_joint_information_and_width_identities_hold_at_strictest_solution() -> None:
    current_se = 0.5
    result = joint_precision_result(
        0.5,
        null_working=0.0,
        current_se=current_se,
        target_power=0.8,
        max_type_s=0.01,
        max_type_m=1.25,
    )
    assert result.required_se is not None
    assert result.required_information_multiplier is not None

    assert result.required_information_multiplier == pytest.approx(
        (current_se / result.required_se) ** 2,
        rel=1e-15,
    )
    assert result.approx_95_ci_width_working == approximate_wald_ci_width(result.required_se)


def test_each_solved_guardrail_hits_its_forward_metric_on_the_feasible_side() -> None:
    result = joint_precision_result(
        0.5,
        null_working=0.0,
        current_se=0.75,
        target_power=0.8,
        max_type_s=0.01,
        max_type_m=1.25,
    )

    for row in result.target_results:
        assert row.required_se is not None
        [metric] = design_metrics_for_true_effects(
            [0.5],
            null_working=0.0,
            se=row.required_se,
        )
        if row.target == "Power":
            assert metric.selected_claim_probability >= row.requested_value
        elif row.target == "Maximum Type S":
            assert metric.type_s is not None and metric.type_s <= row.requested_value
        else:
            assert row.target == "Maximum Type M"
            assert metric.type_m is not None and metric.type_m <= row.requested_value


def test_monotonic_bracket_places_a_larger_se_on_the_failing_side() -> None:
    result = joint_precision_result(
        0.5,
        null_working=0.0,
        current_se=0.75,
        target_power=0.8,
        max_type_s=0.01,
        max_type_m=1.25,
    )

    for row in result.target_results:
        assert row.required_se is not None
        [less_precise] = design_metrics_for_true_effects(
            [0.5],
            null_working=0.0,
            se=row.required_se * (1.0 + 1e-8),
        )
        if row.target == "Power":
            assert less_precise.selected_claim_probability < row.requested_value
        elif row.target == "Maximum Type S":
            assert less_precise.type_s is not None
            assert less_precise.type_s > row.requested_value
        else:
            assert row.target == "Maximum Type M"
            assert less_precise.type_m is not None
            assert less_precise.type_m > row.requested_value


def test_two_sided_sensitivity_envelope_decreases_with_effect_magnitude() -> None:
    results = precision_sensitivity(
        [0.1, 0.2, 0.4, 0.8],
        null_working=0.0,
        current_se=0.5,
        target_power=0.8,
        max_type_m=1.25,
    )
    multipliers = [result.required_information_multiplier for result in results]

    assert all(multiplier is not None for multiplier in multipliers)
    assert multipliers == sorted(multipliers, reverse=True)


def test_log_ratio_conversion_and_ci_reconstruction_feed_the_same_working_scale_solver() -> None:
    odds_ratio = 1.5
    true_effect_working = to_working_scale("odds_ratio", odds_ratio)
    current_se = 0.2
    z975 = 1.959963984540054
    estimate_working = math.log(1.25)
    lower = math.exp(estimate_working - z975 * current_se)
    upper = math.exp(estimate_working + z975 * current_se)
    reconstruction = reconstruct_wald_from_95_ci(
        "odds_ratio",
        lower=lower,
        upper=upper,
    )

    direct = joint_precision_result(
        true_effect_working,
        null_working=0.0,
        current_se=current_se,
        target_power=0.8,
        max_type_m=1.25,
    )
    reconstructed = joint_precision_result(
        true_effect_working,
        null_working=0.0,
        current_se=reconstruction.standard_error,
        target_power=0.8,
        max_type_m=1.25,
    )

    assert reconstruction.standard_error == pytest.approx(current_se, rel=1e-15)
    assert reconstructed.required_se == pytest.approx(direct.required_se, rel=1e-15)
    assert reconstructed.required_information_multiplier == pytest.approx(
        direct.required_information_multiplier,
        rel=2e-15,
    )
    assert from_working_scale("odds_ratio", true_effect_working) == pytest.approx(odds_ratio)
