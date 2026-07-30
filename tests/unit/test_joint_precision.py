from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from wald_inference import (
    JointPrecisionResult,
    design_metrics_for_true_effects,
    joint_precision_result,
    precision_sensitivity,
    precision_target_results,
)
from wald_inference.errors import ValidationError


def test_precision_target_status_aliases_preserve_legacy_serialized_fields() -> None:
    [result] = precision_target_results(
        0.5,
        null_working=0.0,
        current_se=0.5,
        target_power=0.8,
    )

    assert result.feasible
    assert not result.current_precision_sufficient
    assert result.achieved_selected_claim_probability == result.achieved_power
    assert tuple(asdict(result)) == (
        "target",
        "requested_value",
        "required_se",
        "required_information_multiplier",
        "approx_95_ci_width_working",
        "achieved_power",
        "achieved_type_s",
        "achieved_type_m",
        "note",
    )


def test_joint_result_is_immutable_and_uses_strictest_finite_target() -> None:
    current_se = 0.15816617164664273
    result = joint_precision_result(
        0.2,
        null_working=0.0,
        current_se=current_se,
        target_power=0.8,
        max_type_s=0.01,
        max_type_m=1.25,
    )

    assert isinstance(result, JointPrecisionResult)
    assert result.feasible
    assert not result.current_precision_sufficient
    assert result.required_se == min(
        row.required_se for row in result.target_results if row.required_se is not None
    )
    assert result.required_information_multiplier == max(
        row.required_information_multiplier
        for row in result.target_results
        if row.required_information_multiplier is not None
    )
    assert result.binding_targets == ("Power",)
    assert result.target_results[0].target == "Power"
    assert result.achieved_selected_claim_probability == result.target_results[0].achieved_power
    with pytest.raises(FrozenInstanceError):
        result.feasible = False  # type: ignore[misc]


def test_joint_achieved_values_are_forward_metrics_at_joint_precision() -> None:
    result = joint_precision_result(
        0.2,
        null_working=0.0,
        current_se=0.5,
        target_power=0.8,
        max_type_s=0.01,
        max_type_m=1.25,
    )
    assert result.required_se is not None

    [metric] = design_metrics_for_true_effects(
        [result.true_effect_working],
        null_working=0.0,
        se=result.required_se,
    )

    assert result.achieved_selected_claim_probability == metric.selected_claim_probability
    assert result.achieved_type_s == metric.type_s
    assert result.achieved_type_m == metric.type_m


def test_current_precision_sufficient_is_exactly_one_and_reports_all_ties() -> None:
    result = joint_precision_result(
        1.0,
        null_working=0.0,
        current_se=0.1,
        target_power=0.8,
        max_type_s=0.01,
        max_type_m=1.25,
    )

    assert result.feasible
    assert result.current_precision_sufficient
    assert result.required_se == 0.1
    assert result.required_information_multiplier == 1.0
    assert result.required_information_multiplier.hex() == "0x1.0000000000000p+0"
    assert result.binding_targets == ("Power", "Maximum Type S", "Maximum Type M")
    assert all(row.current_precision_sufficient for row in result.target_results)
    assert "exactly 1.0" in result.note
    assert "1e-08" in result.note


def test_binding_relative_tolerance_controls_near_ties() -> None:
    [power_row] = precision_target_results(
        0.5,
        null_working=0.0,
        current_se=0.75,
        target_power=0.8,
    )
    assert power_row.required_se is not None
    [nearby_metric] = design_metrics_for_true_effects(
        [0.5],
        null_working=0.0,
        se=power_row.required_se * (1.0 + 4e-8),
    )
    assert nearby_metric.type_m is not None
    kwargs = {
        "null_working": 0.0,
        "current_se": 0.75,
        "target_power": 0.8,
        "max_type_m": nearby_metric.type_m,
    }

    exact = joint_precision_result(0.5, **kwargs, binding_relative_tolerance=0.0)
    tolerant = joint_precision_result(0.5, **kwargs, binding_relative_tolerance=1e-6)

    assert exact.binding_targets == ("Power",)
    assert tolerant.binding_targets == ("Power", "Maximum Type M")


def test_mandatory_threshold_infeasibility_propagates_and_preserves_rows() -> None:
    result = joint_precision_result(
        0.5,
        null_working=0.0,
        current_se=1.0,
        selection_rule="ci_excludes_mcid",
        claim_direction="positive",
        threshold_working=1.0,
        target_power=0.8,
        max_type_s=0.5,
    )

    assert not result.feasible
    assert result.required_se is None
    assert result.required_information_multiplier is None
    assert result.approx_95_ci_width_working is None
    assert result.binding_targets == ()
    assert not result.current_precision_sufficient
    assert [row.feasible for row in result.target_results] == [False, True]
    assert [row.target for row in result.target_results] == ["Power", "Maximum Type S"]
    assert "Mandatory target(s) infeasible: Power" in result.note
    assert "'ci_excludes_mcid'" in result.note
    assert "not beyond the claim threshold 1.0" in result.note
    assert "1000000000000.0" in result.note
    assert "Per-target results are preserved" in result.note


def test_extreme_requirement_beyond_information_cap_is_explicitly_infeasible() -> None:
    result = joint_precision_result(
        1e-6,
        null_working=0.0,
        current_se=1.0,
        target_power=0.999999,
        near_null_delta=0.0,
    )

    assert not result.feasible
    assert not result.target_results[0].feasible
    assert "supported information range" in result.target_results[0].note
    assert "No finite bracket" in result.note
    assert "1000000000000.0" in result.note


def test_near_null_joint_result_remains_explicitly_infeasible() -> None:
    result = joint_precision_result(
        0.0,
        null_working=0.0,
        current_se=0.5,
        target_power=0.8,
        max_type_s=0.01,
        max_type_m=1.25,
    )

    assert not result.feasible
    assert [row.feasible for row in result.target_results] == [False, False, False]
    assert all(row.achieved_type_s is None for row in result.target_results)
    assert all(row.achieved_type_m is None for row in result.target_results)
    assert "Power, Maximum Type S, Maximum Type M" in result.note
    assert "near the null" in result.note


def test_sensitivity_preserves_order_duplicates_feasibility_gaps_and_is_deterministic() -> None:
    kwargs = {
        "null_working": 0.0,
        "current_se": 0.5,
        "target_power": 0.8,
        "max_type_m": 1.25,
    }
    effects = [0.2, 0.0, 0.5, 0.2]

    first = precision_sensitivity(effects, **kwargs)
    second = precision_sensitivity(np.asarray(effects), **kwargs)

    assert first == second
    assert [row.true_effect_working for row in first] == effects
    assert [row.feasible for row in first] == [True, False, True, True]
    assert first[0] == first[3]
    json.dumps([asdict(row) for row in first], allow_nan=False)


def test_joint_results_are_strict_json_for_feasible_and_infeasible_cases() -> None:
    feasible = joint_precision_result(
        0.5,
        null_working=0.0,
        current_se=0.5,
        target_power=0.8,
    )
    infeasible = joint_precision_result(
        0.0,
        null_working=0.0,
        current_se=0.5,
        max_type_m=1.25,
    )

    encoded = json.dumps(
        {"feasible": asdict(feasible), "infeasible": asdict(infeasible)},
        allow_nan=False,
        sort_keys=True,
    )

    assert '"required_information_multiplier": null' in encoded
    assert all(
        value is None or math.isfinite(value)
        for result in (feasible, infeasible)
        for value in (
            result.required_se,
            result.required_information_multiplier,
            result.approx_95_ci_width_working,
            result.achieved_selected_claim_probability,
            result.achieved_type_s,
            result.achieved_type_m,
        )
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "At least one precision guardrail"),
        ({"target_power": 0.8, "binding_relative_tolerance": -1e-8}, "Binding relative"),
        ({"target_power": 0.8, "binding_relative_tolerance": 1.0}, "Binding relative"),
        ({"target_power": 0.8, "binding_relative_tolerance": math.inf}, "Binding relative"),
    ],
)
def test_joint_input_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        joint_precision_result(
            0.5,
            null_working=0.0,
            current_se=0.5,
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("effects", "message"),
    [
        ([], "At least one design true effect"),
        ("0.5", "numeric values"),
        ([[0.5]], "numeric values"),
        ([0.5, math.nan], "finite"),
        ([0.5, math.inf], "finite"),
    ],
)
def test_sensitivity_rejects_malformed_or_nonfinite_effects(
    effects: object,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        precision_sensitivity(
            effects,  # type: ignore[arg-type]
            null_working=0.0,
            current_se=0.5,
            target_power=0.8,
        )
