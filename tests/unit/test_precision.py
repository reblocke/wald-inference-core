from __future__ import annotations

import math
import sys
from collections.abc import Callable

import pytest

from wald_inference.errors import ValidationError
from wald_inference.precision import (
    approximate_wald_ci_width,
    information_scaled_standard_error,
    precision_target_results,
    solve_required_precision,
)


def test_information_scaling_and_ci_width_formulas_are_direct_core_apis() -> None:
    scaled_se = information_scaled_standard_error(0.5, 4.0)

    assert scaled_se == 0.25
    assert approximate_wald_ci_width(scaled_se) == pytest.approx(
        2.0 * 1.959963984540054 * scaled_se
    )
    assert approximate_wald_ci_width(scaled_se, 2.0) == 1.0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: information_scaled_standard_error(0.5, 0.0),
            "Design information multiplier must be finite and greater than 0.",
        ),
        (
            lambda: information_scaled_standard_error(0.5, math.inf),
            "Design information multiplier must be finite.",
        ),
        (
            lambda: information_scaled_standard_error(0.0, 1.0),
            "Design standard error must be finite and positive.",
        ),
        (
            lambda: approximate_wald_ci_width(0.5, 0.0),
            "Precision target CI multiplier must be finite and positive.",
        ),
        (
            lambda: approximate_wald_ci_width(sys.float_info.max),
            "Design confidence-interval width exceeds the finite floating-point range.",
        ),
    ],
)
def test_information_and_width_helpers_preserve_validation_messages(
    call: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        call()


def test_required_precision_power_target_tightens_se_when_current_probability_is_low() -> None:
    [result] = precision_target_results(
        0.5,
        null_working=0.0,
        current_se=0.5,
        target_power=0.8,
    )

    assert result.target == "Power"
    assert result.required_se is not None and result.required_se < 0.5
    assert (
        result.required_information_multiplier is not None
        and result.required_information_multiplier > 1.0
    )
    assert result.achieved_power == pytest.approx(0.8)


def test_precision_target_order_and_frozen_b06_values() -> None:
    current_se = 0.15816617164664273

    results = precision_target_results(
        0.2,
        null_working=0.0,
        current_se=current_se,
        target_power=0.8,
        max_type_s=0.01,
        max_type_m=1.25,
    )

    assert [result.target for result in results] == [
        "Power",
        "Maximum Type S",
        "Maximum Type M",
    ]
    assert results[0].required_se == pytest.approx(0.07138824202335431)
    assert results[0].required_information_multiplier == pytest.approx(4.908782966731538)
    assert results[0].achieved_power == pytest.approx(0.8000000052567556)
    assert results[1].required_se == current_se
    assert results[1].required_information_multiplier == 1.0
    assert results[1].note == "Current CI-implied precision already meets this target."
    assert results[2].required_se == pytest.approx(0.08603639264896931)
    assert results[2].required_information_multiplier == pytest.approx(3.3795806884323065)
    for result in results:
        assert result.required_se is not None
        assert result.required_information_multiplier == pytest.approx(
            (current_se / result.required_se) ** 2
        )
        assert result.approx_95_ci_width_working == pytest.approx(
            2.0 * 1.959963984540054 * result.required_se
        )


def test_stricter_type_m_precision_target_requires_more_information() -> None:
    [looser] = precision_target_results(
        0.5,
        null_working=0.0,
        current_se=0.5,
        max_type_m=1.5,
    )
    [stricter] = precision_target_results(
        0.5,
        null_working=0.0,
        current_se=0.5,
        max_type_m=1.25,
    )

    assert looser.required_information_multiplier is not None
    assert stricter.required_information_multiplier is not None
    assert stricter.required_information_multiplier > looser.required_information_multiplier


def test_near_null_precision_target_returns_no_finite_solution() -> None:
    [result] = precision_target_results(
        0.0,
        null_working=0.0,
        current_se=0.5,
        target_power=0.8,
    )

    assert result.required_se is None
    assert result.required_information_multiplier is None
    assert result.approx_95_ci_width_working is None
    assert result.achieved_power is None
    assert "near the null" in result.note


def test_solve_required_precision_returns_strictest_finite_target() -> None:
    results = precision_target_results(
        0.5,
        null_working=0.0,
        current_se=0.5,
        target_power=0.8,
        max_type_m=1.25,
    )
    aggregate = solve_required_precision(
        0.5,
        null_working=0.0,
        current_se=0.5,
        target_power=0.8,
        max_type_m=1.25,
    )
    finite_required_se = [
        result.required_se for result in results if result.required_se is not None
    ]

    assert aggregate["required_se"] == pytest.approx(min(finite_required_se))
    assert aggregate["required_information_multiplier"] == pytest.approx(
        max(
            result.required_information_multiplier
            for result in results
            if result.required_information_multiplier is not None
        )
    )


def test_aggregate_precision_is_none_when_any_target_is_infeasible() -> None:
    per_target = precision_target_results(
        0.5,
        null_working=0.0,
        current_se=1.0,
        selection_rule="ci_excludes_mcid",
        claim_direction="positive",
        threshold_working=1.0,
        target_power=0.8,
        max_type_s=0.5,
    )
    aggregate = solve_required_precision(
        0.5,
        null_working=0.0,
        current_se=1.0,
        selection_rule="ci_excludes_mcid",
        claim_direction="positive",
        threshold_working=1.0,
        target_power=0.8,
        max_type_s=0.5,
    )

    assert [result.required_se is None for result in per_target] == [True, False]
    assert "supported information range" in per_target[0].note
    assert all(value is None for value in aggregate.values())


def test_no_requested_precision_targets_returns_empty_and_none_aggregate() -> None:
    assert precision_target_results(0.5, null_working=0.0, current_se=0.5) == []
    aggregate = solve_required_precision(0.5, null_working=0.0, current_se=0.5)

    assert all(value is None for value in aggregate.values())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_power": 1.0}, "Target power"),
        ({"max_type_s": 0.0}, "Maximum Type S"),
        ({"max_type_m": 1.0}, "Maximum Type M"),
        ({"current_se": 0.0, "target_power": 0.8}, "standard error"),
        ({"z975": 0.0, "target_power": 0.8}, "CI multiplier"),
    ],
)
def test_invalid_precision_targets_raise_validation_errors(
    kwargs: dict[str, object],
    message: str,
) -> None:
    base_kwargs = {
        "true_effect_working": 0.5,
        "null_working": 0.0,
        "current_se": 0.5,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValidationError, match=message):
        precision_target_results(**base_kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: precision_target_results(
                "abc",  # type: ignore[arg-type]
                null_working=0.0,
                current_se=0.5,
                target_power=0.8,
            ),
            "precision target effect",
        ),
        (
            lambda: precision_target_results(
                0.5,
                null_working=0.0,
                current_se="abc",  # type: ignore[arg-type]
                target_power=0.8,
            ),
            "standard error",
        ),
        (
            lambda: precision_target_results(
                0.5,
                null_working=0.0,
                current_se=0.5,
                target_power="abc",  # type: ignore[arg-type]
            ),
            "Target power",
        ),
    ],
)
def test_malformed_precision_inputs_raise_validation_error(
    call: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        call()
