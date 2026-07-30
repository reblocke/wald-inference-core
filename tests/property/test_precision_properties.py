from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wald_inference import joint_precision_result, precision_sensitivity


@settings(max_examples=40, deadline=None)
@given(
    true_effect=st.floats(
        min_value=0.02,
        max_value=4.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    current_se=st.floats(
        min_value=0.02,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    target_power=st.floats(
        min_value=0.1,
        max_value=0.95,
        allow_nan=False,
        allow_infinity=False,
    ),
    max_type_s=st.floats(
        min_value=1e-4,
        max_value=0.25,
        allow_nan=False,
        allow_infinity=False,
    ),
    max_type_m=st.floats(
        min_value=1.05,
        max_value=4.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_feasible_joint_is_exact_strictest_envelope(
    true_effect: float,
    current_se: float,
    target_power: float,
    max_type_s: float,
    max_type_m: float,
) -> None:
    result = joint_precision_result(
        true_effect,
        null_working=0.0,
        current_se=current_se,
        target_power=target_power,
        max_type_s=max_type_s,
        max_type_m=max_type_m,
    )

    assert result.feasible == all(row.feasible for row in result.target_results)
    assert result.feasible
    required_se = [row.required_se for row in result.target_results if row.required_se is not None]
    multipliers = [
        row.required_information_multiplier
        for row in result.target_results
        if row.required_information_multiplier is not None
    ]
    assert result.required_se == min(required_se)
    assert result.required_information_multiplier == max(multipliers)
    assert result.required_information_multiplier is not None
    assert result.required_information_multiplier >= 1.0
    assert result.current_precision_sufficient == (result.required_information_multiplier == 1.0)
    assert result.binding_targets


@settings(max_examples=35, deadline=None)
@given(
    true_effect=st.floats(
        min_value=0.02,
        max_value=4.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    current_se=st.floats(
        min_value=0.02,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    looser=st.floats(
        min_value=0.1,
        max_value=0.85,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_stricter_probability_guardrail_never_requires_less_information(
    true_effect: float,
    current_se: float,
    looser: float,
) -> None:
    stricter = looser + (0.99 - looser) / 2.0
    loose_result = joint_precision_result(
        true_effect,
        null_working=0.0,
        current_se=current_se,
        target_power=looser,
    )
    strict_result = joint_precision_result(
        true_effect,
        null_working=0.0,
        current_se=current_se,
        target_power=stricter,
    )

    assert loose_result.required_information_multiplier is not None
    assert strict_result.required_information_multiplier is not None
    assert (
        strict_result.required_information_multiplier
        >= loose_result.required_information_multiplier
    )


@settings(max_examples=30, deadline=None)
@given(
    effects=st.lists(
        st.floats(
            min_value=-3.0,
            max_value=3.0,
            allow_nan=False,
            allow_infinity=False,
        ).filter(lambda value: abs(value) >= 1e-5),
        min_size=1,
        max_size=7,
    )
)
def test_sensitivity_is_ordered_scalar_mapping(effects: list[float]) -> None:
    kwargs = {
        "null_working": 0.0,
        "current_se": 0.5,
        "target_power": 0.8,
        "max_type_m": 1.25,
    }
    sensitivity = precision_sensitivity(effects, **kwargs)
    scalar = [joint_precision_result(effect, **kwargs) for effect in effects]

    assert sensitivity == scalar
    assert [row.true_effect_working for row in sensitivity] == pytest.approx(effects)
