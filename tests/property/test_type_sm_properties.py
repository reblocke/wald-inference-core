from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wald_inference.precision import information_scaled_standard_error
from wald_inference.type_sm import design_metrics_for_true_effects


@settings(max_examples=80, deadline=None)
@given(
    magnitude=st.floats(
        min_value=1e-6,
        max_value=8.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=1e-6,
        max_value=0.25,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_two_sided_metrics_are_symmetric_in_true_effect_direction(
    magnitude: float,
    alpha: float,
) -> None:
    negative, positive = design_metrics_for_true_effects(
        [-magnitude, magnitude],
        null_working=0.0,
        se=1.0,
        alpha=alpha,
    )

    assert negative.selected_claim_probability == pytest.approx(
        positive.selected_claim_probability,
        rel=1e-12,
        abs=1e-14,
    )
    assert negative.type_s == pytest.approx(positive.type_s, rel=1e-12, abs=1e-14)
    assert negative.type_m == pytest.approx(positive.type_m, rel=1e-12, abs=1e-14)


@settings(max_examples=80, deadline=None)
@given(
    effects=st.lists(
        st.floats(
            min_value=-8.0,
            max_value=8.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=12,
    )
)
def test_vectorized_design_metrics_equal_scalar_calls_and_stay_in_range(
    effects: list[float],
) -> None:
    vector = design_metrics_for_true_effects(
        effects,
        null_working=0.25,
        se=0.75,
        estimate_working=1.0,
    )
    scalar = [
        design_metrics_for_true_effects(
            [effect],
            null_working=0.25,
            se=0.75,
            estimate_working=1.0,
        )[0]
        for effect in effects
    ]

    assert vector == scalar
    for metric in vector:
        assert 0.0 <= metric.selected_claim_probability <= 1.0
        assert metric.power == metric.selected_claim_probability
        assert metric.type_s is None or 0.0 <= metric.type_s <= 1.0
        assert metric.type_m is None or metric.type_m >= 0.0


@settings(max_examples=80, deadline=None)
@given(
    standard_error=st.floats(
        min_value=1e-100,
        max_value=1e100,
        allow_nan=False,
        allow_infinity=False,
    ),
    multiplier=st.floats(
        min_value=1e-100,
        max_value=1e100,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_information_scaling_round_trips(
    standard_error: float,
    multiplier: float,
) -> None:
    scaled = information_scaled_standard_error(standard_error, multiplier)

    assert math.isfinite(scaled)
    assert scaled * math.sqrt(multiplier) == pytest.approx(
        standard_error,
        rel=2e-15,
        abs=0.0,
    )
