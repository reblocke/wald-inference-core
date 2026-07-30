from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from wald_inference import (
    critical_effect_for_target_probability,
    power_curve,
    selected_claim_probability,
)


@settings(max_examples=80, deadline=None)
@given(
    magnitude=st.floats(
        min_value=0.0,
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
def test_two_sided_detectability_is_symmetric(
    magnitude: float,
    alpha: float,
) -> None:
    negative, positive = power_curve(
        [-magnitude, magnitude],
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
    )

    assert negative == pytest.approx(positive, rel=1e-12, abs=1e-14)


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
def test_vectorized_probability_equals_scalar_calls(effects: list[float]) -> None:
    vector = selected_claim_probability(
        effects,
        null_working=0.25,
        standard_error=0.75,
    )
    scalar = np.asarray(
        [
            selected_claim_probability(
                effect,
                null_working=0.25,
                standard_error=0.75,
            )
            for effect in effects
        ]
    )

    assert isinstance(vector, np.ndarray)
    assert vector == pytest.approx(scalar, rel=0.0, abs=0.0)
    assert np.isfinite(vector).all()
    assert ((0.0 <= vector) & (vector <= 1.0)).all()


@settings(max_examples=60, deadline=None)
@given(
    alpha=st.floats(
        min_value=1e-5,
        max_value=0.2,
        allow_nan=False,
        allow_infinity=False,
    ),
    target=st.floats(
        min_value=0.25,
        max_value=0.99,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_critical_effects_hit_targets_and_are_symmetric(
    alpha: float,
    target: float,
) -> None:
    positive = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        target_probability=target,
        claim_direction="positive",
    )
    negative = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        target_probability=target,
        claim_direction="negative",
    )

    assert positive.critical_delta == pytest.approx(
        -negative.critical_delta,
        rel=1e-12,
        abs=1e-14,
    )
    assert positive.achieved_probability >= target
    assert negative.achieved_probability >= target
    assert positive.achieved_probability == pytest.approx(target, abs=2e-15)
    assert negative.achieved_probability == pytest.approx(target, abs=2e-15)


@settings(max_examples=60, deadline=None)
@given(
    alpha=st.floats(
        min_value=1e-5,
        max_value=0.2,
        allow_nan=False,
        allow_infinity=False,
    ),
    direction=st.sampled_from(["positive", "negative"]),
)
def test_probability_is_monotonic_in_selected_one_sided_direction(
    alpha: float,
    direction: str,
) -> None:
    magnitudes = np.linspace(0.0, 8.0, 41)
    effects = magnitudes if direction == "positive" else -magnitudes
    rule = (
        "one_sided_positive_p_lt_alpha"
        if direction == "positive"
        else "one_sided_negative_p_lt_alpha"
    )

    probabilities = power_curve(
        effects,
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        selection_rule=rule,
        claim_direction=direction,
    )

    assert np.all(np.diff(probabilities) >= 0.0)


@pytest.mark.parametrize(
    ("selection_rule", "claim_direction", "sign"),
    [
        ("two_sided_p_lt_alpha", "positive", 1.0),
        ("two_sided_p_lt_alpha", "negative", -1.0),
        ("one_sided_positive_p_lt_alpha", "positive", 1.0),
        ("one_sided_negative_p_lt_alpha", "negative", -1.0),
    ],
)
def test_near_null_probability_is_monotonic_without_ulp_reversals(
    selection_rule: str,
    claim_direction: str,
    sign: float,
) -> None:
    magnitudes = np.concatenate(
        (
            np.asarray([0.0]),
            np.logspace(-16, -2, 300),
        )
    )

    probabilities = power_curve(
        sign * magnitudes,
        null_working=0.0,
        standard_error=1.0,
        alpha=0.05,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )

    assert probabilities[0] == 0.05
    assert np.all(np.diff(probabilities) >= 0.0)


@settings(max_examples=100, deadline=None)
@given(
    alpha=st.floats(
        min_value=1e-10,
        max_value=0.999,
        allow_nan=False,
        allow_infinity=False,
    ),
    target_fraction=st.floats(
        min_value=1e-3,
        max_value=0.999,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_one_sided_middle_route_is_quantile_minimal_across_alpha_range(
    alpha: float,
    target_fraction: float,
) -> None:
    target = alpha + ((1.0 - alpha) * target_fraction)
    critical_z = float(norm.isf(alpha))
    target_quantile = float(norm.isf(target))

    for selection_rule, claim_direction in [
        ("one_sided_positive_p_lt_alpha", "positive"),
        ("one_sided_negative_p_lt_alpha", "negative"),
    ]:
        result = critical_effect_for_target_probability(
            null_working=0.0,
            standard_error=1.0,
            alpha=alpha,
            target_probability=target,
            selection_rule=selection_rule,
            claim_direction=claim_direction,
        )
        magnitude = abs(result.critical_delta)
        preceding_magnitude = float(np.nextafter(magnitude, 0.0))

        assert float(critical_z - magnitude) <= target_quantile
        assert float(critical_z - preceding_magnitude) > target_quantile
        assert result.achieved_probability >= target
