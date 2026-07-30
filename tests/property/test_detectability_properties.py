from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
