from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import wald_inference.detectability as detectability
from wald_inference import (
    CriticalEffectResult,
    ValidationError,
    critical_effect_for_target_probability,
    design_metrics_for_true_effects,
    legacy_critical_effect_distance,
    legacy_critical_effect_markers,
    power_curve,
    selected_claim_probability,
)


@pytest.mark.parametrize(
    ("selection_rule", "claim_direction", "threshold_working"),
    [
        ("two_sided_p_lt_alpha", "positive", None),
        ("one_sided_positive_p_lt_alpha", "positive", None),
        ("one_sided_negative_p_lt_alpha", "negative", None),
        ("ci_excludes_null_in_beneficial_direction", "negative", None),
        ("estimate_exceeds_mcid_and_p_lt_alpha", "positive", 0.8),
        ("ci_excludes_mcid", "negative", -0.8),
    ],
)
def test_selected_claim_probability_reuses_all_six_selection_rules(
    selection_rule: str,
    claim_direction: str,
    threshold_working: float | None,
) -> None:
    effects = np.asarray([-0.5, 0.0, 0.75])

    probabilities = selected_claim_probability(
        effects,
        null_working=0.0,
        standard_error=0.4,
        alpha=0.025,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=threshold_working,
    )
    metrics = design_metrics_for_true_effects(
        effects,
        null_working=0.0,
        se=0.4,
        alpha=0.025,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=threshold_working,
    )

    assert isinstance(probabilities, np.ndarray)
    expected = np.asarray([metric.selected_claim_probability for metric in metrics])
    if selection_rule in {
        "two_sided_p_lt_alpha",
        "one_sided_positive_p_lt_alpha",
        "one_sided_negative_p_lt_alpha",
    }:
        expected[effects == 0.0] = 0.025
    assert np.array_equal(probabilities, expected)


def test_scalar_and_array_probability_results_preserve_public_shape_contract() -> None:
    scalar = selected_claim_probability(
        0.5,
        null_working=0.0,
        standard_error=1.0,
    )
    matrix = selected_claim_probability(
        [[-0.5, 0.0], [0.5, 1.0]],
        null_working=0.0,
        standard_error=1.0,
    )

    assert isinstance(scalar, float)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(matrix[1, 0])
    assert np.isfinite(matrix).all()
    assert ((0.0 <= matrix) & (matrix <= 1.0)).all()


def test_power_curve_is_a_one_dimensional_probability_convenience() -> None:
    effects = [-1.0, 0.0, 1.0]

    curve = power_curve(
        effects,
        null_working=0.0,
        standard_error=0.5,
    )
    direct = selected_claim_probability(
        effects,
        null_working=0.0,
        standard_error=0.5,
    )

    assert isinstance(curve, np.ndarray)
    assert curve.shape == (3,)
    assert np.array_equal(curve, direct)


def test_critical_effect_result_is_typed_frozen_and_hits_target() -> None:
    result = critical_effect_for_target_probability(
        null_working=0.25,
        standard_error=0.5,
        alpha=0.05,
        target_probability=0.8,
        selection_rule="two_sided_p_lt_alpha",
        claim_direction="positive",
    )

    assert isinstance(result, CriticalEffectResult)
    assert result.selection_rule == "two_sided_p_lt_alpha"
    assert result.claim_direction == "positive"
    assert result.critical_delta > 0
    assert result.critical_effect_working == pytest.approx(
        result.null_working + result.critical_delta * result.standard_error,
        rel=0.0,
        abs=0.0,
    )
    assert result.achieved_probability == pytest.approx(0.8, abs=1e-15)
    assert selected_claim_probability(
        result.critical_effect_working,
        null_working=result.null_working,
        standard_error=result.standard_error,
        alpha=result.alpha,
        selection_rule=result.selection_rule,
        claim_direction=result.claim_direction,
    ) == pytest.approx(result.target_probability, abs=1e-14)
    with pytest.raises(FrozenInstanceError):
        result.critical_delta = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("selection_rule", "claim_direction", "expected_sign"),
    [
        ("two_sided_p_lt_alpha", "positive", 1),
        ("two_sided_p_lt_alpha", "negative", -1),
        ("one_sided_positive_p_lt_alpha", "positive", 1),
        ("one_sided_negative_p_lt_alpha", "negative", -1),
    ],
)
def test_supported_inversions_follow_the_selected_direction(
    selection_rule: str,
    claim_direction: str,
    expected_sign: int,
) -> None:
    result = critical_effect_for_target_probability(
        null_working=-0.2,
        standard_error=0.3,
        alpha=0.05,
        target_probability=0.8,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )

    assert math.copysign(1.0, result.critical_delta) == expected_sign
    assert math.copysign(1.0, result.critical_effect_working - result.null_working) == expected_sign
    assert result.achieved_probability >= result.target_probability


def test_target_at_nominal_null_probability_returns_zero_distance() -> None:
    for selection_rule, claim_direction in [
        ("two_sided_p_lt_alpha", "positive"),
        ("one_sided_positive_p_lt_alpha", "positive"),
        ("one_sided_negative_p_lt_alpha", "negative"),
    ]:
        result = critical_effect_for_target_probability(
            null_working=2.0,
            standard_error=0.5,
            alpha=0.05,
            target_probability=0.05,
            selection_rule=selection_rule,
            claim_direction=claim_direction,
        )

        assert result.critical_delta == 0.0
        assert result.critical_effect_working == 2.0
        assert result.achieved_probability == pytest.approx(0.05, abs=1e-15)


@pytest.mark.parametrize(
    "alpha",
    [
        1e-300,
        1e-20,
        1e-6,
        0.025,
        0.05,
        0.18998074903745185,
        0.24552222611130556,
        0.5,
        0.9,
    ],
)
@pytest.mark.parametrize(
    ("selection_rule", "claim_direction"),
    [
        ("two_sided_p_lt_alpha", "positive"),
        ("one_sided_positive_p_lt_alpha", "positive"),
        ("one_sided_negative_p_lt_alpha", "negative"),
    ],
)
def test_probability_at_null_is_bit_exact_for_scalar_vector_and_curve(
    alpha: float,
    selection_rule: str,
    claim_direction: str,
) -> None:
    scalar = selected_claim_probability(
        2.0,
        null_working=2.0,
        standard_error=0.5,
        alpha=alpha,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )
    vector = selected_claim_probability(
        [2.0, 2.0],
        null_working=2.0,
        standard_error=0.5,
        alpha=alpha,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )
    curve = power_curve(
        [2.0, 2.0],
        null_working=2.0,
        standard_error=0.5,
        alpha=alpha,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )

    assert scalar == alpha
    assert isinstance(vector, np.ndarray)
    assert np.array_equal(vector, np.asarray([alpha, alpha]))
    assert np.array_equal(curve, np.asarray([alpha, alpha]))


def test_target_strictly_above_nominal_alpha_requires_nonzero_distance() -> None:
    result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=0.05,
        target_probability=math.nextafter(0.05, 1.0),
    )

    assert result.critical_delta > 0.0
    assert result.achieved_probability >= result.target_probability


@pytest.mark.parametrize(
    ("selection_rule", "claim_direction"),
    [
        ("two_sided_p_lt_alpha", "positive"),
        ("one_sided_positive_p_lt_alpha", "positive"),
    ],
)
def test_near_null_routing_boundary_is_monotonic_from_both_sides(
    selection_rule: str,
    claim_direction: str,
) -> None:
    alpha = 0.05
    boundary = alpha + (detectability.STABLE_INCREMENT_MAX_RELATIVE_TARGET * alpha)
    below = math.nextafter(boundary, alpha)
    above = math.nextafter(boundary, 1.0)

    lower_result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        target_probability=below,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )
    upper_result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        target_probability=above,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )

    assert lower_result.achieved_probability >= below
    assert upper_result.achieved_probability >= above
    assert upper_result.critical_delta >= lower_result.critical_delta
    assert upper_result.critical_delta == pytest.approx(
        lower_result.critical_delta,
        rel=1e-6,
    )


@pytest.mark.parametrize(
    ("selection_rule", "claim_direction"),
    [
        ("two_sided_p_lt_alpha", "positive"),
        ("one_sided_positive_p_lt_alpha", "positive"),
    ],
)
def test_near_one_routing_boundary_is_monotonic_from_both_sides(
    selection_rule: str,
    claim_direction: str,
) -> None:
    alpha = 0.05
    boundary = 1.0 - (detectability.STABLE_COMPLEMENT_MAX_RELATIVE_TARGET * (1.0 - alpha))
    below = math.nextafter(boundary, alpha)
    above = math.nextafter(boundary, 1.0)

    lower_result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        target_probability=below,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )
    upper_result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        target_probability=above,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )

    assert lower_result.achieved_probability >= below
    assert upper_result.achieved_probability >= above
    assert abs(upper_result.critical_delta) >= abs(lower_result.critical_delta)
    assert abs(upper_result.critical_delta) == pytest.approx(
        abs(lower_result.critical_delta),
        rel=1e-7,
    )


def test_halving_standard_error_halves_critical_working_distance() -> None:
    current = critical_effect_for_target_probability(
        null_working=0.4,
        standard_error=0.2,
        target_probability=0.8,
    )
    doubled_information_scale = critical_effect_for_target_probability(
        null_working=0.4,
        standard_error=0.1,
        target_probability=0.8,
    )

    assert doubled_information_scale.critical_delta == current.critical_delta
    assert (
        doubled_information_scale.critical_effect_working - doubled_information_scale.null_working
    ) == pytest.approx(
        (current.critical_effect_working - current.null_working) / 2.0,
        rel=1e-14,
        abs=1e-15,
    )


def test_unbracketed_target_raises_explicit_no_solution_error(monkeypatch) -> None:
    monkeypatch.setattr(
        detectability,
        "_probability_at_delta",
        lambda spec, delta: 0.1,
    )

    with pytest.raises(ValidationError, match="bracket"):
        critical_effect_for_target_probability(
            null_working=0.0,
            standard_error=1.0,
            target_probability=0.8,
        )


@pytest.mark.parametrize(
    "selection_rule",
    [
        "ci_excludes_null_in_beneficial_direction",
        "estimate_exceeds_mcid_and_p_lt_alpha",
        "ci_excludes_mcid",
    ],
)
def test_inverse_rejects_selection_rules_outside_exact_v1_semantics(
    selection_rule: str,
) -> None:
    with pytest.raises(ValidationError, match="supports only"):
        critical_effect_for_target_probability(
            null_working=0.0,
            standard_error=1.0,
            selection_rule=selection_rule,
        )


@pytest.mark.parametrize(
    ("selection_rule", "claim_direction", "required_direction"),
    [
        ("one_sided_positive_p_lt_alpha", "negative", "positive"),
        ("one_sided_negative_p_lt_alpha", "positive", "negative"),
    ],
)
def test_inverse_rejects_incoherent_one_sided_direction(
    selection_rule: str,
    claim_direction: str,
    required_direction: str,
) -> None:
    with pytest.raises(ValidationError, match=required_direction):
        critical_effect_for_target_probability(
            null_working=0.0,
            standard_error=1.0,
            selection_rule=selection_rule,
            claim_direction=claim_direction,
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: selected_claim_probability(
            [],
            null_working=0.0,
            standard_error=1.0,
        ),
        lambda: selected_claim_probability(
            [math.nan],
            null_working=0.0,
            standard_error=1.0,
        ),
        lambda: selected_claim_probability(
            ["bad"],
            null_working=0.0,
            standard_error=1.0,
        ),
        lambda: power_curve(
            [[0.0, 1.0]],
            null_working=0.0,
            standard_error=1.0,
        ),
        lambda: power_curve(
            1.0,  # type: ignore[arg-type]
            null_working=0.0,
            standard_error=1.0,
        ),
        lambda: critical_effect_for_target_probability(
            null_working=0.0,
            standard_error=0.0,
        ),
        lambda: critical_effect_for_target_probability(
            null_working=0.0,
            standard_error=1.0,
            alpha=1.0,
        ),
        lambda: critical_effect_for_target_probability(
            null_working=0.0,
            standard_error=1.0,
            target_probability=0.0,
        ),
        lambda: critical_effect_for_target_probability(
            null_working=0.0,
            standard_error=1.0,
            target_probability=1.0,
        ),
    ],
)
def test_detectability_invalid_inputs_raise_validation_error(call) -> None:
    with pytest.raises(ValidationError):
        call()


def test_unrepresentable_or_overflowing_working_scale_result_fails_explicitly() -> None:
    with pytest.raises(ValidationError, match="represented accurately"):
        critical_effect_for_target_probability(
            null_working=1e16,
            standard_error=1.0,
            target_probability=0.8,
        )
    with pytest.raises(ValidationError, match="finite floating-point range"):
        critical_effect_for_target_probability(
            null_working=np.finfo(float).max,
            standard_error=np.finfo(float).max,
            target_probability=0.8,
        )


def test_legacy_closed_form_benchmark_remains_distinct_and_unchanged() -> None:
    expected_distance = 2.8015852181129683

    assert legacy_critical_effect_distance(1.0) == expected_distance
    assert legacy_critical_effect_markers(0.0, 1.0) == (
        -expected_distance,
        expected_distance,
    )
    exact = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=0.05,
        target_probability=0.8,
    )
    assert exact.critical_delta != expected_distance
    assert abs(exact.critical_delta - expected_distance) < 4e-6
