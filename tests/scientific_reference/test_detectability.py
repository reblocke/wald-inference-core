from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from wald_inference import (
    critical_effect_for_target_probability,
    from_working_scale,
    legacy_critical_effect_distance,
    selected_claim_probability,
)


def test_two_sided_probability_matches_direct_normal_tail_evaluation() -> None:
    alpha = 0.05
    delta = 1.75
    critical_z = norm.isf(alpha / 2.0)
    direct_probability = norm.cdf(-critical_z - delta) + norm.sf(critical_z - delta)

    probability = selected_claim_probability(
        delta,
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        selection_rule="two_sided_p_lt_alpha",
    )

    assert probability == pytest.approx(direct_probability, rel=1e-14, abs=1e-15)


@pytest.mark.parametrize(
    ("selection_rule", "claim_direction", "expected_sign"),
    [
        ("one_sided_positive_p_lt_alpha", "positive", 1.0),
        ("one_sided_negative_p_lt_alpha", "negative", -1.0),
    ],
)
def test_one_sided_inverse_matches_analytic_normal_quantiles(
    selection_rule: str,
    claim_direction: str,
    expected_sign: float,
) -> None:
    alpha = 0.05
    target = 0.80
    expected_magnitude = norm.isf(alpha) + norm.ppf(target)

    result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        target_probability=target,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )

    assert result.critical_delta == pytest.approx(
        expected_sign * expected_magnitude,
        rel=1e-14,
        abs=1e-14,
    )
    assert result.achieved_probability == pytest.approx(target, abs=1e-15)


@pytest.mark.parametrize(
    ("alpha", "target", "expected_delta"),
    [
        (
            0.8465995795106757,
            0.8581265056668502,
            0.049982116413653978839137027258425929,
        ),
        (
            0.075359,
            0.076959,
            0.011171989432112436861317476052792152,
        ),
        (
            0.9,
            0.95,
            0.36330206140687169078886731502209453,
        ),
        (
            0.99,
            0.999,
            0.76388443212697251012101309567182476,
        ),
        (
            1e-12,
            1e-10,
            0.6731429228970757335137790556967245,
        ),
        (
            0.1,
            0.9991,
            4.402940714904462159111723469740108,
        ),
    ],
)
def test_one_sided_inverse_matches_mpmath_high_precision_quantile_differences(
    alpha: float,
    target: float,
    expected_delta: float,
) -> None:
    result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        target_probability=target,
        selection_rule="one_sided_positive_p_lt_alpha",
        claim_direction="positive",
    )

    assert result.critical_delta >= expected_delta
    assert result.critical_delta - expected_delta <= 6e-15
    assert result.achieved_probability >= target
    preceding_delta = float(np.nextafter(result.critical_delta, 0.0))
    preceding_probability = selected_claim_probability(
        preceding_delta,
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        selection_rule="one_sided_positive_p_lt_alpha",
        claim_direction="positive",
    )
    assert preceding_probability < target


@pytest.mark.parametrize(
    ("target", "expected_delta"),
    [
        (0.50, 1.959852920520229),
        (0.80, 2.8015817870136996),
        (0.90, 3.24151498680644),
    ],
)
def test_two_sided_exact_inverse_matches_independent_reference_values(
    target: float,
    expected_delta: float,
) -> None:
    result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=0.05,
        target_probability=target,
    )
    critical_z = norm.isf(0.05 / 2.0)
    direct_probability = norm.cdf(-critical_z - result.critical_delta) + norm.sf(
        critical_z - result.critical_delta
    )

    assert result.critical_delta == pytest.approx(expected_delta, rel=0.0, abs=2e-13)
    assert direct_probability >= target
    assert direct_probability - target <= 8e-15
    preceding_delta = float(np.nextafter(result.critical_delta, 0.0))
    preceding_probability = selected_claim_probability(
        preceding_delta,
        null_working=0.0,
        standard_error=1.0,
        alpha=0.05,
    )
    assert preceding_probability < target


def test_decreasing_one_sided_probability_is_below_high_precision_reference() -> None:
    probability = selected_claim_probability(
        -0.125,
        null_working=0.0,
        standard_error=1.0,
        alpha=0.5,
        selection_rule="one_sided_positive_p_lt_alpha",
        claim_direction="positive",
    )
    high_precision_reference = 0.45026177516988710702069

    assert probability <= high_precision_reference
    assert high_precision_reference - probability <= 1e-14


@pytest.mark.parametrize(
    ("alpha", "selection_rule", "claim_direction", "expected_delta"),
    [
        (
            0.05,
            "two_sided_p_lt_alpha",
            "positive",
            7.783001915816272e-9,
        ),
        (
            0.18998074903745185,
            "two_sided_p_lt_alpha",
            "positive",
            1.1193945803542321e-8,
        ),
        (
            0.18998074903745185,
            "two_sided_p_lt_alpha",
            "negative",
            -1.1193945803542321e-8,
        ),
        (
            0.18998074903745185,
            "one_sided_positive_p_lt_alpha",
            "positive",
            1.0228760600069692e-16,
        ),
        (
            0.18998074903745185,
            "one_sided_negative_p_lt_alpha",
            "negative",
            -1.0228760600069692e-16,
        ),
        (
            0.24552222611130556,
            "two_sided_p_lt_alpha",
            "positive",
            1.0843558048345682e-8,
        ),
        (
            0.24552222611130556,
            "one_sided_positive_p_lt_alpha",
            "positive",
            8.81900106257256e-17,
        ),
    ],
)
def test_next_float_above_alpha_matches_high_precision_reference_root(
    alpha: float,
    selection_rule: str,
    claim_direction: str,
    expected_delta: float,
) -> None:
    target = float(np.nextafter(alpha, 1.0))

    result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=alpha,
        target_probability=target,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )

    assert result.critical_delta == pytest.approx(
        expected_delta,
        rel=3e-9,
        abs=5e-32,
    )
    assert result.achieved_probability >= target


@pytest.mark.parametrize(
    ("selection_rule", "claim_direction", "expected_delta"),
    [
        (
            "two_sided_p_lt_alpha",
            "positive",
            10.169500136141441,
        ),
        (
            "two_sided_p_lt_alpha",
            "negative",
            -10.169500136141441,
        ),
        (
            "one_sided_positive_p_lt_alpha",
            "positive",
            9.85438977855286,
        ),
        (
            "one_sided_negative_p_lt_alpha",
            "negative",
            -9.85438977855286,
        ),
    ],
)
def test_next_float_below_one_matches_high_precision_reference_root(
    selection_rule: str,
    claim_direction: str,
    expected_delta: float,
) -> None:
    target = float(np.nextafter(1.0, 0.0))

    result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=0.05,
        target_probability=target,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
    )

    assert result.critical_delta == pytest.approx(
        expected_delta,
        rel=2e-15,
        abs=2e-14,
    )
    assert result.achieved_probability == target


def test_exact_two_sided_solution_is_not_the_legacy_closed_form_benchmark() -> None:
    exact = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=1.0,
        alpha=0.05,
        target_probability=0.80,
    )
    legacy = legacy_critical_effect_distance(1.0)

    assert exact.critical_delta == pytest.approx(2.8015817870136996, abs=2e-13)
    assert legacy == 2.8015852181129683
    assert legacy - exact.critical_delta == pytest.approx(3.431099269e-6, rel=2e-10)


@pytest.mark.parametrize(
    ("standard_error", "exact_distance", "legacy_distance"),
    [
        (0.15816617164664273, 0.4431154658069169, 0.4431160084907528),
        (0.20687375447019513, 0.5795737427348426, 0.5795744525392302),
    ],
)
def test_frozen_additive_and_log_scale_precision_cases_match_reference_distances(
    standard_error: float,
    exact_distance: float,
    legacy_distance: float,
) -> None:
    result = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=standard_error,
        alpha=0.05,
        target_probability=0.80,
    )

    assert result.critical_effect_working == pytest.approx(exact_distance, abs=5e-14)
    assert legacy_critical_effect_distance(standard_error) == legacy_distance
    assert result.critical_effect_working < legacy_distance


def test_identity_and_log_effects_compose_through_the_effect_registry() -> None:
    standard_error = 0.2
    positive = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=standard_error,
        target_probability=0.80,
        claim_direction="positive",
    )
    negative = critical_effect_for_target_probability(
        null_working=0.0,
        standard_error=standard_error,
        target_probability=0.80,
        claim_direction="negative",
    )

    additive_upper = from_working_scale(
        "mean_difference",
        positive.critical_effect_working,
    )
    ratio_upper = from_working_scale(
        "odds_ratio",
        positive.critical_effect_working,
    )
    ratio_lower = from_working_scale(
        "odds_ratio",
        negative.critical_effect_working,
    )

    assert additive_upper == positive.critical_effect_working
    assert ratio_lower * ratio_upper == pytest.approx(1.0, rel=1e-14)
    assert math.log(ratio_upper) == pytest.approx(positive.critical_effect_working)
