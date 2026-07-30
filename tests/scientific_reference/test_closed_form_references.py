from __future__ import annotations

import math
from fractions import Fraction

import pytest
from scipy.stats import norm

from wald_inference import (
    ValidationError,
    design_metrics_for_true_effects,
    legacy_critical_effect_distance,
    log_support_ratio,
    relative_likelihood,
    support_interval,
    support_interval_for_ratio,
)


def test_s_minus_2_support_matches_closed_form_normal_kernel() -> None:
    theta_hat = 0.35
    standard_error = 0.2
    interval = support_interval(theta_hat, standard_error)

    assert interval.range_working == (
        theta_hat - (2.0 * standard_error),
        theta_hat + (2.0 * standard_error),
    )
    for endpoint in interval.range_working:
        observed = float(relative_likelihood(endpoint, theta_hat, standard_error))
        assert math.isclose(observed, math.exp(-2.0), rel_tol=1e-15, abs_tol=0.0)


def test_pairwise_log_support_matches_independent_normal_kernel_difference() -> None:
    theta_hat = 0.25
    standard_error = 0.5
    candidate_a = 1.0
    candidate_b = 0.0
    expected = (
        -0.5 * ((candidate_a - theta_hat) / standard_error) ** 2
        + 0.5 * ((candidate_b - theta_hat) / standard_error) ** 2
    )

    observed = float(
        log_support_ratio(
            candidate_a,
            candidate_b,
            theta_hat=theta_hat,
            se=standard_error,
        )
    )

    assert math.isclose(observed, expected, rel_tol=1e-15, abs_tol=0.0)


def test_ratio_support_interval_matches_closed_form_normal_kernel() -> None:
    theta_hat = 0.35
    standard_error = 0.2
    ratio = 4.0
    expected_half_width = standard_error * math.sqrt(2.0 * math.log(ratio))
    interval = support_interval_for_ratio(
        theta_hat,
        standard_error,
        mle_to_bound_ratio=ratio,
    )

    assert interval.range_working == (
        theta_hat - expected_half_width,
        theta_hat + expected_half_width,
    )
    for endpoint in interval.range_working:
        observed = float(relative_likelihood(endpoint, theta_hat, standard_error))
        assert math.isclose(observed, 1.0 / ratio, rel_tol=1e-15, abs_tol=0.0)


def test_exp_2_ratio_interval_is_the_legacy_s_minus_2_interval() -> None:
    theta_hat = 0.35
    standard_error = 0.2

    assert support_interval_for_ratio(
        theta_hat,
        standard_error,
        mle_to_bound_ratio=math.exp(2.0),
    ) == support_interval(theta_hat, standard_error)


def test_extreme_binary64_neighbor_cannot_stand_in_for_a_requested_support_boundary() -> None:
    center = float.fromhex("0x1.1ccf385ebc8a0p+1023")
    standard_error = 1.0183045837972807e292
    nearest_lower = float.fromhex("0x1.1ccf385ebc89fp+1023")
    exact_delta = Fraction.from_float(center) - Fraction.from_float(nearest_lower)
    exact_se = Fraction.from_float(standard_error)
    independently_derived_log_ratio = float((exact_delta * exact_delta) / (2 * exact_se * exact_se))

    assert independently_derived_log_ratio == 1.920729410347063
    assert not math.isclose(
        independently_derived_log_ratio,
        math.log(4.0),
        rel_tol=1e-12,
        abs_tol=0.0,
    )
    with pytest.raises(ValidationError, match="cannot represent the requested"):
        support_interval_for_ratio(
            center,
            standard_error,
            mle_to_bound_ratio=4.0,
        )


def test_legacy_detectability_distance_matches_documented_z_sum() -> None:
    standard_error = 0.17
    expected = (norm.ppf(0.975) + norm.ppf(0.80)) * standard_error

    assert math.isclose(
        legacy_critical_effect_distance(standard_error),
        expected,
        rel_tol=1e-15,
        abs_tol=0.0,
    )


def test_two_sided_selected_claim_probability_and_type_s_match_tail_identities() -> None:
    delta = 1.25
    critical = float(norm.ppf(0.975))
    metric = design_metrics_for_true_effects(
        [delta],
        null_working=0.0,
        se=1.0,
        alpha=0.05,
    )[0]

    positive_tail = float(norm.sf(critical - delta))
    negative_tail = float(norm.cdf(-critical - delta))
    selected_probability = positive_tail + negative_tail

    assert math.isclose(
        metric.selected_claim_probability,
        selected_probability,
        rel_tol=1e-15,
        abs_tol=0.0,
    )
    assert metric.type_s is not None
    assert math.isclose(
        metric.type_s,
        negative_tail / selected_probability,
        rel_tol=1e-14,
        abs_tol=0.0,
    )


def test_two_sided_type_m_matches_truncated_normal_first_moment() -> None:
    delta = 1.25
    critical = float(norm.ppf(0.975))
    metric = design_metrics_for_true_effects(
        [delta],
        null_working=0.0,
        se=1.0,
        alpha=0.05,
    )[0]

    positive_probability = float(norm.sf(critical - delta))
    negative_probability = float(norm.cdf(-critical - delta))
    selected_probability = positive_probability + negative_probability
    positive_first_moment = delta * positive_probability + float(norm.pdf(critical - delta))
    negative_absolute_first_moment = -delta * negative_probability + float(
        norm.pdf(critical + delta)
    )
    expected_selected_absolute_z = (
        positive_first_moment + negative_absolute_first_moment
    ) / selected_probability

    assert metric.type_m is not None
    assert math.isclose(
        metric.type_m,
        expected_selected_absolute_z / abs(delta),
        rel_tol=1e-15,
        abs_tol=0.0,
    )
