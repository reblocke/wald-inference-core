from __future__ import annotations

import math

from scipy.stats import norm

from wald_inference import (
    design_metrics_for_true_effects,
    legacy_critical_effect_distance,
    relative_likelihood,
    support_interval,
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
