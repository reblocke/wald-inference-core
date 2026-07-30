from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from wald_inference.detectability import (
    Z80,
    Z975,
    critical_effect_distance,
    critical_effect_markers,
    legacy_critical_effect_distance,
    legacy_critical_effect_markers,
)
from wald_inference.errors import ValidationError
from wald_inference.grid import build_grid, max_safe_grid_span
from wald_inference.likelihood import (
    log_relative_likelihood,
    support_comparison,
    support_interval,
)


def test_grid_is_odd_symmetric_and_contains_the_estimate() -> None:
    grid = build_grid(theta_hat=0.42, se=0.1, n=400)

    assert len(grid) == 401
    assert grid[200] == pytest.approx(0.42)
    assert grid[0] == pytest.approx(0.42 - 0.45)
    assert grid[-1] == pytest.approx(0.42 + 0.45)


def test_grid_expands_to_include_requested_values_with_padding() -> None:
    grid = build_grid(
        theta_hat=0.0,
        se=1.0,
        span_multiplier=1.0,
        include_values=[3.0],
        n=5,
    )

    assert grid[[0, -1]].tolist() == pytest.approx([-3.25, 3.25])


def test_grid_cap_prevents_nonfinite_expansion() -> None:
    grid = build_grid(
        theta_hat=0.0,
        se=1.0,
        include_values=[sys.float_info.max],
        max_span=10.0,
        n=5,
    )

    assert np.isfinite(grid).all()
    assert grid[[0, -1]].tolist() == [-10.0, 10.0]


def test_grid_rejects_nonfinite_or_overflowing_uncapped_spans() -> None:
    with pytest.raises(ValidationError, match="Grid span"):
        build_grid(
            theta_hat=0.0,
            se=sys.float_info.max,
            span_multiplier=2.0,
        )
    with pytest.raises(ValidationError, match="Included grid values"):
        build_grid(theta_hat=0.0, se=1.0, include_values=[math.inf])


def test_max_safe_grid_span_keeps_endpoints_and_z_finite() -> None:
    theta_hat = 1e307
    se = 2.0
    span = max_safe_grid_span(theta_hat, se)
    grid = build_grid(theta_hat, se, max_span=span, n=5)

    assert math.isfinite(span)
    assert np.isfinite(grid).all()


def test_s_minus_2_interval_matches_preserved_support_definition() -> None:
    interval = support_interval(theta_hat=0.42, se=0.157)

    assert interval.support_cutoff == -2.0
    assert interval.relative_likelihood_cutoff == pytest.approx(math.exp(-2.0))
    assert interval.likelihood_ratio_mle_to_bound == pytest.approx(math.exp(2.0))
    assert interval.range_working == pytest.approx((0.42 - 2 * 0.157, 0.42 + 2 * 0.157))
    assert interval.working_clipped is False
    endpoint_log_support = log_relative_likelihood(
        np.asarray(interval.range_working),
        theta_hat=0.42,
        se=0.157,
    )
    assert endpoint_log_support.tolist() == pytest.approx([-2.0, -2.0])


def test_generic_support_interval_uses_requested_log_cutoff() -> None:
    interval = support_interval(
        theta_hat=1.0,
        se=0.25,
        log_relative_likelihood_cutoff=-0.5,
    )

    assert interval.range_working == pytest.approx((0.75, 1.25))
    assert interval.relative_likelihood_cutoff == pytest.approx(math.exp(-0.5))


def test_s_minus_2_interval_clips_unrepresentable_endpoint() -> None:
    interval = support_interval(
        theta_hat=1.3949999999999999e308,
        se=2.015343154852383e307,
    )

    assert interval.lower_working == pytest.approx(9.919313690295233e307)
    assert interval.upper_working == sys.float_info.max
    assert interval.lower_clipped is False
    assert interval.upper_clipped is True


def test_support_comparison_matches_log_domain_algebra() -> None:
    comparison = support_comparison(
        candidate_working=1.0,
        reference_working=0.0,
        theta_hat=0.25,
        se=0.5,
    )
    expected_candidate_log = -0.5 * ((1.0 - 0.25) / 0.5) ** 2
    expected_reference_log = -0.5 * ((0.0 - 0.25) / 0.5) ** 2

    assert comparison.log_relative_likelihood == pytest.approx(expected_candidate_log)
    assert comparison.relative_likelihood == pytest.approx(math.exp(expected_candidate_log))
    assert comparison.log_likelihood_ratio_mle_to_candidate == pytest.approx(
        -expected_candidate_log
    )
    assert comparison.log_likelihood_ratio_candidate_to_reference == pytest.approx(
        expected_candidate_log - expected_reference_log
    )


def test_support_comparison_returns_none_when_display_ratio_overflows() -> None:
    comparison = support_comparison(
        candidate_working=0.0,
        reference_working=40.0,
        theta_hat=0.0,
        se=1.0,
    )

    assert comparison.log_likelihood_ratio_candidate_to_reference == pytest.approx(800.0)
    assert comparison.likelihood_ratio_candidate_to_reference is None


@pytest.mark.parametrize("cutoff", [0.1, math.nan, math.inf])
def test_support_interval_rejects_invalid_cutoff(cutoff: float) -> None:
    with pytest.raises(ValidationError):
        support_interval(
            theta_hat=0.0,
            se=1.0,
            log_relative_likelihood_cutoff=cutoff,
        )


def test_legacy_detectability_benchmark_preserves_z_sum_formula_and_aliases() -> None:
    se = 0.2
    expected_distance = (Z975 + Z80) * se

    assert legacy_critical_effect_distance(se) == pytest.approx(expected_distance)
    assert critical_effect_distance(se) == pytest.approx(expected_distance)
    assert legacy_critical_effect_markers(0.0, se) == pytest.approx(
        (-expected_distance, expected_distance)
    )
    assert critical_effect_markers(0.0, se) == pytest.approx(
        (-expected_distance, expected_distance)
    )


def test_legacy_detectability_rejects_nonfinite_results() -> None:
    with pytest.raises(ValidationError):
        legacy_critical_effect_distance(math.inf)
    with pytest.raises(ValidationError, match="exceed"):
        legacy_critical_effect_markers(sys.float_info.max, 1e307)
