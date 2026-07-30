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
    log_support_ratio,
    support_comparison,
    support_interval,
    support_interval_for_ratio,
    support_ratio,
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


def test_log_support_ratio_is_ordered_and_antisymmetric() -> None:
    a_to_b = log_support_ratio(1.0, 0.0, theta_hat=0.25, se=0.5)
    b_to_a = log_support_ratio(0.0, 1.0, theta_hat=0.25, se=0.5)

    assert float(a_to_b) == pytest.approx(-1.0)
    assert float(b_to_a) == pytest.approx(1.0)
    assert float(a_to_b) == pytest.approx(-float(b_to_a))
    assert float(log_support_ratio(1.0, 1.0, theta_hat=0.25, se=0.5)) == 0.0


def test_log_support_ratio_is_stable_for_huge_equal_or_symmetric_candidates() -> None:
    huge = 1e155

    assert float(log_support_ratio(huge, huge, theta_hat=0.0, se=1.0)) == 0.0
    assert support_ratio(huge, huge, theta_hat=0.0, se=1.0) == 1.0
    assert float(log_support_ratio(huge, -huge, theta_hat=0.0, se=1.0)) == 0.0
    assert support_ratio(huge, -huge, theta_hat=0.0, se=1.0) == 1.0


def test_log_support_ratio_avoids_cancellation_for_adjacent_large_candidates() -> None:
    candidate_a = 1e10
    candidate_b = math.nextafter(candidate_a, math.inf)
    expected = 0.5 * (candidate_b - candidate_a) * (candidate_a + candidate_b)

    observed = float(
        log_support_ratio(
            candidate_a,
            candidate_b,
            theta_hat=0.0,
            se=1.0,
        )
    )

    assert expected == 19073.486328125
    assert observed == expected


def test_log_support_ratio_preserves_a_tiny_center_between_symmetric_candidates() -> None:
    assert float(
        log_support_ratio(
            1.0,
            -1.0,
            theta_hat=1e-20,
            se=1e-10,
        )
    ) == pytest.approx(2.0)
    assert float(
        log_support_ratio(
            1e10,
            -1e10,
            theta_hat=1e-7,
            se=1.0,
        )
    ) == pytest.approx(2000.0)


def test_log_support_ratio_preserves_minimum_subnormal_inputs() -> None:
    smallest_subnormal = float.fromhex("0x0.0000000000001p-1022")

    assert (
        float(
            log_support_ratio(
                0.0,
                smallest_subnormal,
                theta_hat=0.0,
                se=smallest_subnormal,
            )
        )
        == 0.5
    )
    assert support_ratio(
        0.0,
        smallest_subnormal,
        theta_hat=0.0,
        se=smallest_subnormal,
    ) == pytest.approx(math.exp(0.5))
    assert (
        float(
            log_support_ratio(
                0.0,
                smallest_subnormal,
                theta_hat=smallest_subnormal,
                se=smallest_subnormal,
            )
        )
        == -0.5
    )


def test_log_support_ratio_broadcasts_candidate_arrays() -> None:
    candidate_a = np.asarray([-0.5, 0.25, 1.0])
    candidate_b = np.asarray([[0.0], [0.5]])
    observed = log_support_ratio(
        candidate_a,
        candidate_b,
        theta_hat=0.25,
        se=0.5,
    )
    expected = -0.5 * np.square((candidate_a - 0.25) / 0.5) + 0.5 * np.square(
        (candidate_b - 0.25) / 0.5
    )

    assert observed.shape == (2, 3)
    assert observed == pytest.approx(expected)


def test_log_support_ratio_rejects_incompatible_array_shapes() -> None:
    with pytest.raises(ValidationError, match="broadcast-compatible"):
        log_support_ratio(
            np.zeros((2, 3)),
            np.zeros((4,)),
            theta_hat=0.0,
            se=1.0,
        )


@pytest.mark.parametrize(
    ("candidate_a", "candidate_b"),
    [
        (math.nan, 0.0),
        (0.0, math.inf),
        ([0.0, math.nan], 0.0),
        ("not-numeric", 0.0),
    ],
)
def test_log_support_ratio_rejects_nonfinite_or_nonnumeric_candidates(
    candidate_a: object,
    candidate_b: object,
) -> None:
    with pytest.raises(ValidationError):
        log_support_ratio(  # type: ignore[arg-type]
            candidate_a,
            candidate_b,
            theta_hat=0.0,
            se=1.0,
        )


@pytest.mark.parametrize("candidate", [math.nan, math.inf, "not-numeric", [0.0]])
def test_scalar_support_ratio_rejects_invalid_candidates(candidate: object) -> None:
    with pytest.raises(ValidationError, match="Support comparison values must be finite"):
        support_ratio(  # type: ignore[arg-type]
            candidate,
            0.0,
            theta_hat=0.0,
            se=1.0,
        )


def test_support_ratio_retains_log_result_when_exponentiation_overflows() -> None:
    log_ratio = float(log_support_ratio(0.0, 40.0, theta_hat=0.0, se=1.0))

    assert log_ratio == pytest.approx(800.0)
    assert support_ratio(0.0, 40.0, theta_hat=0.0, se=1.0) is None
    assert support_ratio(40.0, 0.0, theta_hat=0.0, se=1.0) == 0.0
    assert support_ratio(0.0, 0.0, theta_hat=0.0, se=1.0) == 1.0


@pytest.mark.parametrize("ratio", [2.0, 4.0, 8.0, 3.5])
def test_support_interval_for_ratio_uses_requested_mle_to_bound_ratio(
    ratio: float,
) -> None:
    interval = support_interval_for_ratio(
        theta_hat=1.0,
        se=0.25,
        mle_to_bound_ratio=ratio,
    )
    expected_distance = math.sqrt(2.0 * math.log(ratio))

    assert interval.support_cutoff == pytest.approx(-math.log(ratio))
    assert interval.relative_likelihood_cutoff == pytest.approx(1.0 / ratio)
    assert interval.likelihood_ratio_mle_to_bound == pytest.approx(ratio)
    assert interval.range_working == pytest.approx(
        (
            1.0 - (0.25 * expected_distance),
            1.0 + (0.25 * expected_distance),
        )
    )


def test_support_interval_for_ratio_preserves_s_minus_2_and_width_ordering() -> None:
    s_minus_2 = support_interval(theta_hat=0.42, se=0.157)
    by_ratio = support_interval_for_ratio(
        theta_hat=0.42,
        se=0.157,
        mle_to_bound_ratio=math.exp(2.0),
    )
    narrow = support_interval_for_ratio(
        theta_hat=0.42,
        se=0.157,
        mle_to_bound_ratio=2.0,
    )
    wide = support_interval_for_ratio(
        theta_hat=0.42,
        se=0.157,
        mle_to_bound_ratio=8.0,
    )

    assert by_ratio == s_minus_2
    assert (wide.upper_working - wide.lower_working) > (narrow.upper_working - narrow.lower_working)


@pytest.mark.parametrize(
    "ratio",
    [
        1.0,
        0.0,
        -2.0,
        math.nan,
        math.inf,
        "not-a-ratio",
        pytest.param(10**10000, id="oversized-integer"),
    ],
)
def test_support_interval_for_ratio_rejects_invalid_ratio(ratio: object) -> None:
    with pytest.raises(
        ValidationError,
        match="MLE-to-bound support ratio must be finite and greater than 1",
    ):
        support_interval_for_ratio(
            theta_hat=0.0,
            se=1.0,
            mle_to_bound_ratio=ratio,  # type: ignore[arg-type]
        )


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
