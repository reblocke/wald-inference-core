from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from wald_inference.compatibility import (
    MAX_FINITE_ABS_Z,
    compatibility_curve,
    confidence_curve,
    standardized_distance,
    summaries,
    wald_point_summary,
)
from wald_inference.errors import ValidationError
from wald_inference.likelihood import log_relative_likelihood, relative_likelihood
from wald_inference.reconstruction import Z975, estimate_se


def test_observed_curves_peak_at_the_estimate() -> None:
    theta_hat = 0.42
    se = 0.157

    assert compatibility_curve(theta_hat, theta_hat, se).item() == pytest.approx(1.0)
    assert confidence_curve(theta_hat, theta_hat, se).item() == pytest.approx(1.0)
    assert relative_likelihood(theta_hat, theta_hat, se).item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "function",
    [
        standardized_distance,
        compatibility_curve,
        relative_likelihood,
        log_relative_likelihood,
    ],
)
def test_observed_functions_preserve_scalar_and_array_result_types(function) -> None:
    scalar = function(1.0, 0.0, 1.0)
    zero_dimensional = function(np.asarray(1.0), 0.0, 1.0)
    sequence = function([1.0], 0.0, 1.0)

    assert type(scalar) is np.float64
    assert type(zero_dimensional) is np.float64
    assert isinstance(sequence, np.ndarray)
    assert sequence.shape == (1,)


def test_ci_bounds_have_point_oh_five_compatibility_and_wald_likelihood() -> None:
    theta_hat = 0.42
    lower = 0.11
    upper = 0.73
    se = estimate_se(theta_hat, lower, upper)

    compatibility = compatibility_curve([lower, upper], theta_hat, se)
    likelihood = relative_likelihood([lower, upper], theta_hat, se)

    assert compatibility.tolist() == pytest.approx([0.05, 0.05], rel=1e-3, abs=1e-5)
    assert likelihood.tolist() == pytest.approx(
        [math.exp(-(Z975**2) / 2.0)] * 2,
        rel=1e-4,
    )


def test_standardized_distance_preserves_direct_arithmetic_for_ordinary_values() -> None:
    values = np.asarray([-3.5, 0.0, 2.25])
    theta_hat = 0.25
    se = 0.5

    result = standardized_distance(values, theta_hat, se)

    np.testing.assert_array_equal(result, (values - theta_hat) / se)


def test_standardized_distance_recovers_representable_subtraction_overflow() -> None:
    maximum = sys.float_info.max

    max_distance = standardized_distance(maximum, -maximum, 2.0).item()
    distance_of_two = standardized_distance(maximum, -maximum, maximum).item()

    assert max_distance == maximum
    assert distance_of_two == 2.0


@pytest.mark.parametrize(
    ("theta_hat", "se"),
    [
        (-sys.float_info.max, math.nextafter(2.0, 0.0)),
        (0.0, 0.5),
    ],
)
def test_unrepresentable_standardized_distance_raises(
    theta_hat: float,
    se: float,
) -> None:
    with pytest.raises(ValidationError, match="Standardized distance.*finite"):
        standardized_distance(sys.float_info.max, theta_hat, se)


def test_standardized_distance_preserves_smallest_subnormal_result() -> None:
    smallest_subnormal = math.ulp(0.0)

    result = standardized_distance(
        smallest_subnormal,
        theta_hat=-smallest_subnormal,
        se=2.0,
    )

    assert result.item() == smallest_subnormal


@pytest.mark.parametrize(
    ("theta", "theta_hat", "se"),
    [
        (math.nan, 0.0, 1.0),
        (0.0, math.inf, 1.0),
        (0.0, 0.0, math.nan),
        (0.0, 0.0, 0.0),
    ],
)
def test_standardized_distance_strictly_rejects_invalid_inputs(
    theta: float,
    theta_hat: float,
    se: float,
) -> None:
    with pytest.raises(ValidationError):
        standardized_distance(theta, theta_hat, se)


def test_point_summary_matches_preserved_null_summary_mapping() -> None:
    point = wald_point_summary(theta_hat=0.42, se=0.157, candidate_working=0.0)
    legacy = summaries(theta_hat=0.42, se=0.157, null_value=0.0)

    assert legacy == {
        "null_relative_likelihood": point.relative_likelihood,
        "log_null_relative_likelihood": point.log_relative_likelihood,
        "likelihood_ratio_mle_to_null": point.likelihood_ratio_mle_to_candidate,
        "log_likelihood_ratio_mle_to_null": (point.log_likelihood_ratio_mle_to_candidate),
        "two_sided_wald_p_value": point.two_sided_wald_p_value,
        "null_z_value": point.z_value,
    }
    assert point.null_relative_likelihood == point.relative_likelihood
    assert point.null_z_value == point.z_value


def test_summary_uses_log_domain_when_ratio_overflows() -> None:
    point = wald_point_summary(theta_hat=0.0, se=1.0, candidate_working=40.0)

    assert point.relative_likelihood == 0.0
    assert point.log_relative_likelihood == pytest.approx(-800.0)
    assert point.likelihood_ratio_mle_to_candidate is None
    assert point.log_likelihood_ratio_mle_to_candidate == pytest.approx(800.0)
    assert point.two_sided_wald_p_value == 0.0


def test_information_free_summary_uses_none_for_unrepresentable_log_values() -> None:
    point = wald_point_summary(
        theta_hat=0.0,
        se=1e-320,
        candidate_working=1e308,
    )

    assert point.relative_likelihood == 0.0
    assert point.log_relative_likelihood is None
    assert point.likelihood_ratio_mle_to_candidate is None
    assert point.log_likelihood_ratio_mle_to_candidate is None
    assert point.two_sided_wald_p_value == 0.0
    assert point.z_value is None


def test_summary_guard_matches_maximum_finite_log_boundary() -> None:
    point = wald_point_summary(
        theta_hat=0.0,
        se=1.0,
        candidate_working=MAX_FINITE_ABS_Z,
    )

    assert point.log_relative_likelihood is not None
    assert math.isfinite(point.log_relative_likelihood)
