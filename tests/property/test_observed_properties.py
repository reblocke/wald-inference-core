from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from wald_inference.compatibility import (
    LOG_MAX_FLOAT,
    compatibility_curve,
    wald_point_summary,
)
from wald_inference.effects import from_working_scale, to_working_scale
from wald_inference.likelihood import (
    log_support_ratio,
    relative_likelihood,
    support_interval,
    support_interval_for_ratio,
    support_ratio,
)
from wald_inference.reconstruction import reconstruct_wald

WORKING_SCALE_FLOATS = st.floats(
    min_value=-1_000_000.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
)
POSITIVE_FLOATS = st.floats(
    min_value=1e-4,
    max_value=20.0,
    allow_nan=False,
    allow_infinity=False,
)


@given(
    theta_hat=WORKING_SCALE_FLOATS,
    se=st.floats(min_value=0.05, max_value=3.0),
    distance=st.floats(min_value=0.0, max_value=6.0),
)
def test_observed_curves_are_symmetric(
    theta_hat: float,
    se: float,
    distance: float,
) -> None:
    left = theta_hat - (distance * se)
    right = theta_hat + (distance * se)

    assert compatibility_curve(left, theta_hat, se).item() == pytest.approx(
        compatibility_curve(right, theta_hat, se).item()
    )
    assert relative_likelihood(left, theta_hat, se).item() == pytest.approx(
        relative_likelihood(right, theta_hat, se).item()
    )


@given(
    theta_hat=WORKING_SCALE_FLOATS,
    se=st.floats(min_value=0.05, max_value=3.0),
    distance_1=st.floats(min_value=0.01, max_value=2.5),
    distance_2=st.floats(min_value=2.6, max_value=6.0),
)
def test_observed_curves_decline_with_distance(
    theta_hat: float,
    se: float,
    distance_1: float,
    distance_2: float,
) -> None:
    theta_1 = theta_hat + (distance_1 * se)
    theta_2 = theta_hat + (distance_2 * se)

    assert (
        compatibility_curve(theta_1, theta_hat, se).item()
        >= compatibility_curve(
            theta_2,
            theta_hat,
            se,
        ).item()
    )
    assert (
        relative_likelihood(theta_1, theta_hat, se).item()
        >= relative_likelihood(
            theta_2,
            theta_hat,
            se,
        ).item()
    )


@given(POSITIVE_FLOATS)
def test_ratio_scale_round_trip_is_stable(value: float) -> None:
    assert from_working_scale(
        "odds_ratio",
        to_working_scale("odds_ratio", value),
    ) == pytest.approx(value)


@given(
    midpoint=st.floats(
        min_value=-20.0,
        max_value=20.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    half_width=st.floats(
        min_value=1e-4,
        max_value=5.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_identity_and_log_reconstruction_are_working_scale_equivalent(
    midpoint: float,
    half_width: float,
) -> None:
    lower_working = midpoint - half_width
    upper_working = midpoint + half_width
    additive = reconstruct_wald(
        "regression_coefficient",
        lower=lower_working,
        upper=upper_working,
    )
    ratio = reconstruct_wald(
        "odds_ratio",
        lower=math.exp(lower_working),
        upper=math.exp(upper_working),
    )

    assert ratio.estimate_working == pytest.approx(additive.estimate_working)
    assert ratio.standard_error == pytest.approx(additive.standard_error)


@given(
    theta_hat=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    ),
    se=st.floats(min_value=1e-4, max_value=1e4),
    cutoff=st.floats(
        min_value=-100.0,
        max_value=0.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_support_interval_has_requested_log_support_at_finite_endpoints(
    theta_hat: float,
    se: float,
    cutoff: float,
) -> None:
    interval = support_interval(
        theta_hat,
        se,
        log_relative_likelihood_cutoff=cutoff,
    )
    assert interval.working_clipped is False
    distance = math.sqrt(-2.0 * cutoff)
    expected_lower = theta_hat - (distance * se)
    expected_upper = theta_hat + (distance * se)
    assert interval.range_working == pytest.approx((expected_lower, expected_upper))


@given(
    theta_hat=st.floats(
        min_value=-1e4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
    ),
    se=st.floats(min_value=0.05, max_value=10.0),
    candidate_a=st.floats(
        min_value=-1e4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
    ),
    candidate_b=st.floats(
        min_value=-1e4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_pairwise_log_support_is_antisymmetric(
    theta_hat: float,
    se: float,
    candidate_a: float,
    candidate_b: float,
) -> None:
    a_to_b = float(
        log_support_ratio(
            candidate_a,
            candidate_b,
            theta_hat=theta_hat,
            se=se,
        )
    )
    b_to_a = float(
        log_support_ratio(
            candidate_b,
            candidate_a,
            theta_hat=theta_hat,
            se=se,
        )
    )

    assert a_to_b == pytest.approx(-b_to_a)
    assert (
        float(
            log_support_ratio(
                candidate_a,
                candidate_a,
                theta_hat=theta_hat,
                se=se,
            )
        )
        == 0.0
    )


@given(
    theta_hat=st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
    ),
    se=st.floats(min_value=0.1, max_value=10.0),
    candidate_a=st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
    ),
    candidate_b=st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_support_ratio_matches_exponentiated_log_when_representable(
    theta_hat: float,
    se: float,
    candidate_a: float,
    candidate_b: float,
) -> None:
    log_ratio = float(
        log_support_ratio(
            candidate_a,
            candidate_b,
            theta_hat=theta_hat,
            se=se,
        )
    )
    ratio = support_ratio(
        candidate_a,
        candidate_b,
        theta_hat=theta_hat,
        se=se,
    )

    if log_ratio > LOG_MAX_FLOAT:
        assert ratio is None
    else:
        assert ratio == pytest.approx(math.exp(log_ratio))


@given(
    theta_hat=st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    se=st.floats(min_value=0.05, max_value=10.0),
    ratio=st.floats(
        min_value=1.000001,
        max_value=1e100,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_ratio_support_interval_endpoints_have_requested_pairwise_support(
    theta_hat: float,
    se: float,
    ratio: float,
) -> None:
    interval = support_interval_for_ratio(
        theta_hat,
        se,
        mle_to_bound_ratio=ratio,
    )

    assert interval.working_clipped is False
    for endpoint in interval.range_working:
        endpoint_log_ratio = float(
            log_support_ratio(
                theta_hat,
                endpoint,
                theta_hat=theta_hat,
                se=se,
            )
        )
        assert endpoint_log_ratio == pytest.approx(math.log(ratio))


@given(
    theta_hat=st.floats(
        min_value=-1e5,
        max_value=1e5,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
    se=st.floats(min_value=0.05, max_value=3.0),
    candidate=st.floats(
        min_value=-1e5,
        max_value=1e5,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)
def test_point_likelihood_ratio_matches_inverse_when_representable(
    theta_hat: float,
    se: float,
    candidate: float,
) -> None:
    summary = wald_point_summary(theta_hat, se, candidate)

    if summary.relative_likelihood == 0.0:
        assert summary.likelihood_ratio_mle_to_candidate is None
        if summary.log_likelihood_ratio_mle_to_candidate is not None:
            assert summary.log_likelihood_ratio_mle_to_candidate > LOG_MAX_FLOAT
    elif summary.likelihood_ratio_mle_to_candidate is None:
        assert summary.log_likelihood_ratio_mle_to_candidate is not None
        assert summary.log_likelihood_ratio_mle_to_candidate > LOG_MAX_FLOAT
    else:
        assert summary.likelihood_ratio_mle_to_candidate == pytest.approx(
            1.0 / summary.relative_likelihood
        )


@given(
    values=st.lists(
        st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=20,
    )
)
def test_vectorized_observed_outputs_are_finite_and_shape_preserving(
    values: list[float],
) -> None:
    array = np.asarray(values)
    compatibility = compatibility_curve(array, theta_hat=0.0, se=2.0)
    likelihood = relative_likelihood(array, theta_hat=0.0, se=2.0)

    assert compatibility.shape == array.shape
    assert likelihood.shape == array.shape
    assert np.isfinite(compatibility).all()
    assert np.isfinite(likelihood).all()
