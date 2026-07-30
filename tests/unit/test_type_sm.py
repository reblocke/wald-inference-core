from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest
from scipy.stats import norm

from wald_inference.errors import ValidationError
from wald_inference.precision import (
    solve_required_delta_for_power,
    solve_required_delta_for_type_m,
    solve_required_delta_for_type_s,
)
from wald_inference.type_sm import design_metrics_for_true_effects


def metric_for_delta(delta: float, *, alpha: float = 0.05):
    return design_metrics_for_true_effects(
        [delta],
        null_working=0.0,
        se=1.0,
        estimate_working=2.0,
        alpha=alpha,
    )[0]


def test_selected_claim_probability_at_null_matches_alpha_and_power_alias() -> None:
    metric = metric_for_delta(0.0, alpha=0.05)

    assert metric.selected_claim_probability == pytest.approx(0.05)
    assert metric.power == metric.selected_claim_probability
    assert metric.type_s is None
    assert metric.type_m is None
    assert metric.observed_exaggeration is None
    assert metric.expected_selected_abs_z is not None
    with pytest.raises(FrozenInstanceError):
        metric.power = 0.2  # type: ignore[misc]


def test_probability_type_s_and_type_m_are_symmetric() -> None:
    positive = metric_for_delta(1.5)
    negative = metric_for_delta(-1.5)

    assert positive.selected_claim_probability == pytest.approx(negative.selected_claim_probability)
    assert positive.type_s == pytest.approx(negative.type_s)
    assert positive.type_m == pytest.approx(negative.type_m)
    assert positive.expected_selected_abs_z == pytest.approx(negative.expected_selected_abs_z)


def test_type_s_uses_wrong_sign_tail() -> None:
    alpha = 0.05
    critical_z = norm.ppf(1.0 - (alpha / 2.0))
    positive = metric_for_delta(1.0, alpha=alpha)
    negative = metric_for_delta(-1.0, alpha=alpha)

    positive_lower_tail = norm.cdf(-critical_z - 1.0)
    positive_upper_tail = norm.sf(critical_z - 1.0)
    negative_lower_tail = norm.cdf(-critical_z + 1.0)
    negative_upper_tail = norm.sf(critical_z + 1.0)

    assert positive.type_s == pytest.approx(
        positive_lower_tail / (positive_lower_tail + positive_upper_tail)
    )
    assert negative.type_s == pytest.approx(
        negative_upper_tail / (negative_lower_tail + negative_upper_tail)
    )


def test_one_sided_positive_selection_rule_matches_normal_tail() -> None:
    alpha = 0.05
    delta = 0.7
    critical_z = norm.isf(alpha)

    [metric] = design_metrics_for_true_effects(
        [delta],
        null_working=0.0,
        se=1.0,
        alpha=alpha,
        selection_rule="one_sided_positive_p_lt_alpha",
    )
    [wrong_direction_metric] = design_metrics_for_true_effects(
        [-delta],
        null_working=0.0,
        se=1.0,
        alpha=alpha,
        selection_rule="one_sided_positive_p_lt_alpha",
    )

    assert metric.selected_claim_probability == pytest.approx(norm.sf(critical_z - delta))
    assert metric.type_s == pytest.approx(0.0)
    assert wrong_direction_metric.type_s == pytest.approx(1.0)


def test_directional_ci_rule_uses_two_sided_ci_tail_in_claim_direction() -> None:
    alpha = 0.05
    critical_z = norm.isf(alpha / 2.0)

    [positive] = design_metrics_for_true_effects(
        [1.0],
        null_working=0.0,
        se=1.0,
        alpha=alpha,
        selection_rule="ci_excludes_null_in_beneficial_direction",
        claim_direction="positive",
    )
    [negative] = design_metrics_for_true_effects(
        [-1.0],
        null_working=0.0,
        se=1.0,
        alpha=alpha,
        selection_rule="ci_excludes_null_in_beneficial_direction",
        claim_direction="negative",
    )

    assert positive.selected_claim_probability == pytest.approx(norm.sf(critical_z - 1.0))
    assert negative.selected_claim_probability == pytest.approx(norm.cdf(-critical_z + 1.0))
    assert positive.type_s == pytest.approx(0.0)
    assert negative.type_s == pytest.approx(0.0)


def test_threshold_selection_rules_match_exact_tail_boundaries() -> None:
    alpha = 0.05
    critical_z = norm.isf(alpha / 2.0)

    [estimate_exceeds] = design_metrics_for_true_effects(
        [3.0],
        null_working=0.0,
        se=1.0,
        alpha=alpha,
        selection_rule="estimate_exceeds_mcid_and_p_lt_alpha",
        claim_direction="positive",
        threshold_working=2.5,
    )
    [ci_excludes] = design_metrics_for_true_effects(
        [5.0],
        null_working=0.0,
        se=1.0,
        alpha=alpha,
        selection_rule="ci_excludes_mcid",
        claim_direction="positive",
        threshold_working=2.5,
    )

    assert estimate_exceeds.selected_claim_probability == pytest.approx(norm.sf(2.5 - 3.0))
    assert ci_excludes.selected_claim_probability == pytest.approx(
        norm.sf((2.5 + critical_z) - 5.0)
    )


def test_large_true_effect_has_low_type_s_and_little_expected_exaggeration() -> None:
    metric = metric_for_delta(8.0)

    assert metric.selected_claim_probability > 0.99
    assert metric.type_s is not None and metric.type_s < 1e-20
    assert metric.type_m == pytest.approx(1.0, rel=0.02)


def test_near_null_boundary_is_inclusive_and_next_float_is_defined() -> None:
    tolerance = 1e-12
    [boundary, outside] = design_metrics_for_true_effects(
        [tolerance, math.nextafter(tolerance, math.inf)],
        null_working=0.0,
        se=1.0,
        estimate_working=2.0,
        near_null_delta=tolerance,
    )

    assert boundary.type_s is None
    assert boundary.type_m is None
    assert boundary.observed_exaggeration is None
    assert outside.type_s is not None
    assert outside.type_m is not None
    assert outside.observed_exaggeration is not None


def test_observed_exaggeration_uses_working_scale_distance_from_null() -> None:
    [metric] = design_metrics_for_true_effects(
        [0.5],
        null_working=0.0,
        se=0.25,
        estimate_working=1.0,
    )

    assert metric.observed_exaggeration == pytest.approx(2.0)


def test_expected_selected_abs_z_matches_exact_formula() -> None:
    delta = 1.25
    alpha = 0.01
    critical_z = norm.ppf(1.0 - (alpha / 2.0))
    metric = metric_for_delta(delta, alpha=alpha)

    upper_tail = norm.sf(critical_z - delta)
    lower_tail = norm.cdf(-critical_z - delta)
    numerator = (
        delta * (upper_tail - lower_tail)
        + norm.pdf(critical_z - delta)
        + norm.pdf(-critical_z - delta)
    )

    assert metric.expected_selected_abs_z == pytest.approx(
        numerator / metric.selected_claim_probability
    )
    assert metric.type_m == pytest.approx(metric.expected_selected_abs_z / abs(delta))


def test_tiny_alpha_uses_survival_quantile_without_dividing_by_zero() -> None:
    [metric] = design_metrics_for_true_effects(
        [1.0],
        null_working=0.0,
        se=1.0,
        alpha=1e-20,
    )

    assert metric.selected_claim_probability > 0
    assert metric.type_s is not None
    assert metric.type_m is not None


def test_too_small_alpha_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="too small"):
        design_metrics_for_true_effects([1.0], null_working=0.0, se=1.0, alpha=1e-320)


def test_required_delta_solvers_hit_requested_targets() -> None:
    alpha = 0.05
    power_delta = solve_required_delta_for_power(alpha, 0.8)
    type_s_delta = solve_required_delta_for_type_s(alpha, 0.01)
    type_m_delta = solve_required_delta_for_type_m(alpha, 1.25)

    assert metric_for_delta(power_delta, alpha=alpha).selected_claim_probability == pytest.approx(
        0.8
    )
    assert metric_for_delta(type_s_delta, alpha=alpha).type_s == pytest.approx(0.01)
    assert metric_for_delta(type_m_delta, alpha=alpha).type_m == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: design_metrics_for_true_effects(["abc"], null_working=0.0, se=1.0),
            "true effects",
        ),
        (
            lambda: design_metrics_for_true_effects(
                [0.5],
                null_working=0.0,
                se="abc",  # type: ignore[arg-type]
            ),
            "standard error",
        ),
        (
            lambda: design_metrics_for_true_effects(
                [0.5],
                null_working=0.0,
                se=1.0,
                near_null_delta=-1.0,
            ),
            "near-null",
        ),
        (
            lambda: design_metrics_for_true_effects(
                [0.5],
                null_working=0.0,
                se=1.0,
                estimate_working=math.inf,
            ),
            "estimate",
        ),
        (
            lambda: design_metrics_for_true_effects([0.5], null_working=math.nan, se=1.0),
            "null value",
        ),
        (
            lambda: solve_required_delta_for_power(0.05, "abc"),  # type: ignore[arg-type]
            "Target power",
        ),
    ],
)
def test_malformed_public_type_sm_inputs_raise_validation_error(
    call: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        call()


def test_multidimensional_true_effects_raise_validation_error() -> None:
    with pytest.raises(
        ValidationError,
        match="Design true effects must be supplied as numeric values.",
    ):
        design_metrics_for_true_effects(
            [[1.0, 2.0]],
            null_working=0.0,
            se=1.0,
        )


def test_standardized_distances_preserve_ordinary_direct_arithmetic() -> None:
    true_effects = [-3.5, 0.0, 2.25]
    null_value = 0.25
    se = 0.5

    metrics = design_metrics_for_true_effects(
        true_effects,
        null_working=null_value,
        se=se,
    )

    assert [metric.delta for metric in metrics] == [
        (true_effect - null_value) / se for true_effect in true_effects
    ]


def test_standardized_distances_recover_representable_subtraction_overflow() -> None:
    maximum = sys.float_info.max

    [max_distance] = design_metrics_for_true_effects(
        [maximum],
        null_working=-maximum,
        se=2.0,
    )
    [distance_of_two] = design_metrics_for_true_effects(
        [maximum],
        null_working=-maximum,
        se=maximum,
    )

    assert max_distance.delta == maximum
    assert distance_of_two.delta == 2.0


@pytest.mark.parametrize(
    ("null_value", "se"),
    [
        (-sys.float_info.max, math.nextafter(2.0, 0.0)),
        (0.0, 0.5),
    ],
)
def test_unrepresentable_standardized_distances_raise_validation_error(
    null_value: float,
    se: float,
) -> None:
    with pytest.raises(ValidationError, match="standardized distance.*finite"):
        design_metrics_for_true_effects(
            [sys.float_info.max],
            null_working=null_value,
            se=se,
        )


def test_standardized_distance_preserves_smallest_subnormal_result() -> None:
    smallest_subnormal = math.ulp(0.0)

    [metric] = design_metrics_for_true_effects(
        [smallest_subnormal],
        null_working=-smallest_subnormal,
        se=2.0,
        near_null_delta=0.0,
    )

    assert metric.delta == smallest_subnormal


@pytest.mark.parametrize(
    ("estimate", "expected"),
    [
        (0.0, 0.5),
        (sys.float_info.max, 1.0),
    ],
)
def test_observed_exaggeration_uses_finite_overflow_distances(
    estimate: float,
    expected: float,
) -> None:
    maximum = sys.float_info.max

    [metric] = design_metrics_for_true_effects(
        [maximum],
        null_working=-maximum,
        se=maximum,
        estimate_working=estimate,
    )

    assert metric.observed_exaggeration == expected


def test_observed_exaggeration_does_not_require_representable_raw_deltas() -> None:
    maximum = sys.float_info.max

    [metric] = design_metrics_for_true_effects(
        [maximum / 2.0],
        null_working=-maximum,
        se=1.75,
        estimate_working=maximum,
    )

    assert metric.delta == pytest.approx(1.5408798298819848e308)
    assert metric.observed_exaggeration == pytest.approx(4.0 / 3.0)


def test_unrepresentable_observed_exaggeration_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="observed exaggeration.*finite"):
        design_metrics_for_true_effects(
            [1e-308],
            null_working=0.0,
            se=1e-308,
            estimate_working=sys.float_info.max,
            near_null_delta=0.0,
        )
