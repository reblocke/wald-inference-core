from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from wald_inference.effects import (
    EFFECT_SPECS,
    from_working_scale,
    get_effect_spec,
    to_working_scale,
)
from wald_inference.errors import ValidationError
from wald_inference.reconstruction import (
    ASYMMETRY_RELATIVE_TOLERANCE,
    ESTIMATE_MATCH_ABSOLUTE_TOLERANCE,
    ESTIMATE_MATCH_RELATIVE_TOLERANCE,
    asymmetry_warning,
    reconstruct_wald,
)


def test_effect_registry_is_the_exact_frozen_registry() -> None:
    assert list(EFFECT_SPECS) == [
        "odds_ratio",
        "risk_ratio",
        "hazard_ratio",
        "incidence_rate_ratio",
        "ratio_of_means",
        "mean_difference",
        "risk_difference",
        "rate_difference",
        "regression_coefficient",
    ]
    assert all(EFFECT_SPECS[key].default_null == 1.0 for key in list(EFFECT_SPECS)[:5])
    assert all(EFFECT_SPECS[key].default_null == 0.0 for key in list(EFFECT_SPECS)[5:])
    assert all(EFFECT_SPECS[key].working_scale == "log" for key in list(EFFECT_SPECS)[:5])
    assert all(EFFECT_SPECS[key].working_scale == "identity" for key in list(EFFECT_SPECS)[5:])


def test_get_effect_spec_rejects_unknown_key_with_registry_choices() -> None:
    with pytest.raises(ValidationError, match="Unsupported effect type.*odds_ratio"):
        get_effect_spec("not_an_effect")


def test_transformations_preserve_scalar_and_array_semantics() -> None:
    scalar = to_working_scale("odds_ratio", 2.0)
    ratio_zero_dimensional = to_working_scale("odds_ratio", np.asarray(2.0))
    identity_zero_dimensional = to_working_scale("mean_difference", np.asarray(2.0))
    sequence = to_working_scale("odds_ratio", [1.0, 2.0])
    array = from_working_scale(
        "odds_ratio",
        np.asarray([0.0, math.log(2.0)]),
    )

    assert type(scalar) is float
    assert type(ratio_zero_dimensional) is np.float64
    assert isinstance(identity_zero_dimensional, np.ndarray)
    assert identity_zero_dimensional.shape == ()
    assert isinstance(sequence, np.ndarray)
    assert isinstance(array, np.ndarray)
    assert sequence.tolist() == pytest.approx([0.0, math.log(2.0)])
    assert array.tolist() == pytest.approx([1.0, 2.0])


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_ratio_transformation_rejects_nonpositive_values(value: float) -> None:
    with pytest.raises(ValidationError, match="strictly positive"):
        to_working_scale("risk_ratio", value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_transformations_reject_nonfinite_inputs(value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        to_working_scale("mean_difference", value)
    with pytest.raises(ValidationError, match="finite"):
        from_working_scale("mean_difference", value)


def test_log_back_transform_rejects_overflow_instead_of_returning_infinity() -> None:
    with pytest.raises(ValidationError, match="Transformed values must be finite"):
        from_working_scale("odds_ratio", 1_000.0)


def test_additive_reconstruction_infers_working_midpoint() -> None:
    result = reconstruct_wald(
        "mean_difference",
        lower=0.11,
        upper=0.73,
    )

    assert result.estimate_source == "inferred_from_ci"
    assert result.estimate_display == pytest.approx(0.42)
    assert result.estimate_working == pytest.approx(0.42)
    assert result.null_display == 0.0
    assert result.default_null_applied is True
    assert result.se_method == "ci_width"
    assert result.warnings == ()


def test_ratio_reconstruction_infers_geometric_midpoint() -> None:
    result = reconstruct_wald(
        "odds_ratio",
        lower=1.2,
        upper=2.7,
    )

    expected = math.sqrt(1.2 * 2.7)
    assert result.estimate_display == pytest.approx(expected)
    assert result.estimate_working == pytest.approx(math.log(expected))
    assert result.null_display == 1.0
    assert result.null_working == 0.0


def test_provided_estimate_is_validation_only_and_can_warn_for_asymmetry() -> None:
    result = reconstruct_wald(
        "mean_difference",
        estimate=0.425,
        lower=0.11,
        upper=0.73,
    )

    assert result.estimate_source == "provided_validated"
    assert result.provided_estimate_display == pytest.approx(0.425)
    assert result.provided_estimate_working == pytest.approx(0.425)
    assert result.estimate_working == pytest.approx(0.42)
    assert result.standard_error == pytest.approx((0.73 - 0.11) / (2 * 1.959963984540054))
    assert result.relative_asymmetry > 0.02
    assert len(result.warnings) == 1
    assert "working scale" in result.warnings[0]


@pytest.mark.parametrize(
    ("effect_type", "wording"),
    [
        ("mean_difference", "working scale"),
        ("odds_ratio", "log scale"),
    ],
)
def test_asymmetry_warning_uses_the_frozen_strict_boundary(
    effect_type: str,
    wording: str,
) -> None:
    spec = get_effect_spec(effect_type)
    boundary = ASYMMETRY_RELATIVE_TOLERANCE

    assert boundary == 0.02
    assert asymmetry_warning(spec, boundary) is None
    warning = asymmetry_warning(spec, np.nextafter(boundary, np.inf))
    assert warning is not None
    assert wording in warning


def test_estimate_match_uses_the_frozen_strict_relative_tolerance() -> None:
    assert ESTIMATE_MATCH_RELATIVE_TOLERANCE == 0.02
    at_boundary = reconstruct_wald(
        "mean_difference",
        estimate=1.0,
        lower=-50.0,
        upper=50.0,
    )
    assert at_boundary.estimate_source == "provided_validated"

    with pytest.raises(ValidationError, match="inconsistent"):
        reconstruct_wald(
            "mean_difference",
            estimate=np.nextafter(1.0, np.inf),
            lower=-50.0,
            upper=50.0,
        )


def test_estimate_match_uses_the_frozen_strict_absolute_tolerance_floor() -> None:
    floor = ESTIMATE_MATCH_ABSOLUTE_TOLERANCE
    assert floor == 1e-12
    at_boundary = reconstruct_wald(
        "mean_difference",
        estimate=floor,
        lower=-1e-13,
        upper=1e-13,
    )
    assert at_boundary.estimate_source == "provided_validated"

    with pytest.raises(ValidationError, match="inconsistent"):
        reconstruct_wald(
            "mean_difference",
            estimate=np.nextafter(floor, np.inf),
            lower=-1e-13,
            upper=1e-13,
        )


def test_inconsistent_provided_estimate_is_rejected() -> None:
    with pytest.raises(ValidationError, match="inconsistent"):
        reconstruct_wald(
            "mean_difference",
            estimate=0.5,
            lower=0.11,
            upper=0.73,
        )


def test_reconstruction_records_explicit_null() -> None:
    result = reconstruct_wald(
        "regression_coefficient",
        lower=-0.2,
        upper=0.4,
        null_value=0.1,
    )

    assert result.null_display == pytest.approx(0.1)
    assert result.default_null_applied is False


def test_ratio_reconstruction_rejects_nonpositive_inputs() -> None:
    with pytest.raises(ValidationError, match="strictly positive"):
        reconstruct_wald("odds_ratio", lower=0.0, upper=2.0)
    with pytest.raises(ValidationError, match="strictly positive"):
        reconstruct_wald("odds_ratio", lower=1.0, upper=2.0, null_value=0.0)


def test_large_opposite_signed_ci_uses_finite_safe_midpoint() -> None:
    result = reconstruct_wald(
        "mean_difference",
        lower=-1e308,
        upper=1e308,
    )

    assert result.estimate_working == 0.0
    assert math.isfinite(result.standard_error)
    assert result.standard_error > 0.0


def test_largest_finite_opposite_signed_ci_keeps_reconstructed_se_finite() -> None:
    result = reconstruct_wald(
        "mean_difference",
        lower=-sys.float_info.max,
        upper=sys.float_info.max,
    )

    assert result.estimate_working == 0.0
    assert math.isfinite(result.standard_error)
    assert result.standard_error > 0.0
