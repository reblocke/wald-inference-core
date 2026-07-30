from __future__ import annotations

import math
import sys

import numpy as np
import pytest

import wald_inference
import wald_inference.compatibility as compatibility_module
import wald_inference.detectability as detectability_module
import wald_inference.grid as grid_module
import wald_inference.precision as precision_module
import wald_inference.reconstruction as reconstruction_module
from wald_inference import legacy
from wald_inference.compatibility import standardized_distance
from wald_inference.effects import get_effect_spec, to_working_scale
from wald_inference.errors import ValidationError
from wald_inference.grid import build_grid
from wald_inference.likelihood import support_comparison, support_interval
from wald_inference.reconstruction import estimate_se

HUGE_INTEGER = 10**10000


def test_legacy_module_is_not_added_to_root_public_surface() -> None:
    assert "legacy" not in wald_inference.__all__
    assert "confidence_curve" not in wald_inference.__all__
    assert wald_inference.estimate_se is not legacy.estimate_se
    assert wald_inference.build_grid is not legacy.build_grid


def test_legacy_public_api_is_exact_and_every_name_resolves() -> None:
    expected = [
        "ASYMMETRY_RELATIVE_TOLERANCE",
        "DEFAULT_GRID_POINTS",
        "DEFAULT_SOLVER_TOLERANCE",
        "DEFAULT_SPAN_MULTIPLIER",
        "ESTIMATE_MATCH_ABSOLUTE_TOLERANCE",
        "ESTIMATE_MATCH_RELATIVE_TOLERANCE",
        "GRID_EXPANSION_PADDING_MULTIPLIER",
        "LOG_MAX_FLOAT",
        "MAX_FINITE_ABS_Z",
        "MAX_FINITE_SPAN",
        "MAX_FLOAT",
        "MAX_INFORMATION_MULTIPLIER",
        "Z80",
        "Z975",
        "asymmetry_warning",
        "build_grid",
        "confidence_curve",
        "estimate_se",
        "from_working_scale",
        "log_relative_likelihood",
        "relative_likelihood",
        "summaries",
        "to_working_scale",
    ]

    assert legacy.__all__ == expected
    assert all(hasattr(legacy, name) for name in expected)


def test_legacy_adapter_constants_and_warning_are_exact_canonical_reexports() -> None:
    expected_sources = {
        "ASYMMETRY_RELATIVE_TOLERANCE": reconstruction_module,
        "DEFAULT_GRID_POINTS": grid_module,
        "DEFAULT_SOLVER_TOLERANCE": precision_module,
        "DEFAULT_SPAN_MULTIPLIER": grid_module,
        "ESTIMATE_MATCH_ABSOLUTE_TOLERANCE": reconstruction_module,
        "ESTIMATE_MATCH_RELATIVE_TOLERANCE": reconstruction_module,
        "GRID_EXPANSION_PADDING_MULTIPLIER": grid_module,
        "LOG_MAX_FLOAT": compatibility_module,
        "MAX_FINITE_ABS_Z": compatibility_module,
        "MAX_FINITE_SPAN": grid_module,
        "MAX_FLOAT": compatibility_module,
        "MAX_INFORMATION_MULTIPLIER": precision_module,
        "Z80": detectability_module,
        "Z975": reconstruction_module,
        "asymmetry_warning": reconstruction_module,
    }

    for name, source_module in expected_sources.items():
        assert getattr(legacy, name) is getattr(source_module, name)

    assert legacy.ASYMMETRY_RELATIVE_TOLERANCE == 0.02
    assert legacy.DEFAULT_GRID_POINTS == 801
    assert legacy.DEFAULT_SOLVER_TOLERANCE == 1e-8
    assert legacy.DEFAULT_SPAN_MULTIPLIER == 4.5
    assert legacy.ESTIMATE_MATCH_ABSOLUTE_TOLERANCE == 1e-12
    assert legacy.ESTIMATE_MATCH_RELATIVE_TOLERANCE == 0.02
    assert legacy.GRID_EXPANSION_PADDING_MULTIPLIER == 0.25
    assert legacy.LOG_MAX_FLOAT == 709.782712893384
    assert legacy.MAX_FINITE_ABS_Z == 1.3407807929942596e154
    assert legacy.MAX_FINITE_SPAN == 4.4942328371557893e307
    assert legacy.MAX_FLOAT == 1.7976931348623157e308
    assert legacy.MAX_INFORMATION_MULTIPLIER == 1e12
    assert legacy.Z80 == 0.8416212335729143
    assert legacy.Z975 == 1.959963984540054


def test_legacy_transformations_preserve_scalar_and_array_types() -> None:
    scalar = legacy.to_working_scale("odds_ratio", 2.0)
    sequence = legacy.to_working_scale("odds_ratio", [1.0, 2.0])
    zero_dimensional = legacy.to_working_scale("odds_ratio", np.asarray(2.0))

    assert type(scalar) is float
    assert scalar == math.log(2.0)
    assert isinstance(sequence, np.ndarray)
    assert sequence.tolist() == [0.0, math.log(2.0)]
    assert isinstance(zero_dimensional, np.float64)
    assert zero_dimensional.item() == math.log(2.0)


def test_legacy_transformation_keeps_raw_malformed_input_exceptions() -> None:
    with pytest.raises(ValueError, match="could not convert string to float"):
        legacy.to_working_scale("mean_difference", "abc")
    with pytest.raises(OverflowError, match="too large to convert to float"):
        legacy.to_working_scale("mean_difference", [HUGE_INTEGER])
    with pytest.raises(TypeError, match="unhashable type"):
        legacy.to_working_scale([], 1.0)  # type: ignore[arg-type]


def test_canonical_transformation_normalizes_malformed_input_exceptions() -> None:
    with pytest.raises(ValidationError, match="numeric and finite"):
        to_working_scale("mean_difference", [HUGE_INTEGER])
    with pytest.raises(ValidationError, match="Unsupported effect type"):
        get_effect_spec([])  # type: ignore[arg-type]


def test_legacy_log_back_transform_preserves_overflow_value_and_warning() -> None:
    with pytest.warns(RuntimeWarning, match="overflow encountered in exp"):
        result = legacy.from_working_scale("odds_ratio", 1_000.0)

    assert type(result) is float
    assert result == math.inf
    with pytest.raises(ValidationError, match="Transformed values must be finite"):
        wald_inference.from_working_scale("odds_ratio", 1_000.0)


def test_legacy_estimate_se_preserves_frozen_value_and_raw_coercion() -> None:
    assert legacy.estimate_se(0.42, 0.11, 0.73) == 0.15816617164664273
    with pytest.raises(TypeError, match="can't multiply sequence"):
        legacy.estimate_se("0", 0.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="inferred CI midpoint"):
        legacy.estimate_se(0.0, math.nan, 1.0)


def test_legacy_estimate_se_preserves_max_range_warning_and_scalar_type() -> None:
    maximum = np.float64(sys.float_info.max)

    with pytest.warns(RuntimeWarning, match="overflow encountered in reduce"):
        result = legacy.estimate_se(0.0, -maximum, maximum)

    assert isinstance(result, np.float64)
    assert result == np.float64(9.172072288278203e307)
    assert math.isfinite(estimate_se(0.0, -maximum, maximum))


def test_legacy_grid_preserves_frozen_ordinary_and_even_point_behavior() -> None:
    assert legacy.build_grid(0.0, 1.0, n=5).tolist() == [
        -4.5,
        -2.25,
        0.0,
        2.25,
        4.5,
    ]
    assert legacy.build_grid(0.0, 1.0, n=6).tolist() == [
        -4.5,
        -3.0,
        -1.5,
        0.0,
        1.5,
        3.0,
        4.5,
    ]


def test_legacy_grid_preserves_nonfinite_direct_call_outputs() -> None:
    assert np.isnan(legacy.build_grid(0.0, math.nan, n=5)).all()
    assert legacy.build_grid(0.0, 1.0, max_span=math.nan, n=5).tolist() == [
        -4.5,
        -2.25,
        0.0,
        2.25,
        4.5,
    ]
    with pytest.warns(RuntimeWarning):
        infinite_grid = legacy.build_grid(0.0, math.inf, n=5)
    assert np.isnan(infinite_grid[:-1]).all()
    assert infinite_grid[-1] == math.inf


def test_legacy_grid_keeps_raw_malformed_include_exception() -> None:
    with pytest.raises(ValueError, match="could not convert string to float"):
        legacy.build_grid(0.0, 1.0, include_values=["x"], n=5)
    with pytest.raises(OverflowError, match="too large to convert to float"):
        legacy.build_grid(0.0, 1.0, include_values=[HUGE_INTEGER], n=5)
    with pytest.raises(ValidationError, match="Included grid values must be finite"):
        build_grid(0.0, 1.0, include_values=[HUGE_INTEGER], n=5)


def test_legacy_curves_preserve_frozen_values_and_scalar_types() -> None:
    confidence = legacy.confidence_curve(1.0, theta_hat=0.0, se=1.0)
    likelihood = legacy.relative_likelihood(1.0, theta_hat=0.0, se=1.0)
    log_likelihood = legacy.log_relative_likelihood(1.0, theta_hat=0.0, se=1.0)

    assert isinstance(confidence, np.float64)
    assert confidence == pytest.approx(0.31731050786291415)
    assert isinstance(likelihood, np.float64)
    assert likelihood == math.exp(-0.5)
    assert isinstance(log_likelihood, np.float64)
    assert log_likelihood == -0.5


def test_legacy_curves_preserve_infinite_se_behavior() -> None:
    assert legacy.confidence_curve(1.0, theta_hat=0.0, se=math.inf) == 1.0
    assert legacy.relative_likelihood(1.0, theta_hat=0.0, se=math.inf) == 1.0
    assert (
        math.copysign(
            1.0,
            legacy.log_relative_likelihood(1.0, theta_hat=0.0, se=math.inf),
        )
        == -1.0
    )


def test_legacy_curves_preserve_nan_and_malformed_exceptions() -> None:
    with pytest.raises(ValidationError, match="Standardized distance exceeds"):
        legacy.confidence_curve(1.0, theta_hat=0.0, se=math.nan)
    with pytest.raises(ValueError, match="could not convert string to float"):
        legacy.confidence_curve("x", theta_hat=0.0, se=1.0)  # type: ignore[arg-type]


def test_legacy_likelihoods_preserve_unrepresentable_log_behavior() -> None:
    maximum = sys.float_info.max

    with pytest.warns(RuntimeWarning, match="overflow encountered in square"):
        relative = legacy.relative_likelihood(maximum, theta_hat=0.0, se=1.0)
    with pytest.warns(RuntimeWarning, match="overflow encountered in square"):
        log_relative = legacy.log_relative_likelihood(maximum, theta_hat=0.0, se=1.0)

    assert relative == 0.0
    assert log_relative == -math.inf


def test_legacy_summaries_need_wrapper_for_nonfinite_direct_calls() -> None:
    assert legacy.summaries(0.0, math.inf, 1.0) == {
        "null_relative_likelihood": 1.0,
        "log_null_relative_likelihood": -0.0,
        "likelihood_ratio_mle_to_null": 1.0,
        "log_likelihood_ratio_mle_to_null": 0.0,
        "two_sided_wald_p_value": 1.0,
        "null_z_value": 0.0,
    }
    information_free = {
        "null_relative_likelihood": 0.0,
        "log_null_relative_likelihood": None,
        "likelihood_ratio_mle_to_null": None,
        "log_likelihood_ratio_mle_to_null": None,
        "two_sided_wald_p_value": 0.0,
        "null_z_value": None,
    }
    assert legacy.summaries(0.0, math.nan, 1.0) == information_free
    assert legacy.summaries(0.0, 1.0, math.nan) == information_free


def test_canonical_functions_reject_huge_integer_coercions_across_modules() -> None:
    calls = [
        lambda: estimate_se(HUGE_INTEGER, 0.0, 1.0),
        lambda: standardized_distance([HUGE_INTEGER], 0.0, 1.0),
        lambda: support_comparison(
            HUGE_INTEGER,
            0.0,
            theta_hat=0.0,
            se=1.0,
        ),
        lambda: support_interval(HUGE_INTEGER, 1.0),
    ]

    for call in calls:
        with pytest.raises(ValidationError):
            call()
