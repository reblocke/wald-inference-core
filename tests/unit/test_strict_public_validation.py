from __future__ import annotations

from collections.abc import Callable

import pytest

from wald_inference import (
    ValidationError,
    build_grid,
    compatibility_curve,
    critical_effect_for_target_probability,
    design_metrics_for_true_effects,
    get_effect_spec,
    information_scaled_standard_error,
    legacy_critical_effect_markers,
    log_support_ratio,
    power_curve,
    reconstruct_wald_from_95_ci,
    selected_claim_probability,
    selection_rule_spec,
    support_comparison,
    support_interval,
    support_interval_for_ratio,
    support_ratio,
    to_working_scale,
    wald_point_summary,
)

HUGE_INTEGER = 10**10000


@pytest.mark.parametrize(
    "call",
    [
        lambda: to_working_scale("mean_difference", HUGE_INTEGER),
        lambda: reconstruct_wald_from_95_ci(
            "mean_difference",
            lower=0.0,
            upper=HUGE_INTEGER,
        ),
        lambda: build_grid(HUGE_INTEGER, 1.0),
        lambda: compatibility_curve([0.0], HUGE_INTEGER, 1.0),
        lambda: selected_claim_probability(
            HUGE_INTEGER,
            null_working=0.0,
            standard_error=1.0,
        ),
        lambda: power_curve(
            [HUGE_INTEGER],
            null_working=0.0,
            standard_error=1.0,
        ),
        lambda: critical_effect_for_target_probability(
            null_working=0.0,
            standard_error=HUGE_INTEGER,
        ),
        lambda: legacy_critical_effect_markers(HUGE_INTEGER, 1.0),
        lambda: selection_rule_spec(alpha=HUGE_INTEGER),
        lambda: design_metrics_for_true_effects(
            [0.0],
            null_working=HUGE_INTEGER,
            se=1.0,
        ),
        lambda: information_scaled_standard_error(HUGE_INTEGER, 1.0),
        lambda: support_comparison(
            HUGE_INTEGER,
            0.0,
            theta_hat=0.0,
            se=1.0,
        ),
        lambda: log_support_ratio(
            HUGE_INTEGER,
            0.0,
            theta_hat=0.0,
            se=1.0,
        ),
        lambda: support_ratio(
            HUGE_INTEGER,
            0.0,
            theta_hat=0.0,
            se=1.0,
        ),
        lambda: support_interval(HUGE_INTEGER, 1.0),
        lambda: support_interval_for_ratio(
            0.0,
            1.0,
            mle_to_bound_ratio=HUGE_INTEGER,
        ),
        lambda: wald_point_summary(0.0, 1.0, HUGE_INTEGER),
    ],
)
def test_oversized_integer_inputs_raise_public_validation_error(
    call: Callable[[], object],
) -> None:
    with pytest.raises(ValidationError):
        call()


def test_unhashable_effect_key_raises_public_validation_error() -> None:
    with pytest.raises(ValidationError, match="Unsupported effect type"):
        get_effect_spec([])  # type: ignore[arg-type]
