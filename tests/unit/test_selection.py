from __future__ import annotations

import math
import sys

import pytest
from scipy.stats import norm

from wald_inference.errors import ValidationError
from wald_inference.selection import selection_rule_spec


def test_all_six_selection_rules_preserve_exact_future_z_boundaries() -> None:
    alpha = 0.05
    two_sided_z = float(norm.isf(alpha / 2.0))
    one_sided_z = float(norm.isf(alpha))
    neg_inf = float("-inf")
    pos_inf = float("inf")

    specs = {
        "two_sided": selection_rule_spec(
            selection_rule="two_sided_p_lt_alpha",
            alpha=alpha,
        ),
        "one_sided_positive": selection_rule_spec(
            selection_rule="one_sided_positive_p_lt_alpha",
            alpha=alpha,
        ),
        "one_sided_negative": selection_rule_spec(
            selection_rule="one_sided_negative_p_lt_alpha",
            alpha=alpha,
        ),
        "directional_ci_positive": selection_rule_spec(
            selection_rule="ci_excludes_null_in_beneficial_direction",
            alpha=alpha,
            claim_direction="positive",
        ),
        "estimate_exceeds_positive": selection_rule_spec(
            selection_rule="estimate_exceeds_mcid_and_p_lt_alpha",
            alpha=alpha,
            claim_direction="positive",
            threshold_working=2.5,
        ),
        "ci_excludes_positive": selection_rule_spec(
            selection_rule="ci_excludes_mcid",
            alpha=alpha,
            claim_direction="positive",
            threshold_working=2.5,
        ),
    }

    assert specs["two_sided"].intervals == (
        (neg_inf, -two_sided_z),
        (two_sided_z, pos_inf),
    )
    assert specs["one_sided_positive"].intervals == ((one_sided_z, pos_inf),)
    assert specs["one_sided_negative"].intervals == ((neg_inf, -one_sided_z),)
    assert specs["directional_ci_positive"].intervals == ((two_sided_z, pos_inf),)
    assert specs["estimate_exceeds_positive"].intervals == ((2.5, pos_inf),)
    assert specs["ci_excludes_positive"].intervals == ((2.5 + two_sided_z, pos_inf),)


def test_negative_claim_rules_preserve_mirrored_threshold_boundaries() -> None:
    alpha = 0.05
    critical_z = float(norm.isf(alpha / 2.0))
    neg_inf = float("-inf")

    directional = selection_rule_spec(
        selection_rule="ci_excludes_null_in_beneficial_direction",
        alpha=alpha,
        claim_direction="negative",
    )
    estimate_exceeds = selection_rule_spec(
        selection_rule="estimate_exceeds_mcid_and_p_lt_alpha",
        alpha=alpha,
        claim_direction="negative",
        threshold_working=-2.5,
    )
    ci_excludes = selection_rule_spec(
        selection_rule="ci_excludes_mcid",
        alpha=alpha,
        claim_direction="negative",
        threshold_working=-2.5,
    )

    assert directional.intervals == ((neg_inf, -critical_z),)
    assert estimate_exceeds.intervals == ((neg_inf, -2.5),)
    assert ci_excludes.intervals == ((neg_inf, -2.5 - critical_z),)


def test_selected_alpha_ci_rule_labels_are_alpha_neutral() -> None:
    labels = [
        selection_rule_spec(
            selection_rule="ci_excludes_null_in_beneficial_direction",
            alpha=0.01,
            claim_direction="positive",
        ).label,
        selection_rule_spec(
            selection_rule="ci_excludes_mcid",
            alpha=0.01,
            claim_direction="positive",
            threshold_working=0.2,
        ).label,
    ]

    assert labels == [
        "CI at selected alpha excludes the null in the selected claim direction",
        "CI at selected alpha excludes the claim threshold",
    ]
    assert all("95%" not in label for label in labels)


def test_nonrequired_threshold_is_validated_and_retained_without_changing_rule() -> None:
    baseline = selection_rule_spec(selection_rule="two_sided_p_lt_alpha")
    supplied = selection_rule_spec(
        selection_rule="two_sided_p_lt_alpha",
        threshold_working=0.2,
    )

    assert supplied.threshold_working == 0.2
    assert supplied.threshold_delta is None
    assert supplied.intervals == baseline.intervals


def test_tiny_representable_alpha_succeeds_and_underflow_alpha_fails() -> None:
    spec = selection_rule_spec(alpha=1e-20)

    finite_bounds = [
        bound for interval in spec.intervals for bound in interval if not math.isinf(bound)
    ]
    assert finite_bounds
    assert all(math.isfinite(bound) for bound in finite_bounds)
    with pytest.raises(ValidationError, match="too small"):
        selection_rule_spec(alpha=1e-320)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "Design alpha"),
        ({"alpha": 1.0}, "Design alpha"),
        ({"alpha": "abc"}, "Design alpha"),
        ({"se": 0.0}, "Design standard error"),
        ({"se": "abc"}, "Design standard error"),
        ({"null_working": math.inf}, "Design null value"),
        ({"selection_rule": "one_sided"}, "Unsupported design selection rule"),
        ({"claim_direction": "sideways"}, "Design claim direction"),
        (
            {"selection_rule": "estimate_exceeds_mcid_and_p_lt_alpha"},
            "Design claim threshold is required",
        ),
        (
            {
                "selection_rule": "ci_excludes_mcid",
                "claim_direction": "positive",
                "threshold_working": -0.2,
            },
            "threshold above the null",
        ),
        (
            {
                "selection_rule": "ci_excludes_mcid",
                "claim_direction": "negative",
                "threshold_working": 0.2,
            },
            "threshold below the null",
        ),
    ],
)
def test_invalid_selection_inputs_preserve_validation_errors(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        selection_rule_spec(**kwargs)  # type: ignore[arg-type]


def test_threshold_distance_recovers_representable_subtraction_overflow() -> None:
    maximum = sys.float_info.max

    spec = selection_rule_spec(
        selection_rule="ci_excludes_mcid",
        null_working=-maximum,
        se=maximum,
        claim_direction="positive",
        threshold_working=maximum,
    )

    assert spec.threshold_delta == 2.0


def test_unrepresentable_threshold_distance_raises_validation_error() -> None:
    maximum = sys.float_info.max

    with pytest.raises(ValidationError, match="standardized distance.*finite"):
        selection_rule_spec(
            selection_rule="ci_excludes_mcid",
            null_working=-maximum,
            se=math.nextafter(2.0, 0.0),
            claim_direction="positive",
            threshold_working=maximum,
        )
