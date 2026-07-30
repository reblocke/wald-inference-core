from __future__ import annotations

import wald_inference


def test_root_public_api_is_exact_and_every_name_resolves() -> None:
    expected = [
        "DEFAULT_CLAIM_DIRECTION",
        "DEFAULT_EFFECT_TYPE",
        "DEFAULT_NEAR_NULL_DELTA",
        "DEFAULT_SELECTION_RULE",
        "DesignMetric",
        "EFFECT_SPECS",
        "EffectSpec",
        "PrecisionTargetResult",
        "SelectionRuleSpec",
        "StandardErrorEstimate",
        "SupportComparison",
        "SupportInterval",
        "ValidationError",
        "WaldPointSummary",
        "WaldReconstruction",
        "__version__",
        "approximate_wald_ci_width",
        "build_grid",
        "compatibility_curve",
        "design_metrics_for_true_effects",
        "estimate_se",
        "estimate_se_details",
        "from_working_scale",
        "get_effect_spec",
        "information_scaled_standard_error",
        "legacy_critical_effect_distance",
        "legacy_critical_effect_markers",
        "log_relative_likelihood",
        "log_support_ratio",
        "max_safe_grid_span",
        "precision_target_results",
        "reconstruct_wald_from_95_ci",
        "relative_likelihood",
        "selection_rule_spec",
        "solve_required_delta_for_power",
        "solve_required_delta_for_type_m",
        "solve_required_delta_for_type_s",
        "solve_required_precision",
        "standardized_distance",
        "support_comparison",
        "support_interval",
        "support_interval_for_ratio",
        "support_ratio",
        "to_working_scale",
        "wald_point_summary",
    ]

    assert wald_inference.__all__ == expected
    assert all(hasattr(wald_inference, name) for name in expected)
