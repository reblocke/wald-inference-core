"""Typed numerical utilities for documented one-parameter Wald inference."""

from .compatibility import (
    compatibility_curve,
    standardized_distance,
    wald_point_summary,
)
from .detectability import (
    critical_effect_for_target_probability,
    legacy_critical_effect_distance,
    legacy_critical_effect_markers,
    power_curve,
    selected_claim_probability,
)
from .effects import (
    DEFAULT_EFFECT_TYPE,
    EFFECT_SPECS,
    from_working_scale,
    get_effect_spec,
    to_working_scale,
)
from .errors import ValidationError
from .grid import build_grid, max_safe_grid_span
from .likelihood import (
    log_relative_likelihood,
    log_support_ratio,
    relative_likelihood,
    support_comparison,
    support_interval,
    support_interval_for_ratio,
    support_ratio,
)
from .precision import (
    approximate_wald_ci_width,
    information_scaled_standard_error,
    joint_precision_result,
    precision_sensitivity,
    precision_target_results,
    solve_required_delta_for_power,
    solve_required_delta_for_type_m,
    solve_required_delta_for_type_s,
    solve_required_precision,
)
from .reconstruction import (
    estimate_se,
    estimate_se_details,
    reconstruct_wald_from_95_ci,
)
from .selection import (
    DEFAULT_CLAIM_DIRECTION,
    DEFAULT_SELECTION_RULE,
    selection_rule_spec,
)
from .type_sm import DEFAULT_NEAR_NULL_DELTA, design_metrics_for_true_effects
from .types import (
    CriticalEffectResult,
    DesignMetric,
    EffectSpec,
    JointPrecisionResult,
    PrecisionTargetResult,
    SelectionRuleSpec,
    StandardErrorEstimate,
    SupportComparison,
    SupportInterval,
    WaldPointSummary,
    WaldReconstruction,
)

__version__ = "0.4.2"

__all__ = [
    "DEFAULT_CLAIM_DIRECTION",
    "DEFAULT_EFFECT_TYPE",
    "DEFAULT_NEAR_NULL_DELTA",
    "DEFAULT_SELECTION_RULE",
    "CriticalEffectResult",
    "DesignMetric",
    "EFFECT_SPECS",
    "EffectSpec",
    "JointPrecisionResult",
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
    "critical_effect_for_target_probability",
    "design_metrics_for_true_effects",
    "estimate_se",
    "estimate_se_details",
    "from_working_scale",
    "get_effect_spec",
    "information_scaled_standard_error",
    "joint_precision_result",
    "legacy_critical_effect_distance",
    "legacy_critical_effect_markers",
    "log_relative_likelihood",
    "log_support_ratio",
    "max_safe_grid_span",
    "power_curve",
    "precision_sensitivity",
    "precision_target_results",
    "reconstruct_wald_from_95_ci",
    "relative_likelihood",
    "selected_claim_probability",
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
