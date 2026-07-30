"""Typed numerical utilities for documented one-parameter Wald inference."""

from .compatibility import (
    compatibility_curve,
    standardized_distance,
    wald_point_summary,
)
from .detectability import (
    legacy_critical_effect_distance,
    legacy_critical_effect_markers,
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
    relative_likelihood,
    support_comparison,
    support_interval,
)
from .precision import (
    approximate_wald_ci_width,
    information_scaled_standard_error,
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
    DesignMetric,
    EffectSpec,
    PrecisionTargetResult,
    SelectionRuleSpec,
    StandardErrorEstimate,
    SupportComparison,
    SupportInterval,
    WaldPointSummary,
    WaldReconstruction,
)

__version__ = "0.1.0"

__all__ = [
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
    "to_working_scale",
    "wald_point_summary",
]
