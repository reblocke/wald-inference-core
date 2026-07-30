from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

EffectFamily = Literal["additive", "ratio"]
WorkingScale = Literal["identity", "log"]
EstimateSource = Literal["inferred_from_ci", "provided_validated"]
DesignSelectionRule = Literal[
    "two_sided_p_lt_alpha",
    "one_sided_positive_p_lt_alpha",
    "one_sided_negative_p_lt_alpha",
    "ci_excludes_null_in_beneficial_direction",
    "estimate_exceeds_mcid_and_p_lt_alpha",
    "ci_excludes_mcid",
]
DesignClaimDirection = Literal["positive", "negative"]


@dataclass(frozen=True)
class EffectSpec:
    key: str
    label: str
    family: EffectFamily
    working_scale: WorkingScale
    default_null: float
    positive_only: bool


EFFECT_SPECS: dict[str, EffectSpec] = {
    "odds_ratio": EffectSpec(
        key="odds_ratio",
        label="Odds ratio",
        family="ratio",
        working_scale="log",
        default_null=1.0,
        positive_only=True,
    ),
    "risk_ratio": EffectSpec(
        key="risk_ratio",
        label="Risk ratio",
        family="ratio",
        working_scale="log",
        default_null=1.0,
        positive_only=True,
    ),
    "hazard_ratio": EffectSpec(
        key="hazard_ratio",
        label="Hazard ratio",
        family="ratio",
        working_scale="log",
        default_null=1.0,
        positive_only=True,
    ),
    "incidence_rate_ratio": EffectSpec(
        key="incidence_rate_ratio",
        label="Incidence rate ratio",
        family="ratio",
        working_scale="log",
        default_null=1.0,
        positive_only=True,
    ),
    "ratio_of_means": EffectSpec(
        key="ratio_of_means",
        label="Ratio of means",
        family="ratio",
        working_scale="log",
        default_null=1.0,
        positive_only=True,
    ),
    "mean_difference": EffectSpec(
        key="mean_difference",
        label="Mean difference",
        family="additive",
        working_scale="identity",
        default_null=0.0,
        positive_only=False,
    ),
    "risk_difference": EffectSpec(
        key="risk_difference",
        label="Risk difference",
        family="additive",
        working_scale="identity",
        default_null=0.0,
        positive_only=False,
    ),
    "rate_difference": EffectSpec(
        key="rate_difference",
        label="Rate difference",
        family="additive",
        working_scale="identity",
        default_null=0.0,
        positive_only=False,
    ),
    "regression_coefficient": EffectSpec(
        key="regression_coefficient",
        label="Regression coefficient",
        family="additive",
        working_scale="identity",
        default_null=0.0,
        positive_only=False,
    ),
}

DEFAULT_EFFECT_TYPE = "odds_ratio"


class CurveRequest(TypedDict, total=False):
    effect_type: str
    estimate: float | None
    lower: float
    upper: float
    null_value: float
    thresholds: list[float]
    display_range_lower: float | None
    display_range_upper: float | None
    display_natural_axis: bool
    grid_points: int
    show_cutoffs: bool
    design_enabled: bool
    design_alpha: float
    design_selection_rule: str
    design_claim_direction: str
    design_claim_threshold: float | None
    design_information_multiplier: float
    design_precision_target_effect: float | None
    design_target_power: float | None
    design_max_type_s: float | None
    design_max_type_m: float | None
    design_true_effects: list[float]
    design_plausible_range_lower: float | None
    design_plausible_range_upper: float | None


class ThresholdSupportPayload(TypedDict):
    threshold_display: float
    threshold_working: float
    relative_likelihood: float
    log_relative_likelihood: float
    likelihood_ratio_mle_to_threshold: float | None
    log_likelihood_ratio_mle_to_threshold: float
    likelihood_ratio_threshold_to_null: float | None
    log_likelihood_ratio_threshold_to_null: float | None
    direction_from_estimate: str
    direction_from_null: str


class SMinus2IntervalPayload(TypedDict):
    support_cutoff: float
    relative_likelihood_cutoff: float
    likelihood_ratio_mle_to_bound: float
    range_display: list[float]
    range_working: list[float]


class MetaPayload(TypedDict):
    effect_spec: dict[str, object]
    display_axis_scale: str
    estimate_source: EstimateSource
    default_null_applied: bool
    grid_points: int
    show_cutoffs: bool
    se_method: str
    relative_asymmetry: float
    thresholds_display: list[float]
    thresholds_working: list[float]
    display_range_active: bool
    display_range_display: list[float] | None
    display_range_working: list[float] | None
    threshold_support_summaries: list[ThresholdSupportPayload]
    s_minus_2_interval: SMinus2IntervalPayload


class SummaryPayload(TypedDict):
    estimate_display: float
    estimate_working: float
    ci_display: list[float]
    ci_working: list[float]
    null_display: float
    null_working: float
    working_scale_se: float
    null_relative_likelihood: float
    log_null_relative_likelihood: float | None
    likelihood_ratio_mle_to_null: float | None
    log_likelihood_ratio_mle_to_null: float | None
    two_sided_wald_p_value: float
    null_z_value: float | None
    critical_effect_markers_display: list[float]
    critical_effect_markers_working: list[float]
    critical_effect_distance_working: float


class GridPayload(TypedDict):
    effect_display: list[float]
    effect_working: list[float]
    z: list[float]
    compatibility: list[float]
    relative_likelihood: list[float]
    log_relative_likelihood: list[float]


class DesignConfigPayload(TypedDict):
    enabled: bool
    alpha: float
    selection_rule: DesignSelectionRule
    selection_rule_label: str
    claim_direction: DesignClaimDirection
    claim_threshold_display: float | None
    claim_threshold_working: float | None
    se_working: float
    current_se_working: float
    design_se_working: float
    information_multiplier: float
    current_ci_width_working: float
    approx_design_ci_width_working: float
    null_working: float
    estimate_working: float
    near_null_delta: float
    type_m_scale_note: str
    plausible_range_display: list[float] | None
    plausible_range_working: list[float] | None


class DesignGridPayload(TypedDict):
    true_effect_display: list[float]
    true_effect_working: list[float]
    delta: list[float]
    power: list[float]
    type_s: list[float | None]
    type_m: list[float | None]
    expected_selected_abs_z: list[float | None]
    observed_exaggeration: list[float | None]


class DesignScenarioPayload(TypedDict):
    label: str
    source: str
    true_effect_display: float
    true_effect_working: float
    delta: float
    power: float
    type_s: float | None
    type_m: float | None
    observed_exaggeration: float | None
    note: str | None


class DesignPrecisionTargetPayload(TypedDict):
    target: str
    requested_value: float
    target_effect_display: float
    target_effect_working: float
    required_se: float | None
    required_information_multiplier: float | None
    approx_95_ci_width_working: float | None
    achieved_power: float | None
    achieved_type_s: float | None
    achieved_type_m: float | None
    note: str


class DesignPayload(TypedDict):
    config: DesignConfigPayload
    grid: DesignGridPayload
    scenarios: list[DesignScenarioPayload]
    precision_targets: list[DesignPrecisionTargetPayload]
    warnings: list[str]


class CurveResponse(TypedDict):
    meta: MetaPayload
    summary: SummaryPayload
    warnings: list[str]
    grid: GridPayload
    design: DesignPayload | None
