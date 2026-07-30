"""Typed domain objects shared by the public numerical modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

EffectFamily: TypeAlias = Literal["additive", "ratio"]
WorkingScale: TypeAlias = Literal["identity", "log"]
EstimateSource: TypeAlias = Literal["inferred_from_ci", "provided_validated"]
StandardErrorMethod: TypeAlias = Literal["ci_width", "mean_side_se"]
SelectionRule: TypeAlias = Literal[
    "two_sided_p_lt_alpha",
    "one_sided_positive_p_lt_alpha",
    "one_sided_negative_p_lt_alpha",
    "ci_excludes_null_in_beneficial_direction",
    "estimate_exceeds_mcid_and_p_lt_alpha",
    "ci_excludes_mcid",
]
ClaimDirection: TypeAlias = Literal["positive", "negative"]


@dataclass(frozen=True)
class EffectSpec:
    """Definition of one supported effect measure and its working scale."""

    key: str
    label: str
    family: EffectFamily
    working_scale: WorkingScale
    default_null: float
    positive_only: bool


@dataclass(frozen=True)
class StandardErrorEstimate:
    """Working-scale standard-error reconstruction details."""

    se: float
    method: StandardErrorMethod
    se_lower: float
    se_upper: float
    se_width: float
    relative_asymmetry: float


@dataclass(frozen=True)
class WaldReconstruction:
    """Validated Wald reconstruction from a reported estimate and 95% CI."""

    effect_spec: EffectSpec
    estimate_display: float
    estimate_working: float
    estimate_source: EstimateSource
    provided_estimate_display: float | None
    provided_estimate_working: float | None
    lower_display: float
    upper_display: float
    lower_working: float
    upper_working: float
    null_display: float
    null_working: float
    default_null_applied: bool
    standard_error: float
    se_method: StandardErrorMethod
    se_lower: float
    se_upper: float
    se_width: float
    relative_asymmetry: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class WaldPointSummary:
    """Observed Wald quantities for one candidate value versus the MLE."""

    candidate_working: float
    relative_likelihood: float
    log_relative_likelihood: float | None
    likelihood_ratio_mle_to_candidate: float | None
    log_likelihood_ratio_mle_to_candidate: float | None
    two_sided_wald_p_value: float
    z_value: float | None

    @property
    def null_relative_likelihood(self) -> float:
        """Legacy name when the candidate is the null value."""

        return self.relative_likelihood

    @property
    def log_null_relative_likelihood(self) -> float | None:
        """Legacy name when the candidate is the null value."""

        return self.log_relative_likelihood

    @property
    def likelihood_ratio_mle_to_null(self) -> float | None:
        """Legacy name when the candidate is the null value."""

        return self.likelihood_ratio_mle_to_candidate

    @property
    def log_likelihood_ratio_mle_to_null(self) -> float | None:
        """Legacy name when the candidate is the null value."""

        return self.log_likelihood_ratio_mle_to_candidate

    @property
    def null_z_value(self) -> float | None:
        """Legacy name when the candidate is the null value."""

        return self.z_value


@dataclass(frozen=True)
class SupportComparison:
    """Relative support for a candidate and a reference value."""

    candidate_working: float
    reference_working: float
    relative_likelihood: float
    log_relative_likelihood: float
    likelihood_ratio_mle_to_candidate: float | None
    log_likelihood_ratio_mle_to_candidate: float
    likelihood_ratio_candidate_to_reference: float | None
    log_likelihood_ratio_candidate_to_reference: float


@dataclass(frozen=True)
class SupportInterval:
    """Working-scale interval meeting a log-relative-likelihood cutoff."""

    support_cutoff: float
    relative_likelihood_cutoff: float
    likelihood_ratio_mle_to_bound: float | None
    lower_working: float
    upper_working: float
    lower_clipped: bool
    upper_clipped: bool

    @property
    def range_working(self) -> tuple[float, float]:
        """Return the lower and upper working-scale endpoints."""

        return self.lower_working, self.upper_working

    @property
    def working_clipped(self) -> bool:
        """Whether either endpoint was clipped to the finite float range."""

        return self.lower_clipped or self.upper_clipped

    @property
    def log_relative_likelihood_cutoff(self) -> float:
        """Explicit alias for the cutoff on the log-relative-likelihood scale."""

        return self.support_cutoff


@dataclass(frozen=True)
class SelectionRuleSpec:
    """Selected-claim rule represented as intervals on the future Wald Z scale."""

    key: SelectionRule
    label: str
    alpha: float
    claim_direction: ClaimDirection
    threshold_working: float | None
    threshold_delta: float | None
    intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CriticalEffectResult:
    """Exact detectability threshold for one selected-claim rule and direction."""

    selection_rule: SelectionRule
    claim_direction: ClaimDirection
    alpha: float
    target_probability: float
    null_working: float
    standard_error: float
    critical_delta: float
    critical_effect_working: float
    achieved_probability: float


@dataclass(frozen=True)
class DesignMetric:
    """Repeated-study design metric for one assumed true effect."""

    true_effect_working: float
    delta: float
    selected_claim_probability: float
    type_s: float | None
    type_m: float | None
    expected_selected_abs_z: float | None
    observed_exaggeration: float | None

    @property
    def power(self) -> float:
        """Legacy read-only alias for selected-claim probability."""

        return self.selected_claim_probability


@dataclass(frozen=True)
class PrecisionTargetResult:
    """Required precision for one requested design target."""

    target: str
    requested_value: float
    required_se: float | None
    required_information_multiplier: float | None
    approx_95_ci_width_working: float | None
    achieved_power: float | None
    achieved_type_s: float | None
    achieved_type_m: float | None
    note: str
