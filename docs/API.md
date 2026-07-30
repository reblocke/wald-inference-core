# Public API

The distribution is `wald-inference`; import it as `wald_inference`. The names in
`wald_inference.__all__` are the deliberate stable root surface. Functions validate finite inputs
and raise `ValidationError` for invalid or unrepresentable values.

## Effects

```python
EffectSpec
EFFECT_SPECS
get_effect_spec(effect_type)
to_working_scale(effect_type, values)
from_working_scale(effect_type, values)
```

`EFFECT_SPECS` is the frozen effect registry. Transformations return a Python `float` for a Python
scalar input and a NumPy array for sequence or positive-dimensional array input. A
zero-dimensional array retains the frozen NumPy-operation result: ratio transformations return a
NumPy scalar, while identity transformations return a zero-dimensional array. Ratio
transformations reject nonpositive natural-scale values.

Defaults exported for adapters:

```python
DEFAULT_EFFECT_TYPE
```

## Reconstruction

```python
WaldReconstruction
StandardErrorEstimate
reconstruct_wald_from_95_ci(
    effect_type="odds_ratio",
    estimate=None,
    lower=None,
    upper=None,
    null_value=None,
)
estimate_se(theta_hat, lower, upper)
estimate_se_details(theta_hat, lower, upper)
```

`reconstruct_wald_from_95_ci` is the canonical constructor. `lower` and `upper` are required.
`estimate`, when supplied, validates the interval midpoint within the frozen tolerance; the
reconstruction remains CI-derived. The immutable result includes display and working values,
estimate source, default-null status, standard-error details, relative asymmetry, and warnings.

## Compatibility and point summaries

```python
standardized_distance(theta, theta_hat, se)
compatibility_curve(theta, theta_hat, se)
WaldPointSummary
wald_point_summary(theta_hat, se, candidate_working)
```

Curve functions accept a scalar, sequence, or NumPy array. A scalar or zero-dimensional array input
returns a NumPy scalar; a sequence or positive-dimensional array input returns a NumPy array.
Compatibility is the two-sided normal survival probability for the absolute standardized distance.

## Relative support

```python
relative_likelihood(theta, theta_hat, se)
log_relative_likelihood(theta, theta_hat, se)
log_support_ratio(candidate_a_working, candidate_b_working, *, theta_hat, se)
support_ratio(candidate_a_working, candidate_b_working, *, theta_hat, se)
SupportInterval
support_interval(theta_hat, se, *, log_relative_likelihood_cutoff=-2.0)
support_interval_for_ratio(theta_hat, se, *, mle_to_bound_ratio)
SupportComparison
support_comparison(candidate_working, reference_working, *, theta_hat, se)
```

Relative likelihood is normalized to one at `theta_hat`. `support_interval` defaults to the
evidential S−2 cutoff and returns working-scale endpoints plus finite-range clipping flags.
`support_comparison` reports candidate support relative to both the reconstructed estimate and a
reference value, with log-domain values available when exponentiated ratios overflow.

`log_support_ratio(A, B, ...)` returns `log L(A) - log L(B)`, so a positive result means A is more
supported than B under the normalized Wald reconstruction. Its candidate inputs accept scalars,
sequences, or NumPy arrays and follow NumPy broadcasting rules. `support_ratio` is the scalar
exponentiated form: it returns `None` when the finite log ratio is larger than the maximum
representable exponent and may return `0.0` on floating-point underflow. Use `log_support_ratio`
whenever preserving magnitude or direction at numerical extremes matters.

`support_interval_for_ratio(..., mle_to_bound_ratio=R)` requires a finite `R > 1` and delegates to
the canonical log-cutoff interval with cutoff `-log(R)`. It contains values for which the
reconstructed estimate is no more than R times as supported under the normalized Wald likelihood.
S−2 is exactly the special case `R = exp(2)`; a 2:1 interval is not an S−2 interval.

These APIs compare parameter values under a one-parameter Wald approximation. They do not recover
the fitted model's exact profile likelihood and do not produce posterior probabilities or Bayes
factors.

## Exact detectability

```python
selected_claim_probability(
    true_effect_working,
    *,
    null_working,
    standard_error,
    alpha=0.05,
    selection_rule="two_sided_p_lt_alpha",
    claim_direction="positive",
    threshold_working=None,
)
power_curve(
    true_effects_working,
    *,
    null_working,
    standard_error,
    alpha=0.05,
    selection_rule="two_sided_p_lt_alpha",
    claim_direction="positive",
    threshold_working=None,
)
CriticalEffectResult
critical_effect_for_target_probability(
    *,
    null_working,
    standard_error,
    alpha=0.05,
    target_probability=0.80,
    selection_rule="two_sided_p_lt_alpha",
    claim_direction="positive",
)
```

`selected_claim_probability` evaluates the canonical selection intervals under a future
standard-normal Wald statistic centered at
`delta = (true_effect_working - null_working) / standard_error`. It supports all six selection rules
documented below. Scalar input returns a Python `float`; array-like input returns a NumPy array with
the same shape. `power_curve` is the corresponding nonempty one-dimensional convenience.

`critical_effect_for_target_probability` returns the smallest effect in the requested direction
whose exact selected-claim probability meets the finite target strictly between zero and one. It
uses explicit finite bracketing and monotonic bisection. Inversion is intentionally limited to:

```text
two_sided_p_lt_alpha             positive or negative branch
one_sided_positive_p_lt_alpha    positive branch
one_sided_negative_p_lt_alpha    negative branch
```

The immutable `CriticalEffectResult` records the rule, direction, alpha, target, null, standard
error, signed standardized `critical_delta`, working-scale critical effect, and achieved
probability. Unsupported inverse rules, incoherent one-sided directions, a target without a finite
bracket, or an unrepresentable working-scale result raise `ValidationError`.

For ratio measures, these functions operate on the log working scale. Use the effect registry
transformations to map returned critical effects to the natural scale; equal log distances are
multiplicatively rather than arithmetically symmetric.

## Legacy detectability benchmark

```python
legacy_critical_effect_distance(se)
legacy_critical_effect_markers(null_value, se)
```

These functions expose the frozen z-sum benchmark for `alpha=0.05` and nominal power `0.80`. The
legacy label is intentional: this is not exact generalized power or a study-specific sample-size
calculation.

## Grids

```python
build_grid(
    theta_hat,
    se,
    span_multiplier=4.5,
    n=801,
    include_values=None,
    max_span=None,
)
max_safe_grid_span(theta_hat, se, *, natural_axis_upper_bound=None)
```

These helpers build finite working-scale arrays for downstream consumers. They do not own display
axis transformations, UI ranges, or plotting.

## Selection rules

```python
SelectionRuleSpec
selection_rule_spec(
    *,
    selection_rule="two_sided_p_lt_alpha",
    alpha=0.05,
    null_working=0.0,
    se=1.0,
    claim_direction="positive",
    threshold_working=None,
)
```

The returned immutable object describes selected intervals on the future Wald-Z scale. Its interval
endpoints use positive or negative infinity as mathematical open-tail sentinels; the object is not a
strict-JSON payload. Exported adapter defaults:

```python
DEFAULT_SELECTION_RULE
DEFAULT_CLAIM_DIRECTION
```

Threshold-conditioned rules require a threshold on the appropriate side of the null.

## Type S/M

```python
DesignMetric
design_metrics_for_true_effects(
    true_effects_working,
    *,
    null_working,
    se,
    estimate_working=None,
    alpha=0.05,
    selection_rule="two_sided_p_lt_alpha",
    claim_direction="positive",
    threshold_working=None,
    near_null_delta=1e-12,
)
```

The result is a list of immutable `DesignMetric` values in input order.
`selected_claim_probability` is canonical; `power` is a read-only compatibility property that
delegates to the same value. Type S/M and observed exaggeration use `None` at or near the null.

The exported default is:

```python
DEFAULT_NEAR_NULL_DELTA
```

## Precision

```python
information_scaled_standard_error(standard_error, information_multiplier)
approximate_wald_ci_width(standard_error, z975=1.959963984540054)
PrecisionTargetResult
precision_target_results(
    true_effect_working,
    *,
    null_working,
    current_se,
    alpha=0.05,
    target_power=None,
    max_type_s=None,
    max_type_m=None,
    selection_rule="two_sided_p_lt_alpha",
    claim_direction="positive",
    threshold_working=None,
    near_null_delta=1e-12,
    z975=1.959963984540054,
)
solve_required_precision(...)
```

`precision_target_results` returns one immutable row per requested target in stable order.
`solve_required_precision` returns the strictest aggregate solution and returns all-`None` fields if
any requested target lacks a finite solution.

The following preserved two-sided delta solvers are also public:

```python
solve_required_delta_for_power(alpha, target_power)
solve_required_delta_for_type_s(alpha, max_type_s)
solve_required_delta_for_type_m(alpha, max_type_m)
```

## Errors and version

```python
ValidationError
__version__
```

`__version__` is `0.3.0`. Canonical numerical outputs intended for serialization contain finite
values or documented `None`; invalid inputs do not return sentinel NaN or infinity. The documented
`SelectionRuleSpec.intervals` infinities are structural open-tail boundaries, not calculated result
values.

## Compatibility aliases

Module-level compatibility imports needed during workbench migration delegate to, or directly
re-export, canonical definitions. They are not duplicated implementations and are intentionally
omitted from the root `__all__` when a clearer canonical name exists. General consumers should use
the names documented above.

The narrow adapter-only surface is imported explicitly:

```python
from wald_inference import legacy
```

`wald_inference.legacy` exposes the frozen direct-call behavior of `to_working_scale`,
`from_working_scale`, `estimate_se`, `build_grid`, `confidence_curve`, `relative_likelihood`,
`log_relative_likelihood`, and `summaries`. This includes historical raw coercion exceptions,
NumPy overflow warnings, and nonfinite outputs for invalid direct calls. It exists only to rewire the
integrated workbench without changing behavior and is excluded from the root `__all__`.

Version 0.1.1 also makes the workbench adapter's numerical configuration imports stable:

```python
from wald_inference.legacy import (
    ASYMMETRY_RELATIVE_TOLERANCE,
    DEFAULT_GRID_POINTS,
    DEFAULT_SOLVER_TOLERANCE,
    DEFAULT_SPAN_MULTIPLIER,
    ESTIMATE_MATCH_ABSOLUTE_TOLERANCE,
    ESTIMATE_MATCH_RELATIVE_TOLERANCE,
    GRID_EXPANSION_PADDING_MULTIPLIER,
    LOG_MAX_FLOAT,
    MAX_FINITE_ABS_Z,
    MAX_FINITE_SPAN,
    MAX_FLOAT,
    MAX_INFORMATION_MULTIPLIER,
    Z80,
    Z975,
    asymmetry_warning,
)
```

These are direct re-exports of the values and helper used by the canonical implementation, not
copied constants or formulas. They are public for compatibility adapters; new numerical code should
use the strict canonical functions rather than build behavior from these implementation settings.
