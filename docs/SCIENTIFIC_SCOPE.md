# Scientific Scope

## Question supported

Given a reported estimate and confidence interval that are treated as one-parameter normal/Wald
quantities, or a repeated-study scenario with an explicitly specified Wald selection rule, what
reconstruction, compatibility, normalized relative-support, Type S/M, and precision quantities
follow from those assumptions?

The package answers that narrow mathematical question. It does not establish that the assumptions
are appropriate for a particular study.

## Effect measures and working scales

The supported ratio measures are:

- odds ratio;
- risk ratio;
- hazard ratio;
- incidence-rate ratio; and
- ratio of means.

They require positive natural-scale inputs, use the log working scale, and default to null value
`1`.

The supported additive measures are:

- mean difference;
- risk difference;
- rate difference; and
- regression coefficient.

They use the identity working scale and default to null value `0`.

## Reconstruction from a confidence interval

The required lower and upper bounds are interpreted as a 95% Wald confidence interval unless the
public function explicitly accepts another confidence level. On the working scale, the reconstructed
estimate is the safe midpoint of the bounds and the standard error is determined by their distance
relative to the appropriate standard-normal quantile. A supplied estimate validates the
reconstruction; it does not silently replace the interval midpoint.

The result records whether the estimate was reconstructed or validated, the transformed bounds and
null, the standard error and method, relative interval asymmetry, and warnings. Finite-range
protection is part of the contract: invalid or unrepresentable derived values raise
`ValidationError` rather than leaking `NaN` or infinity.

This reconstruction cannot recover the original model, likelihood, covariance structure, variance
estimator, degrees-of-freedom correction, or study design.

## Observed-data functions

For a candidate working-scale effect \(\theta\), reconstructed estimate \(\hat{\theta}\), and standard
error \(SE\):

```text
z(theta) = (theta - theta_hat) / SE
compatibility(theta) = 2 * Normal.sf(abs(z(theta)))
log_relative_likelihood(theta) = -0.5 * z(theta)^2
relative_likelihood(theta) = exp(log_relative_likelihood(theta))
```

The relative likelihood is normalized to one at the reconstructed estimate. It is a Wald
approximation, not the exact profile likelihood from the fitted model.

The evidential S−2 support interval uses working-scale endpoints
`\(\hat{\theta} \pm 2SE\)`. Pairwise support comparisons are algebraic consequences of the same
log-relative-likelihood function; they are not separately fitted likelihoods.

## Detectability benchmark

The legacy critical-effect calculation is a closed-form z-sum benchmark retained for baseline
compatibility. It must be described as the legacy benchmark, not as exact generalized power or a
study-specific sample-size result. Generalized detectability functions, when present, must state
their selection rule and assumptions explicitly.

## Selected claims and Type S/M

The supported selection-rule keys are:

```text
two_sided_p_lt_alpha
one_sided_positive_p_lt_alpha
one_sided_negative_p_lt_alpha
ci_excludes_null_in_beneficial_direction
estimate_exceeds_mcid_and_p_lt_alpha
ci_excludes_mcid
```

Each rule defines a repeated-study selection event under a one-parameter normal/Wald sampling model.
Directional and threshold-conditioned rules require an explicit direction and, where applicable, a
threshold on the working scale.

`selected_claim_probability` is the unconditional probability that the configured selection event
occurs. A legacy `power` alias may delegate to it, but must not create a second implementation.

Type S is the conditional probability of a selected estimate having the wrong sign relative to the
true working-scale distance from the null. Type M is the expected selected absolute working-scale
distance divided by the absolute true working-scale distance. Ratio-measure Type M therefore uses
log distance from the null, not direct inflation of the natural ratio.

Type S and Type M are repeated-study operating characteristics conditional on the assumed true effect
and selection rule. They are not posterior probabilities that an observed estimate is wrong.
Quantities with an undefined direction or denominator at or near the null are `None`, not zero.
Observed exaggeration is a separate deterministic comparison and is also undefined at or near the
null.

## Precision calculations

For information multiplier \(m > 0\):

```text
SE_design = SE_current / sqrt(m)
```

Inverse precision calculations solve for a required standard error that meets a requested
selected-claim-probability, maximum Type S, or maximum Type M target at a specified true effect and
selection rule. A precision result reports required standard error, information multiplier,
approximate 95% working-scale confidence-interval width, achieved metrics, and an explanatory note.

Infeasible or unrepresentable targets return the documented no-solution result or raise
`ValidationError` according to whether the input is valid but unattainable or invalid. The package
does not translate an information multiplier into sample size for a particular study design.

## Conditioning and interpretation

- Observed compatibility and support condition on the reported interval under the reconstructed Wald
  model.
- Selected-claim, Type S/M, and precision quantities are repeated-study design calculations
  conditioned on user-specified true effects and rules.
- These two layers may share a standard-error reconstruction but answer different questions.
- User-specified thresholds are mathematical inputs; the package does not determine whether a
  threshold is clinically or scientifically meaningful.

## Non-goals

The package does not provide:

- exact fitted-model or profile likelihood;
- arbitrary non-Wald interval inversion;
- Bayesian inference or posterior probabilities;
- multivariable covariance reconstruction;
- design-specific sample-size formulas;
- clinical decision support or medical-device functionality;
- plotting, browser payloads, UI text, CSV/PNG export, persistence, or telemetry; or
- guarantees that a published interval satisfies the Wald assumptions.
