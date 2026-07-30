# Scientific Scope

## Question supported

Given a reported estimate and confidence interval that are treated as one-parameter normal/Wald
quantities, or a repeated-study scenario with an explicitly specified Wald selection rule, what
reconstruction, compatibility, normalized relative-support, detectability, Type S/M, and precision
quantities follow from those assumptions?

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
log_support_ratio(A, B) =
    log_relative_likelihood(A) - log_relative_likelihood(B)
```

The relative likelihood is normalized to one at the reconstructed estimate. It is a Wald
approximation, not the exact profile likelihood from the fitted model.

A positive `log_support_ratio(A, B)` means A is more supported than B; reversing A and B reverses
the sign. Exponentiating gives `L(A) / L(B)` when that value is representable. This ordering is part
of the result and must not be collapsed into an unlabeled “likelihood ratio.”

For a finite MLE-to-bound support criterion `R > 1`, the included effects satisfy:

```text
log_relative_likelihood(theta) >= -log(R)
abs(z(theta)) <= sqrt(2 * log(R))
endpoint = theta_hat +/- SE * sqrt(2 * log(R))
```

The evidential S−2 support interval is the special case `R = exp(2)`, with working-scale endpoints
`\(\hat{\theta} \pm 2SE\)`. A 2:1 support interval is not S−2. The S−2 terminology and support
interpretation follow the Zampieri et al. methodology source recorded in
[migration provenance](MIGRATION_PROVENANCE.md#methodology-references-carried-forward).

Pairwise ratios and support intervals are algebraic consequences of the same
log-relative-likelihood function; they are not separately fitted likelihoods. Compatibility values
instead map the same absolute Wald distance to a two-sided tail area. Neither quantity is a
posterior probability, and a relative-support ratio is not a Bayes factor.

## Exact detectability and the legacy benchmark

For an assumed true working-scale effect \(\theta_{\mathrm{true}}\), null
\(\theta_{\mathrm{null}}\), and fixed Wald standard error \(SE > 0\):

```text
delta = (theta_true - theta_null) / SE
Z ~ Normal(delta, 1)
```

For `c = Normal.isf(alpha / 2)`, two-sided p-value selection has exact probability:

```text
P(selected) = P(Z < -c) + P(Z > c)
```

For `c = Normal.isf(alpha)`, the one-sided positive and negative probabilities are respectively:

```text
P(selected positive) = P(Z > c)
P(selected negative) = P(Z < -c)
```

Forward selected-claim probability also supports the other three canonical interval rules below by
evaluating their shared future-Z intervals. Critical-effect inversion is narrower: it returns the
smallest working-scale effect in a requested positive or negative direction whose exact probability
under one of the three p-value rules is at least the target. The two-sided rule is symmetric in
standardized distance; its positive and negative calls therefore produce paired working-scale
distances around the null.

This critical effect is a repeated-study detectability threshold under the stated fixed-SE Wald
model. It is not the observed estimate, a confidence bound, evidence about a realized result, a
user-defined scientifically meaningful effect, or a clinically validated minimum important
difference. Computing a probability at a post hoc observed estimate can be optimistic or circular
and does not convert that probability into observed evidence. Study-specific power or sample-size
planning may differ because it can require degrees-of-freedom corrections, non-normal reference
distributions, nuisance parameters, covariance, clustering, attrition, event rates, or a
design-specific relationship between information and sample size.

For ratio measures, effects and standard errors enter detectability calculations on the log working
scale. Paired natural-scale values are multiplicatively symmetric around the null ratio, not
arithmetically symmetric.

The retained legacy calculation:

```text
distance = (z_(1 - alpha/2) + z_power) * SE
```

is frozen at `alpha=0.05` and nominal probability `0.80`. It is a closed-form legacy benchmark, not
the exact solution of the two-tailed probability equation. The Perugini et al. source recorded in
[migration provenance](MIGRATION_PROVENANCE.md#methodology-references-carried-forward) supports the
critical-effect-size design rationale; the transparent normal/Wald definitions above govern the
implemented quantity.

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
- Critical-effect inversion is also a repeated-study design calculation and must not be interpreted
  as evidence about the observed estimate.
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
