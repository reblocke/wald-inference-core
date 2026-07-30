# Wald Inference Core

`wald-inference` is a pure-Python package for deriving documented Wald reconstruction,
compatibility, normalized relative-support, selected-claim, Type S/M, and precision quantities from
reported estimates and confidence intervals or explicitly specified repeated-study scenarios.

The distribution is `wald-inference`; the import package is `wald_inference`. Version `0.1.0` was
the behavior-preserving extraction from the immutable
[`conf_curve_likelihood` pre-split baseline](https://github.com/reblocke/conf_curve_likelihood/releases/tag/pre-split-baseline-2026-07-29).
Version `0.1.1` preserves that numerical behavior and documents stable adapter-only imports.
Version `0.2.0` adds generic pairwise support ratios and MLE-to-bound support intervals without
changing the preserved calculations. Version `0.2.1` makes those interval APIs fail closed when a
finite floating-point endpoint cannot accurately represent its requested support boundary.

## Question supported

Given a reported estimate and 95% confidence interval treated as a one-parameter normal/Wald result,
or a repeated-study scenario with a specified Wald selection rule, what compatibility, normalized
relative-support, Type S/M, and precision quantities follow from those assumptions?

The package answers that mathematical question. It does not establish that the assumptions fit a
particular study.

## What the package does

- registers five ratio and four additive effect measures with explicit working scales;
- reconstructs an estimate and standard error from a reported 95% confidence interval;
- validates an optional reported estimate against the interval midpoint;
- computes standardized distances, two-sided compatibility, and normalized Wald relative
  likelihood;
- computes S−2, generic log-support, and MLE-to-bound-ratio intervals plus pairwise support
  comparisons;
- preserves the legacy z-sum critical-effect benchmark;
- represents six selected-claim rules and computes selected-claim probability, Type S, Type M, and
  observed exaggeration; and
- evaluates information scaling and inverse precision targets.

Canonical calculation results intended for serialization contain finite numbers or intentional
`None`. `SelectionRuleSpec.intervals` uses positive and negative infinity only as mathematical
open-tail sentinels; invalid or otherwise unrepresentable canonical inputs raise `ValidationError`.

## What it does not do

- recover an exact fitted-model or profile likelihood;
- infer the original model, variance estimator, covariance structure, study design, or sample-size
  model;
- invert arbitrary non-Wald intervals;
- provide Bayesian posterior probabilities;
- determine whether a threshold is clinically meaningful;
- provide clinical decision support or medical-device functionality; or
- provide plotting, browser payloads, UI wording, exports, persistence, telemetry, or hosted
  computation.

Type S/M quantities are repeated-study operating characteristics under an assumed true effect and
selection rule. They are not posterior probabilities that an observed estimate is wrong.

## Installation

For development from a clone:

```bash
uv sync --locked --all-groups
```

GitHub Releases, not PyPI, are the authorized distribution channel. After downloading the v0.2.1
wheel:

```bash
python -m pip install ./wald_inference-0.2.1-py3-none-any.whl
```

The release page is <https://github.com/reblocke/wald-inference-core/releases>. A downstream
application should pin the exact release wheel and its `uv.lock` resolution, never an unreviewed
branch.

## Minimal observed example

```python
import numpy as np

from wald_inference import (
    compatibility_curve,
    reconstruct_wald_from_95_ci,
    relative_likelihood,
    support_interval_for_ratio,
)

reconstruction = reconstruct_wald_from_95_ci(
    "mean_difference",
    lower=0.11,
    upper=0.73,
)
points = np.array(
    [
        reconstruction.lower_working,
        reconstruction.estimate_working,
        reconstruction.upper_working,
    ]
)
compatibility = compatibility_curve(
    points,
    reconstruction.estimate_working,
    reconstruction.standard_error,
)
support = relative_likelihood(
    points,
    reconstruction.estimate_working,
    reconstruction.standard_error,
)
four_to_one_interval = support_interval_for_ratio(
    reconstruction.estimate_working,
    reconstruction.standard_error,
    mle_to_bound_ratio=4.0,
)
```

`compatibility[1]` and `support[1]` are one at the reconstructed estimate. The endpoint
compatibilities are approximately `0.05` under the 95% Wald reconstruction.
`four_to_one_interval` contains working-scale values for which the CI-implied estimate is no more
than four times as supported under the normalized Wald reconstruction. A finite interval is
returned only when each non-clipped endpoint independently reproduces the requested log-support
boundary within the documented numerical tolerance; otherwise the call raises `ValidationError`
instead of labeling a materially different representable float as that boundary.

## Minimal Type S/M example

Continuing from the reconstruction above:

```python
from wald_inference import design_metrics_for_true_effects

metrics = design_metrics_for_true_effects(
    [0.0, 0.1, 0.3],
    null_working=reconstruction.null_working,
    se=reconstruction.standard_error,
    estimate_working=reconstruction.estimate_working,
    alpha=0.05,
    selection_rule="two_sided_p_lt_alpha",
)
```

Each immutable `DesignMetric` reports `selected_claim_probability`, `type_s`, `type_m`,
`expected_selected_abs_z`, and `observed_exaggeration`. Near-null Type S/M and observed
exaggeration are `None`.

## Minimal precision example

Continuing from the same reconstruction:

```python
from wald_inference import precision_target_results

targets = precision_target_results(
    0.2,
    null_working=reconstruction.null_working,
    current_se=reconstruction.standard_error,
    target_power=0.80,
    max_type_s=0.01,
    max_type_m=1.25,
)
```

Each `PrecisionTargetResult` records required standard error, relative information multiplier,
approximate 95% working-scale interval width, achieved metrics, and a note. The calculation does not
translate relative information into a study-specific sample size.

## Working scales

Ratio measures—odds ratio, risk ratio, hazard ratio, incidence-rate ratio, and ratio of means—require
positive inputs and use the log working scale with default null `1`. Mean difference, risk
difference, rate difference, and regression coefficient use the identity working scale with default
null `0`.

Type M for a ratio measure is therefore an exaggeration ratio of log distances from the null, not
direct inflation of the natural-scale ratio.

## Public API and scientific scope

- [API reference](docs/API.md)
- [Scientific scope and conditioning](docs/SCIENTIFIC_SCOPE.md)
- [Validation and baseline cases](docs/VALIDATION.md)
- [Migration provenance](docs/MIGRATION_PROVENANCE.md)
- [Decisions](docs/DECISIONS.md)
- [Maintenance and releases](docs/MAINTENANCE.md)
- [Privacy](docs/PRIVACY.md)

The root `wald_inference.__all__` is the deliberate stable root import surface. Downstream
applications adapt domain objects into app contracts; they do not copy formulas.

The documented `wald_inference.legacy` surface provides direct compatibility re-exports for the
integrated workbench adapter without expanding the root API. General consumers should use the
canonical root imports.

## Verification

```bash
uv sync --locked --all-groups
make verify
git status --short
```

The full gate checks formatting, lint, metadata, unit/property/regression tests, frozen numerical
parity, wheel/sdist contents, and a cold-wheel installation.

## Portfolio relationship

This repository is the numerical source of truth for the Wald-inference applet portfolio. It has no
hosted application or GitHub Pages deployment. Focused applications and the integrated workbench
consume exact released core versions and own presentation, accessibility, export, and browser
privacy behavior.

## Version, citation, license, and contact

- Version prepared for release: `0.2.1`
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- License: MIT; see [`LICENSE`](LICENSE)
- Copyright: Copyright (c) 2026 Brian Locke
- Maintainer: Brian Locke (`@reblocke`)
