# Wald Inference Core

`wald-inference` is a pure-Python package for deriving documented Wald reconstruction,
compatibility, normalized relative-support, detectability, selected-claim, Type S/M, and precision
quantities from reported estimates and confidence intervals or explicitly specified repeated-study
scenarios.

The distribution is `wald-inference`; the import package is `wald_inference`. Version `0.1.0` was
the behavior-preserving extraction from the immutable
[`conf_curve_likelihood` pre-split baseline](https://github.com/reblocke/conf_curve_likelihood/releases/tag/pre-split-baseline-2026-07-29).
Version `0.1.1` preserves that numerical behavior and documents stable adapter-only imports.
Version `0.2.0` adds generic pairwise support ratios and MLE-to-bound support intervals without
changing the preserved calculations. Version `0.2.1` makes those interval APIs fail closed when a
finite floating-point endpoint cannot accurately represent its requested support boundary. Version
`0.3.0` adds selected-claim probability curves and certified directed critical-effect inversion
while retaining the legacy z-sum benchmark separately. Version `0.4.0` adds typed joint
precision-guardrail and assumed-effect sensitivity results while preserving every existing
per-target calculation. Version `0.4.1` repairs threshold-transition bracketing in inverse
precision and makes structured pairwise comparisons reuse the canonical exact-binary64 support
ratio. It also fails closed on natural-ratio underflow. Version `0.4.2` hardens dependency
governance, exact annotated-tag binding, and immutable GitHub release verification without changing
numerical behavior.

## Why this package exists

The Wald-inference applet portfolio needs one released, reviewable implementation of its shared
numerical definitions. Keeping reconstruction, compatibility, relative support, detectability,
Type S/M, and precision primitives here prevents each browser app from acquiring a subtly
different formula or edge-case convention. Focused applications consume an exact Core release and
remain responsible for their own inputs, displays, explanations, exports, and privacy behavior.

## Intended use and audience

This package is for researchers, methodologists, educators, and scientific-software developers who
want to reproduce or build narrowly scoped tools around the documented one-parameter Wald model.
It accepts published aggregate estimates and intervals or explicit hypothetical repeated-study
assumptions; it does not need or accept patient identifiers. Users remain responsible for deciding
whether the effect scale, Wald approximation, confidence level, selection rule, and assumed true
effects are appropriate for their question.

## Question supported

Given a reported estimate and 95% confidence interval treated as a one-parameter normal/Wald result,
or a repeated-study scenario with a specified Wald selection rule, what compatibility, normalized
relative-support, detectability, Type S/M, and precision quantities follow from those assumptions?

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
- computes exact selected-claim probabilities and power curves for all six canonical selection
  rules;
- solves exact directed critical effects for two-sided, one-sided positive, and one-sided negative
  p-value rules;
- preserves the legacy z-sum critical-effect benchmark as a distinct closed-form benchmark;
- represents six selected-claim rules and computes selected-claim probability, Type S, Type M, and
  observed exaggeration; and
- evaluates information scaling, inverse precision targets, joint guardrails, and assumed-effect
  sensitivity.

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

GitHub Releases, not PyPI, are the authorized distribution channel. After downloading the v0.4.2
wheel:

```bash
python -m pip install ./wald_inference-0.4.2-py3-none-any.whl
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

## Minimal detectability example

Continuing from the reconstruction above:

```python
from wald_inference import (
    critical_effect_for_target_probability,
    power_curve,
)

critical = critical_effect_for_target_probability(
    null_working=reconstruction.null_working,
    standard_error=reconstruction.standard_error,
    alpha=0.05,
    target_probability=0.80,
    selection_rule="two_sided_p_lt_alpha",
    claim_direction="positive",
)
probabilities = power_curve(
    [0.0, 0.1, 0.3],
    null_working=reconstruction.null_working,
    standard_error=reconstruction.standard_error,
    alpha=0.05,
)
```

`critical` is the smallest positive working-scale effect whose conservatively rounded binary64
evaluation of the exact selected-claim probability meets the target under the stated normal/Wald
model. For the symmetric two-sided rule, call the solver with `claim_direction="negative"` to
obtain the paired lower value. This mathematical detectability threshold is not a confidence
bound, observed estimate, user-defined meaningful effect, clinically validated MCID, or
study-specific sample-size result. The preserved `legacy_critical_effect_distance` is a nearby
closed-form benchmark, not the exact two-tailed solution.

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
from wald_inference import joint_precision_result, precision_sensitivity

joint = joint_precision_result(
    0.2,
    null_working=reconstruction.null_working,
    current_se=reconstruction.standard_error,
    target_power=0.80,
    max_type_s=0.01,
    max_type_m=1.25,
)
sensitivity = precision_sensitivity(
    [0.1, 0.2, 0.3],
    null_working=reconstruction.null_working,
    current_se=reconstruction.standard_error,
    target_power=0.80,
    max_type_s=0.01,
    max_type_m=1.25,
)
```

`joint.target_results` preserves one immutable `PrecisionTargetResult` per mandatory guardrail.
Each row exposes feasibility, required standard error, relative information multiplier, approximate
95% working-scale interval width, achieved metrics, and its solver note. The joint result is
feasible only when every requested target is feasible; it uses the smallest required SE, reports
all constraints tied within the documented relative multiplier tolerance, and reports multiplier
`1.0` exactly when current precision satisfies every target. Sensitivity results retain input
effect order and make no-solution gaps explicit.

The information multiplier is relative information under the Wald scaling
`SE_new = SE_current / sqrt(multiplier)`. It is not automatically a sample-size multiplier and this
package does not translate it into a study-specific sample size.

## Working scales

Ratio measures—odds ratio, risk ratio, hazard ratio, incidence-rate ratio, and ratio of means—require
positive inputs and use the log working scale with default null `1`. Mean difference, risk
difference, rate difference, and regression coefficient use the identity working scale with default
null `0`.

Detectability functions accept and return working-scale effects. For a ratio measure, transform
critical working values through `from_working_scale`; equal positive and negative log distances are
multiplicatively, not arithmetically, symmetric on the natural scale.

Type M for a ratio measure is therefore an exaggeration ratio of log distances from the null, not
direct inflation of the natural-scale ratio.

## Scientific basis and citation roles

The publications below motivate terminology or scientific context. They do not authorize the
software's exact formulas, numerical tolerances, edge-case behavior, or API. Those implementation
details are governed by the tagged Core source, documented definitions, validation suite, and
frozen parity evidence.

| Capability | Methodology source | Role in this package |
|---|---|---|
| Compatibility interpretation | Rafi Z, Greenland S. “Semantic and cognitive tools to aid statistical science: replace confidence and significance by compatibility and surprise.” *BMC Medical Research Methodology*. 2020;20:244. [doi:10.1186/s12874-020-01105-9](https://doi.org/10.1186/s12874-020-01105-9). | Supports compatibility terminology and interpretation of p-value functions; it is not a code or numerical-fixture dependency. |
| Normalized relative support and S−2 | Zampieri FG, Cahusac PMB, Maia IS, et al. “Trial Analysis and Interpretation in Critical Care Using the Evidential (Likelihood) Approach: Rationale and Practical Considerations.” *American Journal of Respiratory and Critical Care Medicine*. 2025;211(9):1610–1621. [doi:10.1164/rccm.202504-0809TR](https://doi.org/10.1164/rccm.202504-0809TR). | Supports evidential-likelihood, likelihood-ratio, support, and S−2 terminology; Core still computes a CI-reconstructed Wald approximation, not an exact fitted-model likelihood. |
| Critical-effect rationale | Perugini A, Gambarota F, Toffalini E, et al. “The Benefits of Reporting Critical-Effect-Size Values.” *Advances in Methods and Practices in Psychological Science*. 2025;8(2):25152459251335298. [doi:10.1177/25152459251335298](https://doi.org/10.1177/25152459251335298). | Supplies design context; Core's explicit probability model and directed inverse define the implemented quantity. |
| Type S and Type M | Gelman A, Carlin J. “Beyond Power Calculations: Assessing Type S (Sign) and Type M (Magnitude) Errors.” *Perspectives on Psychological Science*. 2014;9(6):641–651. [doi:10.1177/1745691614551642](https://doi.org/10.1177/1745691614551642). | Supports Type S/M concepts and repeated-study design interpretation; Core's documented selection rules govern the calculations. |
| Joint precision guardrails | The Type S/M and critical-effect sources above provide concept context; no external paper is claimed to define this joint inverse solver. | The typed solver, feasibility rules, tolerances, tests, and released Core behavior are the authority. Relative information is not automatically sample size. |

The Rafi and Greenland article was retrieved on 2026-08-01 and is available under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retrieval and provenance notes for all
four sources are recorded in [Migration provenance](docs/MIGRATION_PROVENANCE.md). No publication
figure, table, dataset, code, or substantial text is copied into this repository.

## Public API and scientific scope

- [API reference](docs/API.md)
- [Scientific scope and conditioning](docs/SCIENTIFIC_SCOPE.md)
- [Validation and baseline cases](docs/VALIDATION.md)
- [Migration provenance](docs/MIGRATION_PROVENANCE.md)
- [Decisions](docs/DECISIONS.md)
- [Maintenance and releases](docs/MAINTENANCE.md)
- [Governance](docs/GOVERNANCE.md)
- [Privacy](docs/PRIVACY.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)

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

- Version prepared for release: `0.4.2`
- Governance and maintenance contact: repository issues and pull requests using the documented
  synthetic-input boundary.
- Vulnerabilities: use the private process in [SECURITY.md](SECURITY.md), never a public report.
- Software citation: cite the exact tagged Core release or commit used; machine-readable metadata
  is in [`CITATION.cff`](CITATION.cff). Also cite the directly relevant methodology source above
  when discussing its terminology or scientific rationale.
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- License: MIT; see [`LICENSE`](LICENSE)
- Copyright: Copyright (c) 2026 Brian Locke
- Maintainer: Brian Locke (`@reblocke`)
