# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## [0.3.0] - 2026-07-30

### Added

- Added scalar/array `selected_claim_probability` and one-dimensional `power_curve`, both routed
  through the canonical intervals for all six existing selection rules.
- Added `critical_effect_for_target_probability` with stable near-null probability increments,
  one-sided quantile-domain inversion, and complement-domain one- and two-sided inversion near
  probability one.
- Added immutable `CriticalEffectResult` with signed standardized delta, working-scale critical
  effect, target, and achieved probability.

### Validation

- Added unit, property, and independent scientific-reference tests for six-rule parity, scalar and
  vector behavior, null probability, monotonicity, symmetry, analytic one-sided quantiles, direct
  two-sided tail evaluation, target inversion, log-scale composition, invalid inputs, and
  floating-point extremes.
- Added high-precision boundary and high-alpha references, explicit predecessor-minimality checks,
  targets one floating-point step above alpha and below one, and bit-exact null and near-null
  monotonicity contracts.
- Extended the exact root API, release metadata, distribution inspection, and cold-wheel smoke
  contracts for v0.3.0.
- Retained the frozen baseline-parity gate for every pre-existing numerical output, including the
  unchanged legacy z-sum distance and markers.

### Scientific impact

- The exact critical effect is defined as the smallest directed working-scale effect meeting the
  requested selected-claim probability under a fixed-SE one-parameter normal/Wald model.
- The legacy z-sum calculation remains a separately labeled closed-form benchmark. Neither quantity
  is a confidence bound, observed estimate, clinically validated meaningful effect, or
  study-specific sample-size calculation.

## [0.2.0] - 2026-07-29

### Added

- Added a vectorized, log-domain-first `log_support_ratio` API with explicit numerator/denominator
  ordering and broadcasting validation.
- Added scalar `support_ratio`, which returns `None` rather than infinity when exponentiating a
  finite log ratio would overflow.
- Added `support_interval_for_ratio` for finite MLE-to-bound criteria greater than one, including
  the existing S−2 criterion as the special case `R = exp(2)`.

### Validation

- Added unit, property, and independent closed-form tests for pairwise antisymmetry, identity,
  broadcasting, overflow, analytic interval endpoints, interval-width ordering, invalid criteria,
  and exact legacy S−2 parity.
- Extended the exact root API contract and cold-wheel smoke test while retaining the unchanged
  `wald_inference.legacy` contract.
- Retained the frozen baseline-parity gate for every pre-existing numerical output and added
  v0.2.0 expectations to the deterministic archive coverage.

### Scientific impact

- The new APIs expose algebraic consequences of the existing normalized one-parameter Wald
  likelihood reconstruction. They do not recover an exact fitted-model likelihood, introduce a
  posterior interpretation, or change any pre-existing formula, default, tolerance, or result.

## [0.1.1] - 2026-07-29

### Added

- Documented stable imports under `wald_inference.legacy` for the numerical bounds, grid defaults,
  reconstruction tolerances and warning helper, detectability quantiles, and solver limits needed
  by the backward-compatible integrated workbench adapter.

### Validation

- Added an exact `wald_inference.legacy.__all__` contract, identity checks proving the adapter
  imports are direct re-exports of the canonical definitions, and cold-wheel coverage of the
  compatibility surface.
- Retained the exact root `wald_inference.__all__` contract and frozen baseline-parity gate.

### Scientific impact

- None. This patch adds import stability and release metadata only; it does not change a formula,
  constant value, validation rule, selection boundary, tolerance, effect registry entry, or
  undefined-value convention.

## [0.1.0] - 2026-07-29

### Added

- Initial public `wald-inference` package extracted from the frozen
  `reblocke/conf_curve_likelihood` baseline.
- Typed public APIs for effect transformations, Wald reconstruction, compatibility, normalized
  relative support, detectability benchmarks, selection rules, Type S/M metrics, and inverse
  precision targets.
- Unit, property, scientific-reference, and frozen-baseline regression tests.
- Machine-readable and human-readable baseline-parity reporting.
- Reproducible wheel and source-distribution builds with cold-wheel smoke checks.

### Validation

- The release gate compares core-owned outputs with behavior source
  `830756ecb11b4e8161f8dfe1fc75afc346ef4467` using `rtol=1e-12` and `atol=1e-14`.
- The release workflow records observed artifact hashes in `SHA256SUMS` and publishes its
  machine-readable result as `baseline-parity.json`.
- CI run URLs, final artifact hashes, and test counts are release evidence and are not claimed
  before the corresponding workflow completes.

### Scientific impact

- This is a behavior-preserving extraction. No formula, selection tail, numerical tolerance,
  effect registry entry, or undefined-value convention is intentionally changed from the frozen
  baseline.

### Known limitations

- The package implements documented Wald approximations, not exact fitted-model likelihood,
  arbitrary non-Wald intervals, Bayesian inference, or design-specific sample-size calculations.

[Unreleased]: https://github.com/reblocke/wald-inference-core/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/reblocke/wald-inference-core/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/reblocke/wald-inference-core/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/reblocke/wald-inference-core/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/reblocke/wald-inference-core/releases/tag/v0.1.0
