# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## [0.4.1] - 2026-07-30

### Fixed

- Evaluate the exact active-cutoff transition for
  `estimate_exceeds_mcid_and_p_lt_alpha` before inverse-precision bracketing. This prevents a
  halving step from skipping a finite feasible power band when the assumed true effect is in the
  selected direction but does not exceed the claim threshold.
- Route the candidate-to-reference field in `support_comparison` through the canonical
  exact-binary64 `log_support_ratio` implementation. Extreme finite candidates no longer lose a
  representable pairwise log ratio through subtraction of separately rounded log likelihoods.
- Reject a strict public log-ratio back-transform when exponential underflow would return natural
  zero, which is outside the ratio effect registry's strictly positive domain. The frozen legacy
  adapter retains its historical underflow behavior.

### Validation

- Added positive and negative threshold-transition regressions that certify the largest feasible
  standard error, plus an unattainable-target control.
- Added extreme symmetric and adjacent-candidate regressions proving that `support_comparison` and
  `log_support_ratio` share one pairwise numerical authority.
- Added scalar/array ratio-underflow, cold-wheel, and legacy-boundary regressions.
- Retained the existing ordinary-path precision values and notes, candidate-versus-MLE comparison
  fields, root API, legacy adapter, and frozen baseline-parity contract.

### Scientific impact

- Forward selected-claim probabilities, selection boundaries, and Type S/M definitions are
  unchanged. The precision repair changes only previously false no-solution results in the
  affected threshold-conditioned power cases.
- The pairwise repair changes only finite extreme-value comparisons affected by catastrophic
  cancellation. It does not recover a fitted-model likelihood or change the normalized Wald
  interpretation.
- The transform repair replaces an out-of-domain natural zero with `ValidationError`. The stricter
  validation does not change any representable natural-scale result.

## [0.4.0] - 2026-07-30

### Added

- Added immutable root-public `JointPrecisionResult` and `joint_precision_result`, which summarize
  mandatory selected-claim-probability, Type S, and Type M guardrails through the preserved
  per-target solvers.
- Added deterministic root-public `precision_sensitivity` for a nonempty one-dimensional sequence
  of finite assumed working-scale true effects.
- Added read-only per-target feasibility, current-sufficiency, and explicit
  `achieved_selected_claim_probability` properties without changing the frozen
  `PrecisionTargetResult` dataclass fields.

### Joint semantics

- The feasible joint result uses the smallest required SE and largest relative information
  multiplier, reports every binding target within default relative multiplier tolerance `1e-8`,
  and returns multiplier `1.0` exactly when current precision satisfies all targets.
- Any infeasible mandatory target makes the joint result infeasible while preserving every
  per-target row. Notes name the target(s) and selected-effect/rule assumptions and identify
  applicable near-null, threshold, or finite-bracketing conditions.
- Sensitivity retains input order and duplicates and represents infeasible effects as explicit
  no-solution gaps rather than interpolated values.

### Validation

- Added unit, property, and scientific-reference coverage for strictest-envelope and tie behavior,
  current sufficiency, mandatory infeasibility propagation, near-null and threshold behavior,
  finite strict JSON, deterministic scalar/sensitivity equivalence, target ordering, achieved
  forward metrics, information and CI-width identities, expected sensitivity monotonicity, and
  log-ratio/CI reconstruction composition.
- Extended the exact root API, release metadata, distribution inspection, and cold-wheel smoke
  contracts for v0.4.0.
- Retained zero-difference frozen parity across all 23,095 pre-existing values, including B06/B07
  precision rows, notes, undefined values, and legacy dictionary aggregates.

### Scientific impact

- The joint and sensitivity results are repeated-study Wald calculations conditioned on explicit
  assumed true effects, selection rules, thresholds, and guardrails. They are not evidence about an
  observed estimate or distributions over the true effect.
- Relative information follows `SE_new = SE_current / sqrt(multiplier)` and is not automatically a
  sample-size multiplier or a formal study-design calculation.

## [0.3.0] - 2026-07-30

### Added

- Added scalar/array `selected_claim_probability` and one-dimensional `power_curve`, both routed
  through the canonical intervals for all six existing selection rules.
- Added `critical_effect_for_target_probability` with one coherent, conservatively certified
  binary64 probability kernel for forward evaluation, inversion, and achieved probability.
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
- Added handoff-neighbor, high-alpha, high-probability, nonzero-null representability, and
  public-forward coherence regressions. A guarded critical value, direction-specific tail
  evaluation, and scale-aware probability guards keep roots on the conservative side of hard-coded
  high-precision references across extreme finite alpha values; ordered-binary64 bracketing finds
  the minimal representable working effect without a fixed ULP cap. Replaced adaptive scalar
  quadrature with fixed stable quadrature, reducing a 10,000-point near-null curve to well under one
  second in local validation.
- Extended the exact root API, release metadata, distribution inspection, and cold-wheel smoke
  contracts for v0.3.0.
- Retained the frozen baseline-parity gate for every pre-existing numerical output, including the
  unchanged legacy z-sum distance and markers.

### Scientific impact

- The exact critical effect is defined as the smallest directed working-scale effect meeting the
  requested conservatively rounded selected-claim probability under a fixed-SE one-parameter
  normal/Wald model.
- The legacy z-sum calculation remains a separately labeled closed-form benchmark. Neither quantity
  is a confidence bound, observed estimate, clinically validated meaningful effect, or
  study-specific sample-size calculation.

## [0.2.1] - 2026-07-30

### Fixed

- Made `support_interval` and `support_interval_for_ratio` fail closed when finite binary64
  endpoint quantization produces a materially different support boundary.
- Independently re-evaluate every non-clipped endpoint with the exact-binary64 pairwise log-support
  kernel and require relative agreement at `1e-12` with no absolute-tolerance floor.
- Preserved deliberately overflow-clipped endpoints and their explicit flags, without claiming that
  the clipped value equals the requested analytic boundary.

### Validation

- Added the exact hexadecimal `theta_hat=1e308` adjacent-float regression: criteria from 2:1 through
  8:1 previously collapsed to endpoints whose actual ratio was `6.825935561925903`.
- Added unit, property, independent rational scientific-reference, near-zero cutoff, minimum
  subnormal standard-error, clipping, and cold-wheel regressions.
- Retained exact root and legacy API contracts and zero-difference frozen parity across all 23,095
  compared values.

### Scientific impact

- Analytic Wald formulas and all accurately representable interval endpoints are unchanged.
  Extreme scale combinations that cannot encode the requested boundary now raise `ValidationError`
  rather than returning a finite but scientifically mislabeled interval.

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

[Unreleased]: https://github.com/reblocke/wald-inference-core/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/reblocke/wald-inference-core/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/reblocke/wald-inference-core/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/reblocke/wald-inference-core/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/reblocke/wald-inference-core/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/reblocke/wald-inference-core/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/reblocke/wald-inference-core/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/reblocke/wald-inference-core/releases/tag/v0.1.0
