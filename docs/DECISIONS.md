# Decisions

Use this file for durable decisions that are not obvious from code alone. Append superseding entries
rather than rewriting historical decisions.

## 2026-07-29: Extract one pure numerical source of truth

**Context:** The integrated `conf_curve_likelihood` repository combined numerical formulas, browser
payloads, UI rendering, and exports. The applet portfolio needs independently deployable interfaces
without formula forks.

**Decision:** Place effect transformations, Wald reconstruction, compatibility, normalized relative
support, detectability benchmarks, selection rules, Type S/M, and precision calculations in this
pure-Python package. Browser contracts, plotting, prose generation, exports, and app-specific
adapters remain downstream.

**Consequences:** Every downstream release pins a released core version. Missing mathematical
primitives are implemented and released here before adoption. Cross-repository formula copies are a
release blocker.

## 2026-07-29: Preserve the tagged baseline for v0.1.0

**Context:** Extraction can accidentally alter tail boundaries, tolerances, finite-value behavior, or
undefined-value conventions even when formulas appear equivalent.

**Decision:** Anchor v0.1.0 to `pre-split-baseline-2026-07-29`, tag target
`5fd501dd947d9b951d736014cfc2b310efa5e7b0`, and approved behavior source
`830756ecb11b4e8161f8dfe1fc75afc346ef4467`. Compare core-owned values at `rtol=1e-12` and
`atol=1e-14`; compare discrete contract values exactly.

**Consequences:** Refactoring is permitted only with parity evidence. Formula improvements are
deferred and require an explicit scientific-impact decision and versioned release.

## 2026-07-29: Use explicit package and API boundaries

**Context:** Repository, distribution, and import names have distinct conventions.

**Decision:** Use repository `wald-inference-core`, distribution `wald-inference`, and import package
`wald_inference`. Export a deliberate typed public surface from `wald_inference.__all__`; private
helpers remain private. Compatibility aliases delegate to canonical functions.

**Consequences:** Package metadata and release checks enforce the naming distinction. Downstream
adapters depend only on documented public imports.

## 2026-07-29: Use Brian Locke and MIT consistently

**Context:** The source metadata audit resolved conflicting historical names and a license
placeholder before extraction.

**Decision:** Use `Brian Locke` as author and maintainer, retain the MIT License, and use
`Copyright (c) 2026 Brian Locke`.

**Consequences:** `pyproject.toml`, `CITATION.cff`, `README.md`, `LICENSE`, changelog, and release
metadata must agree. Do not infer an email, affiliation, middle initial, ORCID, or DOI.

## 2026-07-29: Publish verified GitHub artifacts before stable promotion

**Context:** The core must release a wheel and source distribution, but PyPI publication is not
authorized. Independent portfolio validation needs immutable artifacts before final promotion.

**Decision:** Build byte-reproducible artifacts from an annotated tag, cold-install the wheel,
publish wheel, sdist, checksums, and parity report to a GitHub prerelease, and promote that same
release after independent validation. The write-enabled job never rebuilds artifacts.

**Consequences:** Promotion changes release status only. Failed validation produces a new patch
version rather than replacing published assets. PyPI remains out of scope.

## 2026-07-29: Pin the build backend and use SPDX license metadata

**Context:** Unpinned isolated build requirements undermine reproducibility, and the source
repository's table-form license field is deprecated by current setuptools.

**Decision:** Pin the setuptools backend in the project and locked build group, build with
`--no-build-isolation`, declare `license = "MIT"` and `license-files = ["LICENSE"]`, and set
`SOURCE_DATE_EPOCH` from the release commit.

**Consequences:** Release CI builds two archived copies and requires byte-identical artifacts.
Backend upgrades are explicit dependency changes with rebuilt parity and packaging evidence.

## 2026-07-29: Isolate frozen direct-call behavior in an adapter-only module

**Context:** The integrated workbench historically called `confcurve.core` functions directly.
Some malformed and nonfinite inputs therefore produced raw Python exceptions, NumPy warnings, or
nonfinite arrays before the browser contract applied its strict response checks. Making those edge
calls strict during rewiring would be a migration behavior change.

**Decision:** Preserve that direct-call contract only under `wald_inference.legacy`. Canonical root
functions remain finite and normalize malformed inputs to `ValidationError`. Both surfaces delegate
to the same private formula kernels, and the legacy module is excluded from the root `__all__`.

**Consequences:** The workbench adapter can migrate without semantic drift, including its null
summary sentinels. New consumers must not depend on the legacy module, and the compatibility surface
may be removed only after downstream adapters no longer require it.

## 2026-07-29: Stabilize adapter configuration imports in the legacy module

**Context:** The integrated workbench adapter also needs the frozen numerical bounds, defaults,
tolerances, quantiles, solver limits, and asymmetry-warning helper. Importing those names from
implementation submodules would couple the adapter to private module layout even though their
values and behavior must remain fixed during migration.

**Decision:** In v0.1.1, directly re-export the required existing definitions from
`wald_inference.legacy`, document their exact names, and lock that module's `__all__`. Do not expand
the root `wald_inference.__all__`, recompute any constant, or add a second warning implementation.

**Consequences:** Backward-compatibility adapters may rely on the documented legacy imports while
general consumers continue to use canonical root functions. Any future removal or value change
requires an explicit versioned migration; this decision does not broaden the package's scientific
scope or change numerical behavior.

## 2026-07-29: Expose generic support criteria through log-domain-first APIs

**Context:** A focused relative-support app needs arbitrary pairwise comparisons and support
interval criteria such as 2:1, 4:1, and 8:1. Computing these relationships downstream would fork
the Wald formula, while changing the existing `support_interval` signature or S−2 default would
unnecessarily disturb the frozen interface.

**Decision:** In v0.2.0, add root-public `log_support_ratio`, scalar `support_ratio`, and
`support_interval_for_ratio`. Define pairwise ordering explicitly as `log L(A) - log L(B)`, use the
log-domain result as the authoritative extreme-value representation, require finite
MLE-to-bound ratios greater than one, and delegate ratio-based intervals to the existing canonical
log-cutoff implementation. Preserve the existing `support_interval` API and S−2 default unchanged.

**Consequences:** Downstream applications can offer generic support criteria without app-local
formulas. They must retain and label the log ratio when exponentiation overflows, identify the
numerator and denominator, and describe all results as normalized Wald approximations rather than
exact fitted likelihoods, Bayes factors, or posterior probabilities. The legacy module and every
pre-v0.2.0 calculation remain unchanged.

## 2026-07-30: Verify finite support endpoints and fail closed when unrepresentable

**Context:** At an extreme working-scale center, the analytic half-width can be smaller than or
comparable to one binary64 spacing unit. Multiple requested support criteria can then round to the
same adjacent endpoints even though those floats have one fixed, materially different support
ratio. Finite-range clipping flags do not describe this condition because the endpoints themselves
are finite.

**Decision:** In v0.2.1, construct an exact-input fallback endpoint and independently certify every
non-clipped support-interval endpoint with exact rational arithmetic over its binary64 inputs.
Require agreement with the requested MLE-to-bound log support at relative tolerance `1e-12` and
absolute tolerance `0`. Preserve the v0.2.0 endpoint bit pattern whenever it passes that
certification, use the fallback when half-scaling erases a representable subnormal center, and
raise `ValidationError` when neither endpoint is faithful. Continue returning explicitly flagged
overflow-clipped endpoints without claiming that a clipped bound equals the requested boundary.

**Consequences:** Inputs whose analytic boundaries are not accurately representable now fail closed
instead of returning a misleading interval. The no-absolute-floor rule also protects near-zero
cutoffs from silently collapsing to the center. Representable endpoints, formulas, root-public
names, the legacy adapter, and frozen baseline values remain unchanged.

## 2026-07-30: Separate exact detectability from the legacy z-sum benchmark

**Context:** A focused critical-effect app needs vectorized selected-claim probability and inverse
detectability thresholds. The frozen integrated calculation instead provides only the fixed
`alpha=0.05`, nominal-80% z-sum marker, which is close to but not exactly the solution of the
two-tailed probability equation. Reimplementing selection tails in each downstream app would create
multiple numerical authorities.

**Decision:** In v0.3.0, add root-public `selected_claim_probability`, `power_curve`,
`critical_effect_for_target_probability`, and immutable `CriticalEffectResult`. Route every forward
probability through the canonical selection intervals for all six existing rules. Restrict monotonic
inversion to the two-sided, matching one-sided positive, and matching one-sided negative p-value
rules. Normalize exact-null probability locally to alpha without changing the frozen selection
module. Use fixed Gauss-Legendre integration of the canonical positive derivative for
`abs(delta) <= 0.125`; outside that neighborhood, evaluate one-sided tails directly and two-sided
unselected probability in a stable complement/log domain. Apply a small directed-rounding guard so
the public float is a conservative representation of the exact model probability: guard the
canonical critical value upward by four ULPs, use a lower quadrature bound for selected-direction
near-null increments, use a guarded direct tail in the opposing one-sided direction, and apply a
64-ULP probability-component guard increased to 256 ULPs below `1e-8`. Use this one public kernel
for arrays, scalar calls, generic monotonic bisection, and achieved probability, regardless of
target proximity to alpha or one. Require each returned magnitude to satisfy the public probability
while its immediately preceding float does not. Compose and re-evaluate the working-scale effect,
use only relative representability tolerance with no absolute floor, and search the ordered
binary64 working-effect lattice rather than imposing a fixed adjacent-float adjustment cap. Reject
unsupported, uncertifiable, or unrepresentable results. Preserve
`legacy_critical_effect_distance` and `legacy_critical_effect_markers` unchanged as a separately
labeled closed-form benchmark.

**Consequences:** Downstream applications can calculate exact Wald detectability without copying
selection formulas and can compose log-scale results through the effect registry. They must keep
the exact result distinct from the legacy benchmark, meaningful-effect inputs, confidence bounds,
observed estimates, and study-specific sample-size calculations. The six selection rules and all
pre-v0.3.0 outputs remain unchanged.

## 2026-07-30: Aggregate precision guardrails without changing per-target authority

**Context:** A focused inverse-precision app needs a typed joint result, binding constraints, and
assumed-effect sensitivity. The released v0.3.0 API solves individual probability, Type S, and Type
M precision targets but leaves aggregation and infeasibility interpretation to downstream code.
Reimplementing either target formulas or aggregation independently in each app would create
divergent numerical and semantic authorities.

**Decision:** In v0.4.0, retain `precision_target_results` as the numerical authority and add
root-public immutable `JointPrecisionResult`, `joint_precision_result`, and
`precision_sensitivity`. Solve each requested target independently through the existing forward
metrics and monotonic SE solver. If all mandatory targets are feasible, choose the smallest
required SE and largest information multiplier and report all multipliers tied under relative
tolerance `1e-8` with zero absolute tolerance. Clamp neither a target nor the joint requirement
below current information: current sufficiency remains exactly `1.0`. If any mandatory target is
infeasible, return no joint numeric solution while preserving every target row and naming the
relevant target and assumptions. Preserve near-null undefined Type S/M behavior. Make sensitivity
a deterministic ordered scalar map, not a new numerical solver.

Add read-only feasibility, current-sufficiency, and selected-claim-probability aliases to
`PrecisionTargetResult`; do not change its existing dataclass fields, serialized form, notes, or
B06/B07 values. Keep the legacy `solve_required_precision` dictionary behavior, including its
all-`None` no-target and mandatory-infeasibility results.

**Consequences:** Downstream apps receive one typed authority for joint feasibility, binding
constraints, and sensitivity without copying forward or inverse formulas. Joint and sensitivity
results remain conditional one-parameter Wald calculations. Relative information is not declared
to be sample size; study-design projection, UI, plotting, and exports remain downstream concerns.
All pre-v0.4.0 numerical outputs and the legacy adapter remain unchanged.
