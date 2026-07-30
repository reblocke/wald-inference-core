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
