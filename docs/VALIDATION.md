# Validation

## Frozen authority

The first release is a behavior-preserving extraction from:

| Item | Value |
|---|---|
| Source repository | `reblocke/conf_curve_likelihood` |
| Baseline tag | `pre-split-baseline-2026-07-29` |
| Tag target | `5fd501dd947d9b951d736014cfc2b310efa5e7b0` |
| Approved behavior source | `830756ecb11b4e8161f8dfe1fc75afc346ef4467` |
| Golden manifest SHA-256 | `f54bb2d8311788c07adcf23fc9f038e35702449e4a77a474abea9411246cabcc` |
| Fixture-set SHA-256 | `81c341b39e711caffc85a444f0c1e4bc1e2d00633474c82e720afeb60def3c4d` |

Core-owned floating-point values are compared at `rtol=1e-12` and `atol=1e-14` under the
locked Python, NumPy, and SciPy stack. Core-owned strings, enum keys, booleans, `None`, integer
counts, and effect-registry keys are compared exactly. Core reconstruction-warning behavior is
covered separately by unit tests because the baseline response warning list also contains app-owned
grid and presentation warnings. A tolerance may not be widened merely to make a failure pass.

Browser payload key order, captions, reviewer prose, Plotly layouts, PNG dimensions, and CSV
presentation are intentionally excluded because they are not owned by this package.

## Baseline cases

The parity corpus covers:

| Cases | Core behavior |
|---|---|
| B01 | Additive reconstruction, peak compatibility/support, CI-bound values |
| B02 | Ratio/log-scale reconstruction and identity/log equivalence |
| B03 | Presentation-only display-window invariance for core summaries |
| B04 | Forward selected-claim, Type S/M, and scenario behavior |
| B05 | Directional threshold rule, ratio working scale, information scaling |
| B06 | Inverse precision targets, achieved constraints, ordering |
| B07 | Near-null undefined values, invalid inputs, infeasible targets |
| B08 | Safe midpoint/difference, clipping, log support, strict finite handling |

The parity runner produces a human-readable console summary and
`reports/baseline-parity.json`. A missing, malformed, skipped, or failing report is a release
failure.

## Test layers

- **Unit tests** cover transformations, reconstruction, observed functions, pairwise support and
  ratio-based support intervals, six-rule detectability probability, restricted critical-effect
  inversion, finite endpoint representability, selection rules, Type S/M, per-target and joint
  precision, sensitivity, strict serialization, error behavior, and the exact root and adapter
  compatibility surfaces.
- **Property tests** cover valid finite ranges, symmetry, identity/log equivalence, pairwise
  antisymmetry, fail-closed requested endpoint support, detectability monotonicity and symmetry,
  information scaling, joint-envelope equivalence, guardrail ordering, scalar/sensitivity
  consistency, and public-probability predecessor minimality across low and high alpha values.
- **Scientific-reference tests** compare formulas with independently expressed normal-distribution
  identities or closed forms, including exact-rational evaluation of extreme binary64 support
  neighbors, analytic one-sided and direct-tail two-sided detectability, hard-coded mpmath quantile
  differences, conservative-sided high-precision roots, extreme-alpha complement behavior,
  floating-boundary roots, solved-guardrail forward metrics, information/CI-width identities,
  effect-sensitivity ordering, ratio-scale transformation, and direct-SE/CI reconstruction
  equivalence.
- **Regression tests** compare every core-owned value with the frozen corpus and preserve exact
  finite/undefined conventions.
- **Packaging tests** inspect wheel/sdist metadata and contents.
- **Cold-wheel smoke** imports and exercises the installed wheel outside the checkout.

## Local release gates

```bash
uv sync --locked --all-groups
make fmt-check
make lint
make metadata-check
make test
make parity
make build
make smoke
git diff --check
git status --short
```

`make build` requires an absent `dist/` directory so stale artifacts cannot be mistaken for the
current build. `make clean` removes generated local artifacts.

## Release artifact gates

The tag workflow:

1. installs an exact checksummed GitHub CLI version before any credentialed release command;
2. requires an annotated tag whose internal and ref names match the package version, binds the
   local/event tag object to the verified remote object, and requires its target to be the exact
   workflow commit;
3. requires GitHub to report a valid cryptographic signature on that annotated tag before
   installing or executing repository code;
4. requires the verified tag target to be contained in the protected `main` history;
5. reads the project version with isolated Python and verifies package, citation, changelog, and
   `__version__` agreement;
6. reruns format, lint, tests, and frozen parity under `uv.lock`;
7. archives the same tag twice and builds with the locked backend and fixed source epoch;
8. requires byte-identical wheels and source distributions and inspects their contents;
9. installs the wheel into an empty virtual environment and runs public-API smoke checks;
10. extracts only the tagged version's nonempty changelog section;
11. transfers one exact four-asset bundle to later jobs;
12. generates GitHub build-provenance attestations for the wheel and source distribution in a
    separate job with read, OIDC, and attestations permissions but no release-write permission;
13. requires immutable releases before creating a draft;
14. creates a draft containing exactly the wheel, source distribution, `SHA256SUMS`, and
    machine-readable parity report;
15. re-downloads and byte-compares every draft asset, reruns the hosted checksums, and requires the
    draft body to match the generated current-version notes byte-for-byte;
16. reconfirms immutability immediately before publication; and
17. publishes once as stable, then verifies immutable-release and release-asset attestations.

The expected v0.4.1 assets are:

```text
wald_inference-0.4.1-py3-none-any.whl
wald_inference-0.4.1.tar.gz
SHA256SUMS
baseline-parity.json
```

## Evidence status

The source baseline identifiers above are fixed and verified by the imported provenance checkpoint.
Final core test counts, artifact hashes, GitHub Actions run URL, release URL, and fresh-clone results
do not exist until their respective commands and external actions complete. They must be recorded in
the release and portfolio migration log after observation; this document does not predict them.
Repository files also do not prove that private vulnerability reporting, Dependabot security
updates, branch/tag protection, or immutable releases are enabled. Those settings require live
verification.
