# Maintenance

## Version policy

The package follows semantic versioning:

- patch: backward-compatible numerical bug fix or packaging/documentation correction;
- minor: new backward-compatible public function or supported behavior; and
- major: incompatible public API or scientific-contract change.

A numerical bug fix must include a scientific-impact note describing affected inputs and outputs.
Do not silently change formulas, selection boundaries, tolerances, or undefined-value conventions.

## Development workflow

1. Branch from current `main`.
2. Characterize behavior before changing numerical code.
3. Make the smallest scoped change.
4. Update tests, API/scientific docs, decisions, and changelog together.
5. Run:

   ```bash
   uv sync --locked --all-groups
   make verify
   git status --short
   ```

6. Review the parity report rather than relying only on a zero exit code.
7. Merge only after required CI and review pass at the expected head.

Dependency upgrades must be explicit. Re-lock the environment, review transitive changes, run the
full suite and parity corpus, rebuild artifacts, and document any numerical delta. Do not widen
tolerances to absorb an unexplained dependency change. Dependabot pull requests are review
proposals, not merge authority; major GitHub Action upgrades remain explicit decisions. The
automated updater must ignore NumPy 2.3 or newer and SciPy 1.15 or newer while the released
`numpy>=2.2.5,<2.3` and `scipy>=1.14.1,<1.15` scientific compatibility ranges remain in force.
Changing either ceiling requires an explicit dependency-compatibility review and release.

## Release workflow

1. Finalize the changelog section, CFF date/version, package version, and `__version__`.
2. Verify live repository settings: required review/CI for `main`, protection for released `v*`
   tags, read-only default workflow permissions, private vulnerability reporting, dependency
   alerts and Dependabot security updates, and immutable releases.
3. Confirm `main` is clean, CI passes, and the expected reviewed head is exact.
4. Create an annotated tag `vX.Y.Z` at that exact commit.
5. Push the tag. The workflow installs an exact checksummed GitHub CLI, then, before installing
   repository code, binds the local/event tag object to the exact remote tag object, confirms its
   target, requires that target to be contained in the protected `main` history, and checks the tag
   name against the package version using isolated Python.
6. The read-only job reruns all gates, builds twice for byte reproducibility, cold-installs the
   wheel, extracts only the tagged version's changelog section, and transfers one checksummed
   bundle.
7. A separate read/OIDC/attestations job generates build provenance for the wheel and source
   distribution without release-write permission.
8. The contents-write job creates a draft with exactly the wheel, source distribution,
   `SHA256SUMS`, and parity report, re-downloads and byte-compares them, compares the draft body
   byte-for-byte, and publishes once as stable.
9. Require the published release to report immutable and verify the release and each release asset
   attestation. Then run independent cold-clone and downstream-adapter validation and record the
   release URL, commit, workflow run, artifact names, SHA-256 values, and adoption status.

The draft is the candidate release; there is no published-prerelease promotion stage. If an
internal candidate check fails, leave the draft visible for inspection. If post-publication
validation finds a defect, document it, fix it in a new commit/version, and release a new annotated
tag. Never replace a published binary or move a published tag.

PyPI publication is not authorized. Downstream repositories must pin the exact GitHub release wheel
and `uv.lock` resolution, not `main`.

## Public API changes

- Add public names deliberately to `wald_inference.__all__`.
- Keep type annotations and `docs/API.md` synchronized.
- Preserve compatible aliases until downstream adapters migrate.
- Deprecations require documentation, tests, and at least one appropriate release before removal.
- Breaking changes require a major version and explicit downstream migration plan.

## Scientific issue response

For a suspected numerical defect:

1. reproduce with synthetic values and preserve the exact environment, observed output, and
   expected rationale without publishing PHI, patient-level data, credentials, or restricted data;
2. determine whether the frozen baseline shares the behavior;
3. add a failing regression test;
4. obtain approval for any formula or convention change;
5. implement one canonical fix;
6. run unit, property, scientific-reference, parity, and downstream tests; and
7. publish a patch release with a scientific-impact note.

Security, dependency-compromise, privacy, and release-integrity reports follow the private process
in `SECURITY.md`, not a public numerical issue.

## Repository boundaries

Do not add browser code, plotting, app contracts, UI prose, CSV/PNG export, telemetry, persistence,
accounts, or server infrastructure. A downstream need for numerical behavior belongs here; a
presentation need belongs in the consuming application.
