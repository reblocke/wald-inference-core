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
tolerances to absorb an unexplained dependency change.

## Release workflow

1. Finalize the changelog section, CFF date/version, package version, and `__version__`.
2. Confirm `main` is clean and CI passes.
3. Create annotated tag `vX.Y.Z` at the reviewed `main` commit.
4. Push the tag. The tag workflow reruns all gates, builds twice, checks byte reproducibility,
   cold-installs the wheel, and publishes a GitHub prerelease.
5. Run independent cold-clone and downstream-adapter validation against the released assets.
6. Promote the existing GitHub release from prerelease to stable without changing its tag or
   assets.
7. Record the release URL, commit, workflow run, artifact names, SHA-256 values, and downstream
   adoption status.

If candidate validation fails, leave the evidence visible, document the finding, fix it in a new
commit/version, and release a new tag. Never replace a published binary under the same tag.

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

1. preserve the exact input, environment, observed output, and expected rationale;
2. determine whether the frozen baseline shares the behavior;
3. add a failing regression test;
4. obtain approval for any formula or convention change;
5. implement one canonical fix;
6. run unit, property, scientific-reference, parity, and downstream tests; and
7. publish a patch release with a scientific-impact note.

## Repository boundaries

Do not add browser code, plotting, app contracts, UI prose, CSV/PNG export, telemetry, persistence,
accounts, or server infrastructure. A downstream need for numerical behavior belongs here; a
presentation need belongs in the consuming application.
