## Scope

Describe the numerical, packaging, documentation, governance, or maintenance problem addressed.
State whether any public API, formula, tolerance, selection boundary, effect definition, or
undefined-value convention changes.

## Risk and release impact

Describe silent-failure risks, scientific and privacy implications, dependency or artifact effects,
and whether the change requires a new release.

## Verification

List exact commands and outcomes. Include parity maxima, artifact checks, skipped tests, warnings,
and generated files when applicable.

## Checklist

- [ ] Numerical behavior and the public API are unchanged, or an approved scientific-impact
      decision and regression coverage are included.
- [ ] Frozen parity tolerances remain `rtol=1e-12` and `atol=1e-14`.
- [ ] Examples, fixtures, logs, and screenshots are synthetic and contain no credentials,
      sensitive data, patient-level data, or protected health information.
- [ ] Every third-party GitHub Action is pinned to a live full commit SHA with an exact version
      comment.
- [ ] No browser/UI behavior, telemetry, persistence, server infrastructure, or PyPI publishing was
      added.
- [ ] `uv sync --locked --all-groups` and `make verify` pass.
- [ ] `reports/baseline-parity.json` and built wheel/source-distribution contents were reviewed.
- [ ] README, API/scientific scope, validation, privacy, governance, decisions, maintenance,
      citation, and changelog were reviewed for synchronization.
