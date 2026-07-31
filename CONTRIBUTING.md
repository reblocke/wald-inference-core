# Contributing

## Repository scope

`wald-inference` is the numerical source of truth for the Wald-inference applet portfolio. Changes
must preserve its documented formulas, public API, finite-value behavior, effect registry,
selection rules, tolerances, and frozen parity contract unless a separately approved scientific
change explicitly supersedes them.

Use public issue forms only for nonsensitive numerical or engineering reports. Report
vulnerabilities through [SECURITY.md](SECURITY.md). Never put credentials, protected health
information, patient-level data, unpublished restricted data, or other sensitive values in an
issue, pull request, fixture, screenshot, URL, log, or workflow artifact. Reproductions and new
fixtures must use synthetic numerical values.

## Change process

1. Start from the current `main` branch and make one reviewable change.
2. State the goal, assumptions, success criteria, silent-failure risks, and verification before
   editing.
3. Characterize existing behavior before changing numerical or release code.
4. Add a failing regression before a bug fix when feasible.
5. Keep one canonical implementation for each formula; compatibility aliases must delegate.
6. Update tests, API/scientific documentation, decisions, maintenance guidance, and changelog
   together when their authority changes.
7. Keep every third-party GitHub Action pinned to a live full commit SHA with an exact version
   comment.
8. Open a pull request and let required checks and review complete at the expected head.

Do not widen a tolerance to absorb an unexplained difference. Do not add browser code, UI
contracts, telemetry, persistence, accounts, servers, or PyPI publishing.

## Verification

Restore the locked environment and run the complete repository gate:

```bash
uv sync --locked --all-groups
make verify
git diff --check
git status --short
```

Review `reports/baseline-parity.json`, the wheel and source-distribution contents, and the
cold-install smoke result. Report every skipped check, warning, and generated artifact.

## Release changes

A release change requires a reviewed pull request and a GitHub-verified signed annotated tag. The
tag must equal `v` plus the authoritative package version and point to the exact reviewed commit.
The release workflow:

1. installs and selects an exact checksummed GitHub CLI version;
2. binds the local/event annotated tag object to the GitHub-verified remote object and verifies it
   before installing or executing repository code;
3. runs the locked test, parity, reproducibility, distribution, and cold-install gates;
4. builds and transfers one complete exact-version asset bundle;
5. generates build-provenance attestations for the wheel and source distribution in a separate
   narrowly permissioned job;
6. requires immutable releases before creating a draft;
7. creates a draft with exactly the wheel, source distribution, checksums, and parity report;
8. re-downloads and byte-compares every asset and verifies the current version's release body
   byte-for-byte;
9. reconfirms immutability; and
10. publishes the verified draft once as a stable immutable release.

If any check fails after draft creation, retain the unpublished draft for inspection. Do not move a
tag or replace an asset. PyPI publication remains prohibited.
