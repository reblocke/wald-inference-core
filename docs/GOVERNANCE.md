# Governance

## Scope and authority

This repository is maintained by Brian Locke (`@reblocke`) as the pure-Python numerical authority
for the Wald-inference applet portfolio. Repository governance does not override the scientific
authority hierarchy in `AGENTS.md`, the frozen parity baseline, or an approved study requirement.

Numerical formulas, public APIs, tolerances, selection boundaries, effect definitions, and
undefined-value conventions require explicit review. Routine packaging, documentation,
dependency, and workflow maintenance may proceed only when the numerical contract remains
unchanged and the full validation gate passes.

## Decision process

- Changes are proposed through reviewable branches and pull requests.
- The maintainer resolves scope, scientific-authority, security, and release decisions.
- Durable changes to scientific or release authority are recorded in `docs/DECISIONS.md`; old
  decisions remain as history and later entries supersede them explicitly.
- A numerical defect requires a regression, an impact statement, parity review, and a new release.
- A contributor must disclose a material conflict of interest relevant to the proposed change.

No issue, pull request, test result, or automated dependency update merges itself. Passing
automation is necessary evidence, not authorization.

## Repository settings outside Git

Repository files cannot prove or enable GitHub administrative settings. Before a protected merge or
release, the maintainer must verify the live settings appropriate to the operation, including:

- required pull-request review and successful CI for `main`;
- protection against deletion or movement of released `v*` tags;
- read-only default workflow token permissions;
- private vulnerability reporting;
- dependency alerts and Dependabot security updates; and
- immutable releases.

If a required setting is unavailable or disabled, stop before the affected operation. Do not
weaken a workflow to route around the setting. The immutable-release workflow gate uses the
`RELEASE_SETTINGS_READ_TOKEN` Actions secret. It must contain a fine-grained, expiring token
restricted to this repository with Administration **read** permission; it is not used to publish
or modify settings.

## Security and privacy

Vulnerabilities use the private process in `SECURITY.md`. Public collaboration uses synthetic
numerical inputs only and must not contain credentials, protected health information, patient-level
data, restricted source material, or sensitive logs.

## Release authority

A future release requires an exact expected head, synchronized package/citation/changelog metadata,
a GitHub-verified signed annotated tag, all local and CI gates, a reproducible wheel and source
distribution, SHA-256 checksums, frozen-parity evidence, build-provenance attestations, exact draft
asset/body verification, an exact checksummed GitHub CLI version selected before credentialed
release commands, and live repository release immutability.

The draft is the candidate release. After its exact assets and current-version notes are verified,
it is published once as a stable immutable release. There is no published-prerelease promotion
stage because an immutable published release cannot be changed into a different publication state.
Failed candidates remain drafts for inspection; a published defect requires a new version and tag.

GitHub Releases are the only authorized distribution channel. PyPI publishing is prohibited unless
the user grants separate explicit authorization in a future decision.
