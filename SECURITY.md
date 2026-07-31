# Security Policy

## Supported versions

Security fixes are considered for the latest GitHub release and the current `main` branch. Older
tags and release assets remain immutable reproducibility records; they are not silently moved,
rebuilt, or replaced.

## Report a vulnerability privately

Use GitHub's **Report a vulnerability** control in this repository's Security tab:

<https://github.com/reblocke/wald-inference-core/security/advisories/new>

Private vulnerability reporting is a GitHub repository setting, not something this file enables.
If the control is unavailable, open only the repository's **Private security coordination
request** issue form. That public fallback must contain no vulnerability class, affected
component, reproduction, impact, exploit detail, secret, or sensitive value. It exists only to
request a private coordination route.

Do not disclose vulnerability details in a public issue, pull request, discussion, commit,
workflow log, or release note. A private report should include:

- the exact released tag or full commit SHA;
- the affected package, dependency, workflow, artifact, or release path;
- a minimal reproduction using synthetic numerical values;
- expected and observed behavior;
- environment and dependency versions; and
- any suspected effect on credentials, artifact integrity, confidentiality, or availability.

Never submit protected health information, patient-level data, credentials, tokens, unpublished
restricted data, local-path details, or other sensitive material. Redact logs and use the smallest
synthetic artifact that demonstrates the issue.

## Scope distinctions

- A vulnerability, credential exposure, dependency compromise, or release-integrity defect belongs
  in private vulnerability reporting.
- A suspected numerical or scientific discrepancy belongs in the public numerical-behavior issue
  process only when it can be reproduced safely with synthetic aggregate values and contains no
  embargoed or sensitive information.
- A routine nonsensitive packaging, documentation, or repository defect may use the public
  engineering issue form.
- Clinical interpretation and patient-specific use are out of scope. This mathematical library is
  not clinical decision support or a regulated medical device.

A security fix requires a new reviewed commit, version, annotated tag, and immutable release.
Never move an affected tag or replace a published asset.
