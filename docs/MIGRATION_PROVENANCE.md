# Migration Provenance

## Source authority

| Item | Value |
|---|---|
| Source repository | [`reblocke/conf_curve_likelihood`](https://github.com/reblocke/conf_curve_likelihood) |
| Audited pre-fix commit | `f77cd13f0286e933a66c0997af288a0dfa167bd5` |
| Approved behavior source | `830756ecb11b4e8161f8dfe1fc75afc346ef4467` |
| Frozen baseline tag | `pre-split-baseline-2026-07-29` |
| Tag target | `5fd501dd947d9b951d736014cfc2b310efa5e7b0` |
| Golden manifest SHA-256 | `f54bb2d8311788c07adcf23fc9f038e35702449e4a77a474abea9411246cabcc` |
| Fixture-set SHA-256 | `81c341b39e711caffc85a444f0c1e4bc1e2d00633474c82e720afeb60def3c4d` |
| Source license | MIT |

The approved behavior source includes the separately reviewed finite-range safety fix. Successful
responses contain only finite numeric values, undefined inferential quantities use `None`, and
unrepresentable derived values raise the public validation exception.

## Exact import checkpoint

Commit `3d30a0f5e1a4f240974c7715520f8a7249f90646` imported four files byte-for-byte before
refactoring:

| Source path | Source Git blob | Imported SHA-256 |
|---|---|---|
| `src/confcurve/core.py` | `d28bc1962c9028eba94338f6738cd769800b14f2` | `346dd746be2c257dfc02f0822fa46e025c56bfc911733746c090e7df37d470a7` |
| `src/confcurve/design.py` | `44ba281e044a691ca514e668ca9cf65d3e7045a9` | `f20af34da0daebb7e7682eeaa8ef644d5a66082cf0f0d3826bf4260bef2cda1b` |
| `src/confcurve/models.py` | `1ee914d0931299d25083eda5b604bf9a0e08c8a8` | `33a06b6ab55fefddd86d44323b6bea192c7016566fc5c2220f6ed1fc577112fe` |
| `src/confcurve/web_contract.py` | `7a28582d25819324556a020b4e64998b252f6627` | `79fc4b9eead6d86dd330aeaf9e545f0f4ed87429ee6698e69e6cd0b658891aaf` |

The temporary `provenance-import/` snapshot is deleted after refactoring so the released repository
contains one implementation of each formula. Git history permanently retains the exact checkpoint.

## Responsibility mapping

| Frozen source responsibility | Core destination |
|---|---|
| Validation exception and finite checks | `errors.py` and private validated helpers |
| Effect registry and transformations | `effects.py` |
| CI-to-estimate/SE reconstruction | `reconstruction.py` |
| Reusable finite grids | `grid.py` |
| Standardized distance and compatibility | `compatibility.py` |
| Relative/log-relative likelihood and support | `likelihood.py` |
| Legacy critical-effect benchmark | `detectability.py` |
| Selection-rule definitions and boundaries | `selection.py` |
| Selected-claim, Type S, Type M, observed exaggeration | `type_sm.py` |
| Inverse precision targets and information scaling | `precision.py` |
| Typed domain results | `types.py` |

Browser `TypedDict` payloads, UI warnings/prose, Plotly/export logic, and display-only layout fields
are not migrated. Where `web_contract.py` contained numerical orchestration, parity tests identify
the core-owned calculation while the new package exposes domain objects rather than browser payloads.

## Extraction method

1. Initialize the public MIT repository with canonical Brian Locke metadata.
2. Import exact source blobs and record source/tag/hash authority.
3. Add characterization and frozen-parity tests.
4. Refactor the copied functions into responsibility-focused modules while preserving formulas.
5. Delete the temporary source snapshot.
6. Verify every core-owned baseline value at `rtol=1e-12`, `atol=1e-14`, with discrete values
   compared exactly.
7. Build, inspect, reproduce, and cold-install the release artifacts.

The golden fixtures are generated characterization outputs from the recorded source. They are not
independent empirical or theoretical reference data.

## Canonical metadata

The source repository owner explicitly selected `Brian Locke` as the canonical public author and
maintainer and retained MIT with `Copyright (c) 2026 Brian Locke`. No email, affiliation, middle
initial, ORCID, or DOI is inferred.

## Methodology references carried forward

The source repository documents:

- Zampieri et al., *American Journal of Respiratory and Critical Care Medicine* (2025), for
  evidential likelihood, likelihood ratios, support, and S−2 terminology; source retrieval date
  2026-04-23:
  <https://academic.oup.com/ajrccm/article/211/9/1610/8300617>
- Perugini et al., *Advances in Methods and Practices in Psychological Science* (2025), for
  critical-effect-size values and design-interpretation rationale; source retrieval date
  2026-04-23:
  <https://journals.sagepub.com/doi/10.1177/25152459251335298>
- Gelman and Carlin (2014), for Type S error, Type M exaggeration, and repeated-study design
  calculations; source retrieval date 2026-06-14:
  <https://journals.sagepub.com/doi/abs/10.1177/1745691614551642>

No external figure, table, dataset, or substantial copied text is added by the extraction. These
references support terminology and methodology; they do not supersede the frozen implementation as
the v0.1.0 parity authority.

## Release evidence

The v0.1.0 artifact names, SHA-256 values, GitHub Actions run, release URL, and independent
fresh-clone results are intentionally recorded only after those actions occur. The tag workflow
publishes `baseline-parity.json` alongside the wheel, source distribution, and `SHA256SUMS`; the
portfolio migration log records the observed external evidence.
