# Exact source import checkpoint

These four files were imported byte-for-byte from the annotated
`pre-split-baseline-2026-07-29` tag in
[`reblocke/conf_curve_likelihood`](https://github.com/reblocke/conf_curve_likelihood/tree/pre-split-baseline-2026-07-29).

- Tag target: `5fd501dd947d9b951d736014cfc2b310efa5e7b0`
- Approved behavior source: `830756ecb11b4e8161f8dfe1fc75afc346ef4467`
- Golden manifest SHA-256:
  `f54bb2d8311788c07adcf23fc9f038e35702449e4a77a474abea9411246cabcc`
- Fixture-set SHA-256:
  `81c341b39e711caffc85a444f0c1e4bc1e2d00633474c82e720afeb60def3c4d`

| Source path | Git blob | SHA-256 |
|---|---|---|
| `src/confcurve/core.py` | `d28bc1962c9028eba94338f6738cd769800b14f2` | `346dd746be2c257dfc02f0822fa46e025c56bfc911733746c090e7df37d470a7` |
| `src/confcurve/design.py` | `44ba281e044a691ca514e668ca9cf65d3e7045a9` | `f20af34da0daebb7e7682eeaa8ef644d5a66082cf0f0d3826bf4260bef2cda1b` |
| `src/confcurve/models.py` | `1ee914d0931299d25083eda5b604bf9a0e08c8a8` | `33a06b6ab55fefddd86d44323b6bea192c7016566fc5c2220f6ed1fc577112fe` |
| `src/confcurve/web_contract.py` | `7a28582d25819324556a020b4e64998b252f6627` | `79fc4b9eead6d86dd330aeaf9e545f0f4ed87429ee6698e69e6cd0b658891aaf` |

This directory is intentionally temporary. The next commit refactors the imported
logic into the `wald_inference` package and deletes this snapshot so the final
repository has one implementation of each formula. Git history retains this exact
checkpoint and its source mapping.
