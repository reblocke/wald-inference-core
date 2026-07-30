# Frozen integrated-baseline corpus

`golden/` is an exact copy of the 51-file corpus directory from
`reblocke/conf_curve_likelihood` tag `pre-split-baseline-2026-07-29`.
Its 50 JSON artifacts retain their generated source metadata and hashes.

- Tag target: `5fd501dd947d9b951d736014cfc2b310efa5e7b0`
- Behavior source: `830756ecb11b4e8161f8dfe1fc75afc346ef4467`
- Manifest SHA-256:
  `f54bb2d8311788c07adcf23fc9f038e35702449e4a77a474abea9411246cabcc`
- Fixture-set SHA-256:
  `81c341b39e711caffc85a444f0c1e4bc1e2d00633474c82e720afeb60def3c4d`

These fixtures are generated characterization outputs, not independent
scientific reference truth. `scripts/verify_baseline_parity.py` classifies
every tested field as core-owned or deliberately app-owned and evaluates the
core-owned subset without importing `confcurve`.

The read-only generation and comparison commands in `golden/README.md` describe
the source application repository and are intentionally preserved byte for
byte; they are not commands available in this repository. Run `make parity`
here. App-only export-schema JSON is retained solely to preserve the exact
51-file directory and its recorded corpus hash; it is fixture provenance, not
browser runtime code shipped in the wheel.
