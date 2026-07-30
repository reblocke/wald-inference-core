# Frozen integrated baseline

This directory is the machine-readable behavior record for
`reblocke/conf_curve_likelihood` at source commit
`830756ecb11b4e8161f8dfe1fc75afc346ef4467`.

The corpus represents validation-matrix case families B01–B08. B01–B06 store complete
browser responses at 401 grid points. B07 and B08 are split into focused
subcases and store either exact validation errors or compact edge summaries;
large arrays are represented by their length, endpoints, null counts, and
finite-value status when full arrays add no migration signal.

The corpus includes B08e, which freezes the approved finite-range safety behavior:
an unrepresentable standardized design distance raises `ValidationError` before
the browser response is serialized. Every successful stored response is strict
JSON, and expected-error fixtures contain no non-standard numeric tokens.

Files under `requests/`, `responses/`, and `export_schemas/` are generated
artifacts. `manifest.json` records their source commit, dependency versions,
the declared Python version and actual Python patch version, comparison policy,
individual SHA-256 hashes, and a corpus hash. The
combined fixture hash is computed over canonical request and expected-result
content; the manifest is not self-hashed.

The runtime patch is generation provenance. Read-only checks require the active
interpreter to remain within the declared Python 3.11 series and require exact
locked package versions; they do not falsely require every CI runner to use the
same 3.11 patch release. `export_schemas/effect_registry.json` freezes every
effect key, specification, and default null exactly.

## Read-only checks

```bash
uv run python scripts/generate_golden_baseline.py --check
uv run python scripts/compare_golden_baseline.py
```

The first command checks stored hashes, case definitions, schemas, dependency
versions, and current behavior. The second exposes the same behavioral
comparison directly. Responses are compared recursively with readable JSON
paths, exact comparison for non-floats, and `rtol=1e-12`, `atol=1e-14` for
scientific floats and arrays. Identity/configuration float paths listed in the
manifest, including the effect-registry default null, are exact. The checks
also enforce canonical strict JSON, unexpected-key rejection, and
contractually significant key order for every repeated response row.

Tests never regenerate expected files.

## Intentional regeneration

Regeneration is not a routine fix for a failing test. It is permitted only
when the recorded source is still the frozen commit and production Python/web
files or dependency/runtime authority files (`pyproject.toml`, `uv.lock`, and
`.python-version`) have not diverged:

```bash
uv run python scripts/generate_golden_baseline.py --write --force
```

The writer refuses an existing corpus without `--force` and refuses a dirty
worktree unless the milestone implementer also supplies the explicit
`--allow-dirty` acknowledgment. Review every resulting diff. A later
intentional scientific or contract change should create a new versioned
baseline rather than overwrite this one.
