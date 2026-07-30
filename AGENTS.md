# Codex AGENTS

## Purpose

- This repository is the pure-Python numerical source of truth for the Wald-inference applet
  portfolio.
- The distribution is `wald-inference`; the import package is `wald_inference`.
- Optimize for scientific correctness, finite-value safety, reproducibility, and readable APIs.

## Repository map

- `src/wald_inference/` contains domain objects and numerical functions.
- `tests/` contains unit, property, scientific-reference, and baseline-regression tests.
- `scripts/verify_baseline_parity.py` compares core-owned outputs with the frozen source corpus.
- `scripts/` also contains release-metadata, distribution, reproducibility, and cold-install gates.
- `docs/` records scientific scope, API, validation, decisions, maintenance, privacy, and provenance.

## Commands

- Setup: `uv sync --locked --all-groups`
- Format: `make fmt`
- Format check: `make fmt-check`
- Lint: `make lint`
- Tests: `make test`
- Baseline parity: `make parity`
- Build: `make build`
- Cold-wheel smoke: `make smoke`
- Full local verification: `make verify`

## Authority

1. User request and approved scientific requirements.
2. The tagged source baseline and committed parity fixtures.
3. `README.md`, `docs/SCIENTIFIC_SCOPE.md`, `docs/VALIDATION.md`, and
   `docs/DECISIONS.md`.
4. Existing public API and tests.

## Working rules

- Before non-trivial edits, state assumptions, ambiguities, tradeoffs, silent-failure risks,
  success criteria, and verification commands.
- Make the smallest change that fully solves the request; avoid unrelated refactors.
- Do not change a formula, selection tail, tolerance, effect registry entry, or undefined-value
  convention without explicit approval and scientific-impact documentation.
- Keep browser payloads, DOM code, Plotly, export rendering, and app-specific wording out of this
  repository.
- Use NumPy and SciPy for numerical work, pytest and Hypothesis for tests, Ruff for formatting and
  linting, and `uv.lock` for the environment.
- Public functions must be typed, documented, exported deliberately, and covered by tests.
- A formula must have one implementation; compatibility aliases may delegate but must not fork it.
- Never widen parity tolerances merely to make a failure pass.
- Do not publish to PyPI without separate explicit authorization.

## Release rules

- Downstream applications consume exact released versions; they never pin an unreviewed branch.
- A numerical bug fix requires a patch release and a scientific-impact note.
- Release artifacts are built from an annotated tag, reproduced byte-for-byte, installed in a cold
  environment, checksummed, and attached to a GitHub release.
- Candidate GitHub releases may begin as prereleases. Promotion must reuse the same tag and
  artifacts without rebuilding.

## Done criteria

- Relevant unit, property, reference, and parity tests pass.
- Strict finite/undefined behavior is preserved.
- Package metadata, citation, changelog, docs, tag, and `__version__` agree.
- Wheel and sdist pass content inspection and cold-install smoke tests.
- The final report names commands, results, artifact hashes, and remaining risks.
