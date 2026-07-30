.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  sync            Install the locked environment"
	@echo "  fmt             Format Python with Ruff"
	@echo "  fmt-check       Check Python formatting"
	@echo "  lint            Run Ruff lint checks"
	@echo "  metadata-check  Verify release metadata consistency"
	@echo "  test            Run the pytest suite"
	@echo "  parity          Verify the frozen numerical baseline"
	@echo "  build           Build and inspect the wheel and sdist"
	@echo "  smoke           Install the built wheel in a cold environment and smoke-test it"
	@echo "  verify          Run all local release gates"
	@echo "  clean           Remove generated local artifacts"

.PHONY: sync
sync:
	uv sync --locked --all-groups

.PHONY: fmt
fmt:
	uv run ruff format .

.PHONY: fmt-check
fmt-check:
	uv run ruff format --check .

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: metadata-check
metadata-check:
	uv run python scripts/check_release_metadata.py

.PHONY: test
test:
	uv run pytest -q

.PHONY: parity
parity:
	uv run python scripts/verify_baseline_parity.py \
		--json-output reports/baseline-parity.json

.PHONY: build
build:
	@test ! -e dist || { echo "dist already exists; run 'make clean' first" >&2; exit 1; }
	uv build --no-build-isolation --out-dir dist
	uv run python scripts/check_distribution.py --dist-dir dist

.PHONY: smoke
smoke:
	uv run python scripts/smoke_installed_package.py --dist-dir dist

.PHONY: verify
verify: fmt-check lint metadata-check test parity build smoke
	git diff --check

.PHONY: clean
clean:
	@rm -rf build dist reports .pytest_cache .ruff_cache
	@find src tests scripts -type d -name __pycache__ -prune -exec rm -rf {} +
