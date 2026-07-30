from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"
MANIFEST_PATH = GOLDEN_ROOT / "manifest.json"

EXPECTED_MANIFEST_SHA256 = "f54bb2d8311788c07adcf23fc9f038e35702449e4a77a474abea9411246cabcc"
EXPECTED_FIXTURE_SET_SHA256 = "81c341b39e711caffc85a444f0c1e4bc1e2d00633474c82e720afeb60def3c4d"
EXPECTED_BEHAVIOR_SOURCE = "830756ecb11b4e8161f8dfe1fc75afc346ef4467"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def test_frozen_corpus_identity_and_every_stored_hash() -> None:
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    assert _sha256_text(manifest_text) == EXPECTED_MANIFEST_SHA256
    manifest = json.loads(manifest_text)
    assert manifest_text == _canonical_json(manifest)
    assert manifest["source"]["commit"] == EXPECTED_BEHAVIOR_SOURCE
    assert manifest["case_count"] == 22
    assert len(manifest["cases"]) == 22
    assert manifest["fixture_set_sha256"] == EXPECTED_FIXTURE_SET_SHA256

    fixture_hash_inputs: list[str] = []
    for case in manifest["cases"]:
        request_text = (GOLDEN_ROOT / case["request_file"]).read_text(encoding="utf-8")
        expected_text = (GOLDEN_ROOT / case["expected_file"]).read_text(encoding="utf-8")
        request = json.loads(request_text)
        expected = json.loads(expected_text)

        assert request_text == _canonical_json(request), case["id"]
        assert expected_text == _canonical_json(expected), case["id"]
        assert _sha256_text(request_text) == case["request_sha256"], case["id"]
        assert _sha256_text(expected_text) == case["expected_sha256"], case["id"]

        fixture_sha = _sha256_text(_canonical_json({"request": request, "expected": expected}))
        assert fixture_sha == case["fixture_sha256"], case["id"]
        fixture_hash_inputs.append(f"{case['id']}:{fixture_sha}")

    for schema in manifest["export_schemas"]:
        schema_text = (GOLDEN_ROOT / schema["path"]).read_text(encoding="utf-8")
        assert schema_text == _canonical_json(json.loads(schema_text)), schema["path"]
        schema_sha = _sha256_text(schema_text)
        assert schema_sha == schema["sha256"], schema["path"]
        fixture_hash_inputs.append(f"{schema['path']}:{schema_sha}")

    observed_set_sha = _sha256_text("\n".join(fixture_hash_inputs) + "\n")
    assert observed_set_sha == EXPECTED_FIXTURE_SET_SHA256
    assert len(list(GOLDEN_ROOT.rglob("*.json"))) == 50


def test_core_and_parity_code_never_import_the_legacy_package() -> None:
    import_pattern = re.compile(r"^\s*(?:from|import)\s+confcurve\b", re.MULTILINE)
    searched_roots = (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", PROJECT_ROOT / "tests")
    offenders: list[str] = []
    for root in searched_roots:
        for path in root.rglob("*.py"):
            if import_pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert offenders == []


def test_distribution_source_contains_no_browser_contract_or_runtime() -> None:
    source_root = PROJECT_ROOT / "src" / "wald_inference"
    forbidden = ("CurveRequest", "CurveResponse", "Plotly", "pyodide", "document.querySelector")
    offenders: list[str] = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(term in text for term in forbidden):
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert offenders == []
