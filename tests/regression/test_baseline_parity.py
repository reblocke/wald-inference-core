from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_core_owned_values_match_without_legacy_imports(tmp_path: Path) -> None:
    report_path = tmp_path / "baseline-parity.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_baseline_parity.py",
            "--json-output",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "pass"
    assert report["summary"]["successful_cases"] == 14
    assert report["summary"]["matched_core_error_cases"] == 6
    assert report["summary"]["app_only_error_exclusions"] == 2
    assert report["summary"]["compared_values"] > 0
