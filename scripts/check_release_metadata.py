#!/usr/bin/env python3
"""Fail closed when public release metadata disagrees."""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import tomllib
from datetime import date
from pathlib import Path

PROJECT_NAME = "wald-inference"
IMPORT_NAME = "wald_inference"
VERSION = "0.4.2"
AUTHOR = "Brian Locke"
LICENSE_EXPRESSION = "MIT"
REPOSITORY = "https://github.com/reblocke/wald-inference-core"


def _cff_scalar(text: str, key: str) -> str | None:
    match = re.search(
        rf"^[ \t]*(?:-[ \t]+)?{re.escape(key)}:[ \t]*"
        rf"(?:\"([^\"]*)\"|'([^']*)'|([^#\n]*?))[ \t]*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    return next(value for value in match.groups() if value is not None).strip()


def _fail_if(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        failures.append(message)


def check_metadata(root: Path, expected_tag: str | None) -> list[str]:
    failures: list[str] = []
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    _fail_if(
        project.get("name") != PROJECT_NAME, f"project.name must be {PROJECT_NAME!r}", failures
    )
    _fail_if(project.get("version") != VERSION, f"project.version must be {VERSION!r}", failures)
    _fail_if(
        project.get("authors") != [{"name": AUTHOR}],
        f"project.authors must contain only {AUTHOR!r}",
        failures,
    )
    _fail_if(
        project.get("maintainers") != [{"name": AUTHOR}],
        f"project.maintainers must contain only {AUTHOR!r}",
        failures,
    )
    _fail_if(
        project.get("license") != LICENSE_EXPRESSION,
        f"project.license must be {LICENSE_EXPRESSION!r}",
        failures,
    )
    _fail_if(
        project.get("license-files") != ["LICENSE"],
        "project.license-files must be ['LICENSE']",
        failures,
    )
    _fail_if(
        project.get("urls", {}).get("Repository") != REPOSITORY,
        f"project.urls.Repository must be {REPOSITORY!r}",
        failures,
    )

    cff_text = (root / "CITATION.cff").read_text(encoding="utf-8")
    expected_cff = {
        "given-names": "Brian",
        "family-names": "Locke",
        "version": VERSION,
        "license": LICENSE_EXPRESSION,
        "repository-code": REPOSITORY,
    }
    for key, expected in expected_cff.items():
        actual = _cff_scalar(cff_text, key)
        _fail_if(actual != expected, f"CITATION.cff {key} must be {expected!r}", failures)

    release_date = _cff_scalar(cff_text, "date-released")
    try:
        if release_date is None:
            raise ValueError("missing")
        date.fromisoformat(release_date)
    except ValueError:
        failures.append("CITATION.cff date-released must be an ISO calendar date")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    changelog_match = re.search(
        rf"^## \[{re.escape(VERSION)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$",
        changelog,
        flags=re.MULTILINE,
    )
    _fail_if(changelog_match is None, f"CHANGELOG.md must contain release {VERSION}", failures)
    if changelog_match is not None and release_date is not None:
        _fail_if(
            changelog_match.group(1) != release_date,
            "CITATION.cff and CHANGELOG.md release dates must agree",
            failures,
        )

    license_lines = (root / "LICENSE").read_text(encoding="utf-8").splitlines()
    _fail_if(
        len(license_lines) < 3 or license_lines[0] != "MIT License",
        "LICENSE must contain the standard MIT heading",
        failures,
    )
    _fail_if(
        len(license_lines) < 3 or license_lines[2] != "Copyright (c) 2026 Brian Locke",
        "LICENSE copyright line must name Brian Locke",
        failures,
    )

    readme = (root / "README.md").read_text(encoding="utf-8")
    for required in (
        "# Wald Inference Core",
        "`wald-inference`",
        "`wald_inference`",
        "Brian Locke",
        "MIT",
    ):
        _fail_if(required not in readme, f"README.md must contain {required!r}", failures)

    forbidden_identity = re.compile(r"Reed Blocke|Brian W\. Locke|Your Name")
    for path in ("pyproject.toml", "CITATION.cff", "LICENSE", "README.md"):
        text = (root / path).read_text(encoding="utf-8")
        _fail_if(
            forbidden_identity.search(text) is not None,
            f"{path} contains superseded or placeholder identity metadata",
            failures,
        )

    if expected_tag is not None:
        expected = f"v{VERSION}"
        _fail_if(expected_tag != expected, f"release tag must be {expected!r}", failures)

    try:
        module = importlib.import_module(IMPORT_NAME)
    except ImportError as exc:
        failures.append(f"cannot import {IMPORT_NAME}: {exc}")
    else:
        _fail_if(
            getattr(module, "__version__", None) != VERSION,
            f"{IMPORT_NAME}.__version__ must be {VERSION!r}",
            failures,
        )
        exported = getattr(module, "__all__", None)
        _fail_if(
            not isinstance(exported, (list, tuple)) or not exported,
            f"{IMPORT_NAME}.__all__ must be a nonempty deliberate sequence",
            failures,
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tag", help="Expected release tag, for example v0.4.2")
    args = parser.parse_args()

    failures = check_metadata(args.root.resolve(), args.tag)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    tag_suffix = f" and tag {args.tag}" if args.tag else ""
    print(f"Release metadata is consistent for {PROJECT_NAME} {VERSION}{tag_suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
