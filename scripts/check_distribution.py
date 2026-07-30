#!/usr/bin/env python3
"""Inspect release archives without installing them."""

from __future__ import annotations

import argparse
import email
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

DIST_STEM = "wald_inference-0.1.1"
WHEEL_NAME = f"{DIST_STEM}-py3-none-any.whl"
SDIST_NAME = f"{DIST_STEM}.tar.gz"
COPYRIGHT = "Copyright (c) 2026 Brian Locke"

REQUIRED_SDIST_FILES = {
    ".python-version",
    "AGENTS.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "Makefile",
    "README.md",
    "llms.txt",
    "pyproject.toml",
    "uv.lock",
    "docs/API.md",
    "docs/DECISIONS.md",
    "docs/MAINTENANCE.md",
    "docs/MIGRATION_PROVENANCE.md",
    "docs/PRIVACY.md",
    "docs/SCIENTIFIC_SCOPE.md",
    "docs/VALIDATION.md",
    "scripts/check_distribution.py",
    "scripts/check_release_metadata.py",
    "scripts/smoke_installed_package.py",
    "scripts/verify_baseline_parity.py",
}


def _archive_text(lines: bytes) -> list[str]:
    return lines.decode("utf-8").splitlines()


def check_distribution(dist_dir: Path) -> list[str]:
    failures: list[str] = []
    wheel = dist_dir / WHEEL_NAME
    sdist = dist_dir / SDIST_NAME
    if not wheel.is_file():
        failures.append(f"missing expected wheel {WHEEL_NAME}")
    if not sdist.is_file():
        failures.append(f"missing expected sdist {SDIST_NAME}")
    unexpected = sorted(
        path.name
        for path in dist_dir.iterdir()
        if path.is_file() and path.suffix in {".whl", ".gz"} and path not in {wheel, sdist}
    )
    if unexpected:
        failures.append(f"unexpected distribution archives: {unexpected}")
    if failures:
        return failures

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_metadata_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        license_names = [name for name in names if name.endswith(".dist-info/licenses/LICENSE")]
        if len(metadata_names) != 1:
            failures.append("wheel must contain exactly one METADATA file")
        else:
            metadata = email.message_from_bytes(archive.read(metadata_names[0]))
            expected_headers = {
                "Name": "wald-inference",
                "Version": "0.1.1",
                "Author": "Brian Locke",
                "Maintainer": "Brian Locke",
                "License-Expression": "MIT",
            }
            for key, expected in expected_headers.items():
                if metadata.get(key) != expected:
                    failures.append(
                        f"wheel METADATA {key} is {metadata.get(key)!r}, expected {expected!r}"
                    )
            if "LICENSE" not in metadata.get_all("License-File", []):
                failures.append("wheel METADATA must declare LICENSE")

        if len(wheel_metadata_names) != 1:
            failures.append("wheel must contain exactly one WHEEL metadata file")
        else:
            wheel_metadata = archive.read(wheel_metadata_names[0]).decode("utf-8")
            if "Tag: py3-none-any" not in wheel_metadata:
                failures.append("wheel must be tagged py3-none-any")

        if len(license_names) != 1:
            failures.append("wheel must contain exactly one license file")
        else:
            license_lines = _archive_text(archive.read(license_names[0]))
            if len(license_lines) < 3 or license_lines[2] != COPYRIGHT:
                failures.append("wheel license contains incorrect copyright metadata")

        if "wald_inference/py.typed" not in names:
            failures.append("wheel must contain wald_inference/py.typed")
        if not any(name == "wald_inference/__init__.py" for name in names):
            failures.append("wheel must contain wald_inference/__init__.py")
        native_suffixes = (".so", ".dylib", ".dll", ".pyd")
        if any(name.endswith(native_suffixes) for name in names):
            failures.append("wheel contains a native/compiled library")
        allowed_roots = {"wald_inference", f"{DIST_STEM}.dist-info"}
        unexpected_roots = {
            PurePosixPath(name).parts[0]
            for name in names
            if name and PurePosixPath(name).parts[0] not in allowed_roots
        }
        if unexpected_roots:
            failures.append(f"wheel contains unexpected roots: {sorted(unexpected_roots)}")

    with tarfile.open(sdist, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        names = [PurePosixPath(member.name) for member in members]
        roots = {name.parts[0] for name in names if name.parts}
        if roots != {DIST_STEM}:
            failures.append(f"sdist root must be {DIST_STEM!r}, found {sorted(roots)}")
        relative_names = {
            str(PurePosixPath(*name.parts[1:])) for name in names if len(name.parts) > 1
        }
        missing = sorted(REQUIRED_SDIST_FILES - relative_names)
        if missing:
            failures.append(f"sdist is missing required files: {missing}")
        if any("provenance-import" in name.parts for name in names):
            failures.append("sdist contains the temporary provenance-import snapshot")
        if any("web" in name.parts for name in names):
            failures.append("sdist contains browser/web content")

        license_member = next(
            (member for member in members if PurePosixPath(member.name).name == "LICENSE"),
            None,
        )
        if license_member is None:
            failures.append("sdist does not contain LICENSE")
        else:
            extracted = archive.extractfile(license_member)
            if extracted is None:
                failures.append("could not read sdist LICENSE")
            else:
                license_lines = _archive_text(extracted.read())
                if len(license_lines) < 3 or license_lines[2] != COPYRIGHT:
                    failures.append("sdist license contains incorrect copyright metadata")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()
    dist_dir = args.dist_dir.resolve()
    if not dist_dir.is_dir():
        print(f"ERROR: distribution directory does not exist: {dist_dir}", file=sys.stderr)
        return 1

    failures = check_distribution(dist_dir)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print(f"Distribution contents passed inspection: {WHEEL_NAME}, {SDIST_NAME}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
