from __future__ import annotations

import gzip
import importlib
import io
import struct
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
BUILD_RELEASE_ARTIFACTS = importlib.import_module("build_release_artifacts")
EXTRACT_RELEASE_NOTES = importlib.import_module("extract_release_notes")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_test_sdist(
    path: Path,
    *,
    timestamp: int,
    uid: int,
    owner: str,
) -> None:
    with (
        path.open("wb") as raw_output,
        gzip.GzipFile(
            filename=f"{owner}.tar",
            mode="wb",
            fileobj=raw_output,
            mtime=timestamp,
        ) as compressed_output,
        tarfile.open(
            fileobj=compressed_output,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive,
    ):
        root = tarfile.TarInfo("wald_inference-0.4.2")
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        root.mtime = timestamp
        root.uid = uid
        root.uname = owner
        archive.addfile(root)

        payload = b"reproducible source\n"
        source = tarfile.TarInfo("wald_inference-0.4.2/example.txt")
        source.mode = 0o644
        source.mtime = timestamp + 1
        source.uid = uid
        source.uname = owner
        source.size = len(payload)
        archive.addfile(source, io.BytesIO(payload))


def test_sdist_normalization_removes_build_time_and_owner_metadata(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    source_date_epoch = 1_700_000_000
    _write_test_sdist(first, timestamp=source_date_epoch + 10, uid=501, owner="first")
    _write_test_sdist(second, timestamp=source_date_epoch + 20, uid=502, owner="second")

    BUILD_RELEASE_ARTIFACTS._normalize_sdist(  # noqa: SLF001
        first,
        source_date_epoch=source_date_epoch,
    )
    BUILD_RELEASE_ARTIFACTS._normalize_sdist(  # noqa: SLF001
        second,
        source_date_epoch=source_date_epoch,
    )

    assert first.read_bytes() == second.read_bytes()
    assert struct.unpack("<I", first.read_bytes()[4:8])[0] == source_date_epoch

    with tarfile.open(first, mode="r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == [
            "wald_inference-0.4.2",
            "wald_inference-0.4.2/example.txt",
        ]
        assert all(member.mtime == source_date_epoch for member in members)
        assert all(member.uid == 0 and member.gid == 0 for member in members)
        assert all(member.uname == "" and member.gname == "" for member in members)
        assert archive.extractfile(members[1]).read() == b"reproducible source\n"


def test_release_notes_extract_only_the_requested_current_version() -> None:
    changelog = """\
# Changelog

## [Unreleased]

- Future work that must not appear.

## [0.4.2] - 2026-07-30

### Fixed

- Current release detail.

## [0.4.0] - 2026-07-29

- Historical detail that must not appear.

[Unreleased]: https://example.invalid/compare/v0.4.2...HEAD
"""

    notes = EXTRACT_RELEASE_NOTES.extract_release_notes(changelog, "0.4.2")

    assert notes == (
        "# wald-inference 0.4.2\n\nReleased 2026-07-30.\n\n### Fixed\n\n- Current release detail.\n"
    )
    assert "Future work" not in notes
    assert "Historical detail" not in notes
    assert "[Unreleased]" not in notes


def test_current_repository_release_notes_are_extractable() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    version = project["version"]

    notes = EXTRACT_RELEASE_NOTES.extract_release_notes(
        (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        version,
    )

    assert notes.startswith(f"# wald-inference {version}\n\nReleased ")
    assert "[Unreleased]" not in notes


@pytest.mark.parametrize(
    ("changelog", "message"),
    [
        ("# Changelog\n\n## [0.4.0] - 2026-07-29\n\n- Old.\n", "found 0"),
        (
            "# Changelog\n\n"
            "## [0.4.2] - 2026-07-30\n\n- First.\n\n"
            "## [0.4.2] - 2026-07-31\n\n- Duplicate.\n",
            "found 2",
        ),
        ("# Changelog\n\n## [0.4.2] - 2026-07-30\n\n", "is empty"),
    ],
)
def test_release_notes_reject_missing_duplicate_or_empty_sections(
    changelog: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EXTRACT_RELEASE_NOTES.extract_release_notes(changelog, "0.4.2")
