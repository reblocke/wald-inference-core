#!/usr/bin/env python3
"""Build the same Git tree twice and emit a verified release bundle."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from check_distribution import SDIST_NAME, WHEEL_NAME, check_distribution


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _output(command: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def _archive_ref(root: Path, ref: str, destination: Path) -> None:
    destination.mkdir(parents=True)
    archive_path = destination.parent / f"{destination.name}.tar"
    _run(
        ["git", "archive", "--format=tar", "--output", str(archive_path), ref],
        cwd=root,
    )
    with tarfile.open(archive_path, mode="r:") as archive:
        archive.extractall(destination)


def _normalize_sdist(path: Path, *, source_date_epoch: int) -> None:
    """Rewrite an sdist with deterministic gzip and tar metadata."""

    normalized_path = path.with_name(f".{path.name}.normalized")
    try:
        with (
            tarfile.open(path, mode="r:gz") as source,
            normalized_path.open("wb") as raw_output,
            gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_output,
                mtime=source_date_epoch,
            ) as compressed_output,
            tarfile.open(
                fileobj=compressed_output,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as destination,
        ):
            for member in sorted(source.getmembers(), key=lambda item: item.name):
                normalized = copy.copy(member)
                normalized.uid = 0
                normalized.gid = 0
                normalized.uname = ""
                normalized.gname = ""
                normalized.mtime = source_date_epoch
                normalized.pax_headers = {}
                file_object = source.extractfile(member) if member.isfile() else None
                try:
                    destination.addfile(normalized, fileobj=file_object)
                finally:
                    if file_object is not None:
                        file_object.close()
        os.replace(normalized_path, path)
    finally:
        normalized_path.unlink(missing_ok=True)


def _build_copy(source: Path, environment: dict[str, str]) -> Path:
    _run(["uv", "sync", "--locked", "--all-groups"], cwd=source, env=environment)
    dist_dir = source / "dist"
    _run(
        [
            "uv",
            "build",
            "--no-build-isolation",
            "--out-dir",
            str(dist_dir),
        ],
        cwd=source,
        env=environment,
    )
    _normalize_sdist(
        dist_dir / SDIST_NAME,
        source_date_epoch=int(environment["SOURCE_DATE_EPOCH"]),
    )
    return dist_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_annotated_tag(root: Path, tag: str, commit: str) -> None:
    tag_ref = f"refs/tags/{tag}"
    object_type = _output(["git", "cat-file", "-t", tag_ref], cwd=root)
    if object_type != "tag":
        raise RuntimeError(f"{tag} must be an annotated tag, found object type {object_type!r}")
    tag_commit = _output(["git", "rev-parse", f"{tag_ref}^{{commit}}"], cwd=root)
    if tag_commit != commit:
        raise RuntimeError(f"{tag} resolves to {tag_commit}, expected {commit}")


def build_bundle(
    *,
    root: Path,
    ref: str,
    tag: str | None,
    parity_report: Path,
    out_dir: Path,
) -> None:
    dirty = _output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root)
    if dirty:
        raise RuntimeError("release builds require a clean Git worktree")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = _output(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=root)
    head = _output(["git", "rev-parse", "HEAD"], cwd=root)
    if commit != head:
        raise RuntimeError(f"release ref {commit} does not match checked-out HEAD {head}")
    if tag is not None:
        _require_annotated_tag(root, tag, commit)
        _run(
            [
                sys.executable,
                "scripts/check_release_metadata.py",
                "--tag",
                tag,
            ],
            cwd=root,
        )

    if not parity_report.is_file():
        raise RuntimeError(f"missing machine-readable parity report: {parity_report}")
    try:
        parity_payload = json.loads(parity_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid parity report: {exc}") from exc
    if not isinstance(parity_payload, dict):
        raise RuntimeError("parity report must contain a JSON object")

    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": _output(["git", "show", "-s", "--format=%ct", commit], cwd=root),
            "TZ": "UTC",
        }
    )

    with tempfile.TemporaryDirectory(prefix="wald-inference-release-") as temporary:
        temp_root = Path(temporary)
        source_a = temp_root / "source-a"
        source_b = temp_root / "source-b"
        _archive_ref(root, commit, source_a)
        _archive_ref(root, commit, source_b)
        dist_a = _build_copy(source_a, environment)
        dist_b = _build_copy(source_b, environment)

        for filename in (WHEEL_NAME, SDIST_NAME):
            artifact_a = dist_a / filename
            artifact_b = dist_b / filename
            if not artifact_a.is_file() or not artifact_b.is_file():
                raise RuntimeError(f"both builds must produce {filename}")
            if _sha256(artifact_a) != _sha256(artifact_b):
                raise RuntimeError(f"reproducible-build comparison failed for {filename}")
            shutil.copy2(artifact_a, out_dir / filename)

    shutil.copy2(parity_report, out_dir / "baseline-parity.json")
    failures = check_distribution(out_dir)
    if failures:
        raise RuntimeError("distribution inspection failed: " + "; ".join(failures))

    checksum_paths = [
        out_dir / WHEEL_NAME,
        out_dir / SDIST_NAME,
        out_dir / "baseline-parity.json",
    ]
    checksum_text = "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths)
    checksum_file = out_dir / "SHA256SUMS"
    checksum_file.write_text(checksum_text, encoding="utf-8", newline="\n")

    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", maxsplit=1)
        actual = _sha256(out_dir / filename)
        if actual != expected:
            raise RuntimeError(f"checksum verification failed for {filename}")

    print(f"Built reproducible release bundle from {commit}:")
    for path in (*checksum_paths, checksum_file):
        print(f"  {path.name}: {_sha256(path)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--tag")
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        build_bundle(
            root=args.root.resolve(),
            ref=args.ref,
            tag=args.tag,
            parity_report=args.parity_report.resolve(),
            out_dir=args.out_dir.resolve(),
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
