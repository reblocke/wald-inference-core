#!/usr/bin/env python3
"""Extract one immutable version section from CHANGELOG.md."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def extract_release_notes(changelog: str, version: str) -> str:
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\] - (?P<date>\d{{4}}-\d{{2}}-\d{{2}})\n"
        rf"(?P<body>.*?)(?=^## \[|^\[[^\]]+\]:|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(changelog))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one changelog section for {version}, found {len(matches)}"
        )
    match = matches[0]
    body = match.group("body").strip()
    if not body:
        raise ValueError(f"changelog section for {version} is empty")
    return f"# wald-inference {version}\n\nReleased {match.group('date')}.\n\n{body}\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        notes = extract_release_notes(
            args.changelog.read_text(encoding="utf-8"),
            args.version,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.output is None:
        print(notes, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(notes, encoding="utf-8", newline="\n")
        print(f"Wrote release notes to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
