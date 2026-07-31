from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = PROJECT_ROOT / ".github" / "workflows"
GH_CLI_VERSION = "2.93.0"
GH_CLI_LINUX_AMD64_SHA256 = "02d1290eba130e0b896f3709ffff22e1c75a51475ddb70476a85abc6b5807af0"

EXPECTED_ACTIONS = {
    "actions/attest": (
        "508db95dd578ae2727ebd6217d5ba78e4fbda05d",
        "4.2.1",
    ),
    "actions/checkout": (
        "d23441a48e516b6c34aea4fa41551a30e30af803",
        "6.1.0",
    ),
    "actions/download-artifact": (
        "37930b1c2abaa49bbe596cd826c3c89aef350131",
        "7.0.0",
    ),
    "actions/setup-python": (
        "ece7cb06caefa5fff74198d8649806c4678c61a1",
        "6.3.0",
    ),
    "actions/upload-artifact": (
        "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "7.0.1",
    ),
    "astral-sh/setup-uv": (
        "37802adc94f370d6bfd71619e3f0bf239e1f3b78",
        "7.6.0",
    ),
}
EXTERNAL_ACTION = re.compile(
    r"^\s*(?:-\s+)?uses:\s*"
    r"(?P<action>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)?)"
    r"@(?P<sha>[0-9a-f]{40})\s+#\s+"
    r"v(?P<version>\d+\.\d+\.\d+)\s*$"
)


def _workflow_paths() -> list[Path]:
    return sorted({*WORKFLOW_ROOT.glob("*.yml"), *WORKFLOW_ROOT.glob("*.yaml")})


def _job_block(workflow: str, job: str, next_job: str | None = None) -> str:
    start = workflow.index(f"  {job}:")
    end = len(workflow) if next_job is None else workflow.index(f"\n  {next_job}:", start)
    return workflow[start:end]


def test_all_external_actions_use_reviewed_full_shas_and_exact_version_comments() -> None:
    observed_versions: dict[str, set[str]] = defaultdict(set)
    workflows = _workflow_paths()

    assert workflows
    for path in workflows:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "uses:" not in line:
                continue
            reference = line.split("uses:", maxsplit=1)[1].strip()
            if reference.startswith("./"):
                continue
            match = EXTERNAL_ACTION.fullmatch(line)
            assert match is not None, f"{path}:{line_number}: unpinned external action: {line}"
            action = match.group("action")
            sha = match.group("sha")
            version = match.group("version")
            assert action in EXPECTED_ACTIONS, f"review new external action {action}"
            assert (sha, version) == EXPECTED_ACTIONS[action], (
                f"{path}:{line_number}: review the new {action} pin and exact version"
            )
            observed_versions[action].add(version)

    assert set(observed_versions) == set(EXPECTED_ACTIONS)
    assert all(len(versions) == 1 for versions in observed_versions.values())


def test_workflow_permissions_are_narrow_and_job_scoped() -> None:
    ci = (WORKFLOW_ROOT / "ci.yml").read_text(encoding="utf-8")
    release = (WORKFLOW_ROOT / "release.yml").read_text(encoding="utf-8")
    verify_build = _job_block(release, "verify-build", "attest")
    attest = _job_block(release, "attest", "release")
    publish = _job_block(release, "release")

    assert "permissions:\n  contents: read" in ci
    assert "contents: write" not in ci
    assert "\npermissions: {}\n" in release
    assert "permissions:\n      contents: read" in verify_build
    assert "contents: write" not in verify_build
    assert "id-token: write" not in verify_build
    assert "attestations: write" not in verify_build
    assert "enable-cache: true" not in verify_build
    assert "enable-cache: false" in verify_build
    assert (
        "permissions:\n      contents: read\n"
        "      id-token: write # Authenticate artifact provenance attestations.\n"
        "      attestations: write # Publish provenance attestations for release artifacts."
        in (attest)
    )
    assert "contents: write" not in attest
    assert (
        "permissions:\n"
        "      contents: write # Create and publish the verified GitHub release." in publish
    )
    assert "id-token: write" not in publish
    assert "attestations: write" not in publish
    assert release.count("contents: write") == 1
    assert release.count("id-token: write") == 1
    assert release.count("attestations: write") == 1


def test_release_requires_verified_signed_tag_before_repository_code() -> None:
    release = (WORKFLOW_ROOT / "release.yml").read_text(encoding="utf-8")

    version_parse = (
        "python -I -c 'import tomllib; "
        'print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])\''
    )
    assert version_parse in release
    assert 'test "$GITHUB_REF_NAME" = "v${project_version}"' in release
    assert 'git cat-file -t "$GITHUB_REF_NAME"' in release
    assert 'git rev-parse "$GITHUB_REF_NAME^{commit}"' in release
    assert "/git/ref/tags/${GITHUB_REF_NAME}" in release
    assert 'git rev-parse "refs/tags/$GITHUB_REF_NAME"' in release
    assert "--jq '.tag'" in release
    assert ".verification.verified" in release
    assert ".verification.reason" in release
    assert ')" = "valid"' in release
    assert "--jq '.object.type'" in release
    assert ')" = "commit"' in release
    assert '"https://github.com/${GITHUB_REPOSITORY}.git"' in release
    assert "+refs/heads/main:refs/remotes/origin/main" in release
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" refs/remotes/origin/main' in release
    assert release.index(".verification.verified") < release.index("git fetch")
    assert release.index("git merge-base --is-ancestor") < release.index(version_parse)
    assert release.index(".verification.verified") < release.index(version_parse)
    assert release.index(".verification.verified") < release.index("uv sync --locked")
    assert release.index(".verification.verified") < release.index(
        "scripts/check_release_metadata.py"
    )


def test_checkout_credentials_are_not_persisted() -> None:
    workflow_text = "\n".join(path.read_text(encoding="utf-8") for path in _workflow_paths())

    checkout_count = workflow_text.count("uses: actions/checkout@")
    assert checkout_count > 0
    assert workflow_text.count("persist-credentials: false") == checkout_count


def test_release_uses_checksummed_patched_github_cli_before_credentialed_commands() -> None:
    release = (WORKFLOW_ROOT / "release.yml").read_text(encoding="utf-8")
    verify_build = _job_block(release, "verify-build", "attest")
    publish = _job_block(release, "release")

    assert f'GH_CLI_VERSION: "{GH_CLI_VERSION}"' in release
    assert f'GH_CLI_LINUX_AMD64_SHA256: "{GH_CLI_LINUX_AMD64_SHA256}"' in release
    assert release.count("Install checksummed GitHub CLI") == 2
    assert release.count("sha256sum --check --strict -") == 2
    assert release.count("Confirm the checksummed GitHub CLI is selected") == 2
    assert verify_build.index("Install checksummed GitHub CLI") < verify_build.index("gh api")
    assert publish.index("Install checksummed GitHub CLI") < publish.index("gh api")
    assert publish.index("Confirm the checksummed GitHub CLI is selected") < publish.index(
        "gh release create"
    )
    assert publish.index("Confirm the checksummed GitHub CLI is selected") < publish.index(
        "gh release verify"
    )


def test_release_is_build_once_attested_draft_first_and_immutable() -> None:
    release = (WORKFLOW_ROOT / "release.yml").read_text(encoding="utf-8")
    publish = _job_block(release, "release")

    for required_asset in [
        'wald_inference-"$version"-py3-none-any.whl',
        'wald_inference-"$version".tar.gz',
        "SHA256SUMS",
        "baseline-parity.json",
    ]:
        assert required_asset in publish

    assert release.count("scripts/build_release_artifacts.py") == 1
    assert "actions/attest@" in release
    assert "subject-path: release-bundle/assets/*.whl" in release
    assert "subject-path: release-bundle/assets/*.tar.gz" in release
    assert release.count("find release-bundle/assets -maxdepth 1 -type f | wc -l") >= 2
    assert release.count('"repos/${GITHUB_REPOSITORY}/immutable-releases"') == 2
    assert release.count("X-GitHub-Api-Version: 2026-03-10") >= 2
    assert release.count("secrets.RELEASE_SETTINGS_READ_TOKEN") == 2
    assert "--draft" in publish
    assert "--verify-tag" in publish
    assert "--prerelease" not in release
    assert "--notes-file release-bundle/release-notes.md" in publish
    assert "--notes-file CHANGELOG.md" not in release
    assert "--json tagName" in publish
    assert "--json name" in publish
    assert "--json body" in publish
    assert "gh release download" in publish
    assert "diff --recursive --brief" in publish
    assert "sha256sum -c SHA256SUMS" in publish
    assert "--draft=false" in publish
    assert "--json isImmutable" in publish
    assert "gh release verify" in publish
    assert "gh release verify-asset" in publish
    assert (
        publish.index("/immutable-releases")
        < publish.index("gh release create")
        < publish.index("gh release download")
        < publish.rindex("/immutable-releases")
        < publish.index("--draft=false")
    )

    for forbidden_build in ["uv build", "make build", "build_release_artifacts.py"]:
        assert forbidden_build not in publish


def test_release_body_comparison_rejects_a_trailing_newline_mismatch(
    tmp_path: Path,
) -> None:
    release = (WORKFLOW_ROOT / "release.yml").read_text(encoding="utf-8")
    publish = _job_block(release, "release")
    expected = tmp_path / "expected.md"
    hosted = tmp_path / "hosted.md"
    expected.write_bytes(b"# Notes\n")
    hosted.write_bytes(b"# Notes\n\n")

    comparison = subprocess.run(
        ["cmp", "--silent", expected, hosted],
        check=False,
    )

    assert comparison.returncode != 0
    assert "jq --exit-status --join-output '.body'" in publish
    assert "cmp --silent" in publish
    assert "release-bundle/release-notes.md" in publish
    assert '"$RUNNER_TEMP/release-body.md"' in publish
    assert 'test "$(cat release-bundle/release-notes.md)"' not in publish


def test_workflows_have_no_pypi_publication_path() -> None:
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8") for path in _workflow_paths()
    ).lower()

    for forbidden in [
        "pypa/gh-action-pypi-publish",
        "pypi.org",
        "token.pypi",
        "twine upload",
        "uv publish",
        "hatch publish",
        "flit publish",
    ]:
        assert forbidden not in workflow_text


def test_dependabot_covers_uv_and_actions_without_major_action_updates_or_automerge() -> None:
    dependabot = (PROJECT_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")

    assert 'package-ecosystem: "uv"' in dependabot
    assert 'package-ecosystem: "github-actions"' in dependabot
    assert dependabot.count('interval: "weekly"') == 2
    assert dependabot.count("default-days: 7") == 2
    assert '"version-update:semver-major"' in dependabot
    assert "automerge" not in dependabot.lower()


def test_public_coordination_files_require_private_security_and_synthetic_inputs() -> None:
    security = (PROJECT_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    contributing = (PROJECT_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    governance = (PROJECT_ROOT / "docs" / "GOVERNANCE.md").read_text(encoding="utf-8")
    privacy = (PROJECT_ROOT / "docs" / "PRIVACY.md").read_text(encoding="utf-8")
    issue_config = (PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml").read_text(
        encoding="utf-8"
    )
    issue_forms = [
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml"))
        if path.name != "config.yml"
    ]
    pull_request = (PROJECT_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(
        encoding="utf-8"
    )
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    public_policy = "\n".join([security, contributing, governance, privacy, *issue_forms]).lower()

    assert "/security/advisories/new" in security
    assert "private vulnerability reporting is a github repository setting" in security.lower()
    assert "do not disclose vulnerability details in a public issue" in security.lower()
    assert "blank_issues_enabled: false" in issue_config
    assert "/security/advisories/new" in issue_config
    assert "synthetic" in public_policy
    assert "protected health information" in public_policy
    assert "pypi publishing is prohibited" in governance.lower()
    assert "immutable releases" in governance.lower()
    assert "make verify" in contributing
    assert "make verify" in pull_request
    assert "[Governance](docs/GOVERNANCE.md)" in readme
    assert "[Security policy](SECURITY.md)" in readme
    assert "[Contributing](CONTRIBUTING.md)" in readme
    assert all("labels:" not in form for form in issue_forms)
