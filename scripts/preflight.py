#!/usr/bin/env python3
"""Cheap release checks that run locally and in CI.

Catches the mistakes that are annoying (or dangerous) to discover after a public
push: version drift, missing release files, accidental databases, personal
paths, secret-looking strings, and oversized assets.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", "build", "dist", "htmlcov", ".tox", ".nox",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".whl", ".gz", ".zip"}
TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".yml", ".yaml", ".json", ".html", ".css", ".js",
    ".sh", ".txt", ".ini", ".cfg", ".in", ".gitignore", ".gitattributes", ".editorconfig",
}

REQUIRED = [
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "RELEASING.md",
    "pyproject.toml",
    "embd/__init__.py",
    "embd/static/index.html",
    "docs/dashboard.png",
    "docs/social-preview.png",
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/pull_request_template.md",
    ".github/dependabot.yml",
    "scripts/preflight.py",
]

FORBIDDEN_TRACKED = re.compile(
    r"(^|/)(\.env(?!\.example$)(\..*)?|\.DS_Store|.*\.(db|db-wal|db-shm|sqlite|sqlite3|pem|key|p12|pfx)|"
    r".*\.pyc|__pycache__/|\.embd/)"
)
FORBIDDEN_WORKTREE = re.compile(
    r"(^|/)(\.env(?!\.example$)(\..*)?|\.DS_Store|.*\.(db|db-wal|db-shm|sqlite|sqlite3|pem|key|p12|pfx))$"
)

SECRET_PATTERNS = [
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
    (
        "long secret assignment",
        re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"'][A-Za-z0-9_./+-]{20,}[\"']"),
    ),
]
PERSONAL_PATHS = [
    re.compile(r"/Users/(?!you(?:/|\b)|shared(?:/|\b))[A-Za-z0-9._-]+"),
    re.compile(r"/home/(?!you(?:/|\b)|shared(?:/|\b))[A-Za-z0-9._-]+"),
    re.compile(r"[A-Za-z]:\\Users\\(?!you(?:\\|\b)|shared(?:\\|\b))[A-Za-z0-9._-]+"),
]


def iter_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        current = Path(dirpath)
        if current != ROOT and current.name in SKIP_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for name in filenames:
            path = current / name
            if path.suffix in SKIP_SUFFIXES:
                continue
            files.append(path)
    return files


def is_text(path: Path) -> bool:
    if path.suffix in TEXT_SUFFIXES:
        return True
    if path.name in {".gitignore", ".gitattributes", ".editorconfig", "Makefile", "LICENSE", "NOTICE"}:
        return True
    if path.name == "embd":
        return True
    return False


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    checked = 0

    print(f"embd preflight · {ROOT}")

    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            errors.append(f"missing required file: {relative}")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    init = (ROOT / "embd" / "__init__.py").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    pyproject_version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    if not pyproject_version or not init_version:
        errors.append("could not parse version from pyproject.toml or embd/__init__.py")
    else:
        if pyproject_version.group(1) != init_version.group(1):
            errors.append(
                f"version mismatch: pyproject={pyproject_version.group(1)} "
                f"__init__={init_version.group(1)}"
            )
        else:
            version = pyproject_version.group(1)
            if f"## [{version}]" not in changelog:
                errors.append(f"CHANGELOG.md has no '## [{version}]' section")
            readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
            if f"**Status:** v{version}" not in readme_text:
                errors.append(f"README.md Status line does not mention v{version}")
            if not errors:
                print(f"  ✓ version {version} consistent")

    for path in iter_files():
        checked += 1
        relative = path.relative_to(ROOT).as_posix()
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"cannot stat {relative}: {exc}")
            continue
        if FORBIDDEN_WORKTREE.search(relative):
            errors.append(f"forbidden local/generated file in working tree: {relative}")
        if size > 5 * 1024 * 1024:
            errors.append(f"file larger than 5 MiB: {relative} ({size} bytes)")
        if not is_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"text file is not valid UTF-8: {relative}")
            continue
        if "\r\n" in text:
            errors.append(f"CRLF line endings: {relative}")
        for label, pattern in SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                errors.append(f"{label} in {relative}: {match.group(0)[:24]}…")
        for pattern in PERSONAL_PATHS:
            match = pattern.search(text)
            if match:
                errors.append(f"personal path in {relative}: {match.group(0)}")

    # GitHub's web upload does not preserve the executable bit. The docs and
    # Makefile use `python3 -m embd` / `bash scripts/demo.sh`, so mode loss is a
    # warning rather than a failed release.
    if not os.access(ROOT / "bin" / "embd", os.X_OK):
        warnings.append("bin/embd is not executable (run it with `bash bin/embd` or use `python3 -m embd`)")
    if not os.access(ROOT / "scripts" / "demo.sh", os.X_OK):
        warnings.append("scripts/demo.sh is not executable (run it with `bash scripts/demo.sh`)")

    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8").split("\0")
    except (OSError, subprocess.CalledProcessError):
        tracked = []
        warnings.append("git ls-files unavailable; skipped tracked-file audit")
    for relative in filter(None, tracked):
        if FORBIDDEN_TRACKED.search(relative):
            errors.append(f"forbidden file is tracked by git: {relative}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "YOUR_GITHUB_USER" in readme:
        warnings.append("README clone URL still contains YOUR_GITHUB_USER (replace after creating the repo)")
    if "OWNER/REPO" in readme:
        warnings.append("README still contains OWNER/REPO placeholders")

    print(f"  ✓ scanned {checked} files")
    for warning in warnings:
        print(f"  ! {warning}")
    if errors:
        print("\npreflight failed:")
        for error in errors:
            print(f"  ✗ {error}")
        return 1
    print("preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
