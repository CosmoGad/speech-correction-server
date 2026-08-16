"""Fail CI when common credential formats are committed, without printing them."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


PATTERNS = (
    r"AKIA[0-9A-Z]{16}",
    r"AIza[0-9A-Za-z_-]{35}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"ghp_[A-Za-z0-9]{30,}",
    r"sk-[A-Za-z0-9]{20,}",
    r"xox[baprs]-[A-Za-z0-9-]{20,}",
)
COMBINED = re.compile("(?:" + ")|(?:".join(PATTERNS) + ")", re.IGNORECASE)
ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace"
    )


def _tracked_worktree_matches() -> list[str]:
    matches: list[str] = []
    for relative in _git("ls-files").splitlines():
        path = ROOT / relative
        try:
            if path.is_file() and COMBINED.search(path.read_text(encoding="utf-8")):
                matches.append(f"worktree:{relative}")
        except UnicodeDecodeError:
            continue
    return matches


def _history_matches() -> list[str]:
    matches: list[str] = []
    for revision in _git("rev-list", "--all").splitlines():
        result = subprocess.run(
            ["git", "grep", "-Il", "-P", "-e", COMBINED.pattern, revision, "--"],
            cwd=ROOT, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError("Unable to inspect Git history for credentials")
        for relative in result.stdout.splitlines():
            matches.append(f"history:{revision[:12]}:{relative}")
    return sorted(set(matches))


def main() -> int:
    matches = _tracked_worktree_matches() + _history_matches()
    if not matches:
        print("Secret scan passed: no supported credential formats found.")
        return 0
    print("Secret scan failed. Rotate and remove each matching credential; values are never printed.")
    print("\n".join(matches))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
