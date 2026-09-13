"""Where the manuscript repo lives.

Several emitters here write tables and figures straight into the sibling manuscript repo.
That path used to be hardcoded relative to this checkout, which silently targets the MAIN
manuscript checkout even when the caller is working in a git worktree -- so a worktree
session would regenerate files it cannot see. `AUDIT_MS_REPO` overrides it; the default is
the previous behaviour, so nothing changes unless the variable is set.

    export AUDIT_MS_REPO=.../2026-immrep25-audit-ms/.claude/worktrees/<name>
"""
from __future__ import annotations
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MS_REPO = os.environ.get(
    "AUDIT_MS_REPO", os.path.join(os.path.dirname(REPO), "2026-immrep25-audit-ms")
)

TABLES = os.path.join(MS_REPO, "tables")
FIGURES = os.path.join(MS_REPO, "figures")


def ms_path(*parts: str) -> str:
    """Join a path inside the manuscript repo, creating the parent directory."""
    p = os.path.join(MS_REPO, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def demo():
    assert os.path.isabs(MS_REPO) and MS_REPO.endswith("2026-immrep25-audit-ms") or \
        os.environ.get("AUDIT_MS_REPO"), MS_REPO
    assert TABLES.endswith(os.path.join("2026-immrep25-audit-ms", "tables")) or \
        os.environ.get("AUDIT_MS_REPO")
    print("paths.demo OK  MS_REPO=%s  (override with AUDIT_MS_REPO)" % MS_REPO)


if __name__ == "__main__":
    demo()
