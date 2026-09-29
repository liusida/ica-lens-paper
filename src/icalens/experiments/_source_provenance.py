"""Record the local ICALens source state used for an experiment."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from icalens.cli._status import log


def source_provenance() -> dict[str, Any]:
    """Return Git provenance when this package is running from a checkout."""
    package_dir = Path(__file__).resolve().parents[1]
    root = _git(package_dir, "rev-parse", "--show-toplevel")
    if root is None:
        return {"git_available": False}

    root_path = Path(root)
    commit = _git(root_path, "rev-parse", "HEAD")
    status = _git(root_path, "status", "--porcelain", "--untracked-files=normal")
    if commit is None or status is None:
        return {"git_available": False}
    return {
        "git_available": True,
        "commit": commit,
        "dirty": bool(status),
    }


def warn_if_dirty(provenance: dict[str, Any]) -> None:
    """Warn without blocking an experiment started from uncommitted source."""
    if provenance.get("dirty"):
        log(
            "WARNING: The ICALens Git worktree has uncommitted changes. "
            "This run will record dirty=true, but the exact source cannot be "
            "reproduced from the recorded commit alone."
        )


def _git(directory: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
