"""Run Tinker evaluation in a dependency-isolated environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from icalens.cli._status import log

TINKER_VERSION = "0.25.0"
TINKER_COOKBOOK_VERSION = "0.5.5"
_ISOLATED = "ICALENS_TINKER_ISOLATED"


def enter_isolated_tinker_environment(argv: list[str]) -> None:
    """Re-execute the current command in Tinker's pinned environment once."""
    if os.environ.get(_ISOLATED) == "1":
        return
    root = (
        Path.home()
        / ".cache"
        / "icalens"
        / "environments"
        / (f"tinker-{TINKER_VERSION}-cookbook-{TINKER_COOKBOOK_VERSION}")
    )
    python = root / "bin" / "python"
    if not _ready(python):
        log(f"Preparing isolated Tinker environment at {root}...")
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "uv",
                "venv",
                "--clear",
                "--system-site-packages",
                "--python",
                sys.executable,
                str(root),
            ],
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                f"tinker=={TINKER_VERSION}",
                f"tinker-cookbook=={TINKER_COOKBOOK_VERSION}",
                "gb10-load-llm>=0.1.2",
                "matplotlib>=3.9",
                "python-dotenv>=1.0",
            ],
            check=True,
        )
    environment = dict(os.environ)
    environment[_ISOLATED] = "1"
    source = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [str(python), "-m", "icalens.cli", *argv], env=environment, check=False
    )
    raise SystemExit(completed.returncode)


def _ready(python: Path) -> bool:
    if not python.is_file():
        return False
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import dotenv, gb10_load_llm, importlib.metadata, matplotlib; "
                "import tinker, tinker_cookbook, transformers; "
                f"assert importlib.metadata.version('tinker') == '{TINKER_VERSION}'; "
                "assert tuple(map(int, transformers.__version__.split('.')[:2])) <= (5, 5)"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0
