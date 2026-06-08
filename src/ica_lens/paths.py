from __future__ import annotations

from pathlib import Path


V6_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = V6_ROOT / "artifacts"
FETCHED_ARTIFACTS_DIR = ARTIFACTS_DIR / "fetched"
RESULTS_DIR = V6_ROOT / "results"
CONFIGS_DIR = V6_ROOT / "configs"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
CHECKSUMS_PATH = ARTIFACTS_DIR / "checksums.sha256"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return V6_ROOT / path
