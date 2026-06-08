from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_json
from .paths import CHECKSUMS_PATH, MANIFEST_PATH, resolve_repo_path


@dataclass(frozen=True)
class ArtifactSet:
    name: str
    kind: str
    dataset_id: str
    revision: str
    local_dir: Path
    description: str
    allow_patterns: tuple[str, ...] = ()

    @property
    def is_configured(self) -> bool:
        return bool(self.dataset_id) and not self.dataset_id.startswith("TODO_")


def load_artifact_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return load_json(path)


def artifact_sets(path: Path = MANIFEST_PATH) -> dict[str, ArtifactSet]:
    manifest = load_artifact_manifest(path)
    raw_sets = manifest.get("artifact_sets")
    if not isinstance(raw_sets, dict):
        raise ValueError(f"{path} is missing an artifact_sets object.")
    parsed: dict[str, ArtifactSet] = {}
    for name, raw in raw_sets.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Artifact set {name!r} must be an object.")
        parsed[name] = ArtifactSet(
            name=str(name),
            kind=str(raw.get("kind", "")),
            dataset_id=str(raw.get("dataset_id", "")),
            revision=str(raw.get("revision", "main")),
            local_dir=resolve_repo_path(str(raw.get("local_dir", f"artifacts/fetched/{name}"))),
            description=str(raw.get("description", "")),
            allow_patterns=tuple(str(pattern) for pattern in raw.get("allow_patterns", [])),
        )
    return parsed


def selected_artifact_sets(*, models: bool, databases: bool) -> list[ArtifactSet]:
    sets = artifact_sets()
    selected: list[ArtifactSet] = []
    if models:
        selected.append(sets["models"])
    if databases:
        selected.append(sets["databases"])
    if not selected:
        selected = [sets["models"], sets["databases"]]
    return selected


def fetch_artifact_set(artifact_set: ArtifactSet, *, dry_run: bool = False) -> dict[str, Any]:
    if artifact_set.kind != "huggingface_dataset":
        raise ValueError(f"Unsupported artifact kind for {artifact_set.name}: {artifact_set.kind!r}")
    if not artifact_set.is_configured:
        return {
            "name": artifact_set.name,
            "status": "not_configured",
            "message": f"Set artifacts.manifest.json dataset_id for {artifact_set.name!r}.",
            "local_dir": str(artifact_set.local_dir),
        }
    if dry_run:
        return {
            "name": artifact_set.name,
            "status": "dry_run",
            "dataset_id": artifact_set.dataset_id,
            "revision": artifact_set.revision,
            "local_dir": str(artifact_set.local_dir),
            "allow_patterns": list(artifact_set.allow_patterns),
        }

    from huggingface_hub import snapshot_download

    artifact_set.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=artifact_set.dataset_id,
        repo_type="dataset",
        revision=artifact_set.revision,
        local_dir=str(artifact_set.local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=list(artifact_set.allow_patterns) or None,
    )
    return {
        "name": artifact_set.name,
        "status": "downloaded",
        "dataset_id": artifact_set.dataset_id,
        "revision": artifact_set.revision,
        "local_dir": str(artifact_set.local_dir),
        "allow_patterns": list(artifact_set.allow_patterns),
    }


def parse_checksums(path: Path = CHECKSUMS_PATH) -> dict[Path, str]:
    checksums: dict[Path, str] = {}
    if not path.exists():
        return checksums
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, relative_path = stripped.split(maxsplit=1)
        checksums[resolve_repo_path(relative_path)] = digest
    return checksums


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(path: Path = CHECKSUMS_PATH) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for file_path, expected in parse_checksums(path).items():
        if not file_path.exists():
            reports.append({"path": str(file_path), "status": "missing", "expected_sha256": expected})
            continue
        actual = sha256_file(file_path)
        reports.append(
            {
                "path": str(file_path),
                "status": "passed" if actual == expected else "failed",
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    return reports


def verify_artifact_layout() -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for artifact_set in artifact_sets().values():
        exists = artifact_set.local_dir.exists()
        has_files = exists and any(path.name != ".gitkeep" for path in artifact_set.local_dir.iterdir())
        if not artifact_set.is_configured:
            status = "not_configured"
        elif not exists:
            status = "missing"
        elif not has_files:
            status = "empty"
        else:
            status = "present"
        reports.append(
            {
                "name": artifact_set.name,
                "status": status,
                "configured": artifact_set.is_configured,
                "dataset_id": artifact_set.dataset_id,
                "revision": artifact_set.revision,
                "local_dir": str(artifact_set.local_dir),
            }
        )
    return reports
