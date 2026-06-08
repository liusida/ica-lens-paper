from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .config import Settings


RELEASED_DATABASES = {"ica_probe_mini.sqlite", "ica_probe_full.sqlite"}


def resolve_db_path(settings: Settings) -> Path:
    if settings.db_path.is_file():
        _validate_readable_file(settings.db_path)
        return settings.db_path
    if not settings.download_missing:
        raise FileNotFoundError(f"SQLite database does not exist: {settings.db_path}")
    if settings.db_path.name not in RELEASED_DATABASES:
        raise FileNotFoundError(
            f"SQLite database does not exist: {settings.db_path}. "
            "Automatic download is only supported for released mini/full databases."
        )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    local_dir = settings.db_path.parent.parent if settings.db_path.parent.name == "databases" else settings.db_path.parent
    downloaded = Path(
        hf_hub_download(
            repo_id=settings.db_repo,
            repo_type="dataset",
            filename=f"databases/{settings.db_path.name}",
            revision=settings.hf_revision,
            local_dir=local_dir,
        )
    )
    _validate_readable_file(downloaded)
    return downloaded


def resolve_ica_dir(settings: Settings, *, model_name: str | None = None, ica_dir: Path | None = None) -> Path:
    target_model = model_name or settings.model_name
    target_dir = ica_dir or settings.ica_dir
    if _has_fastica_artifacts(target_dir):
        return target_dir
    if not settings.download_missing:
        raise FileNotFoundError(f"ICA artifacts do not exist: {target_dir}")
    target_dir.parent.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=settings.artifact_repo,
        repo_type="dataset",
        revision=settings.hf_revision,
        allow_patterns=[f"models/{target_model}/**"],
        local_dir=target_dir.parent.parent,
    )
    if not _has_fastica_artifacts(target_dir):
        raise FileNotFoundError(f"Downloaded artifacts but found no FastICA files in {target_dir}")
    return target_dir


def validate_ica_dir(ica_dir: Path) -> None:
    if not _has_fastica_artifacts(ica_dir):
        raise FileNotFoundError(f"No *_fastica.pt artifacts found in {ica_dir}")


def _has_fastica_artifacts(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*_fastica.pt"))


def _validate_readable_file(path: Path) -> None:
    try:
        with path.open("rb"):
            pass
    except OSError as exc:
        raise RuntimeError(f"File exists but cannot be accessed: {path}") from exc
