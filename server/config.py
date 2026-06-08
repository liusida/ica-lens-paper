from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


V6_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_REPO = "sida/ica-lens-paper"
DEFAULT_DB_REPO = "sida/ica-lens-paper"
DEFAULT_DB_FILENAME = "ica_probe_mini.sqlite"
DEFAULT_MODEL_ID = "openai-community/gpt2"
DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_MODEL_REGISTRY = {
    "gpt2": {"model_id": "openai-community/gpt2", "display_name": "GPT-2", "context_length": 1024, "dtype": "bfloat16", "dataset_path": "NeelNanda/pile-10k", "dataset_name": "", "dataset_split": "train", "dataset_text_column": "text", "dataset_streaming": False},
    "gemma2_2b": {"model_id": "google/gemma-2-2b", "display_name": "Gemma 2 2B", "context_length": 1024, "dtype": "bfloat16", "dataset_path": "NeelNanda/pile-10k", "dataset_name": "", "dataset_split": "train", "dataset_text_column": "text", "dataset_streaming": False},
    "qwen3_5_2b_base": {"model_id": "Qwen/Qwen3.5-2B-Base", "display_name": "Qwen3.5 2B Base", "context_length": 1024, "dtype": "bfloat16", "dataset_path": "NeelNanda/pile-10k", "dataset_name": "", "dataset_split": "train", "dataset_text_column": "text", "dataset_streaming": False},
}


@dataclass(frozen=True)
class ModelSettings:
    model_name: str
    model_id: str
    display_name: str
    ica_dir: Path
    context_length: int
    dtype: str
    dataset_path: str
    dataset_name: str | None
    dataset_split: str
    dataset_text_column: str
    dataset_streaming: bool


@dataclass(frozen=True)
class Settings:
    db_path: Path
    ica_dir: Path
    ica_root: Path
    artifact_repo: str
    db_repo: str
    hf_revision: str | None
    model_id: str
    model_name: str
    device: str
    dtype: str
    context_length: int
    download_missing: bool
    models: dict[str, ModelSettings]
    use_gpt2_layer11_patch: bool = False


def load_settings() -> Settings:
    fetched_root = V6_ROOT / "artifacts" / "fetched"
    db_path = Path(os.environ.get("ICA_EXPLORER_DB_PATH", str(fetched_root / "databases" / DEFAULT_DB_FILENAME))).expanduser()
    ica_root = Path(os.environ.get("ICA_EXPLORER_ICA_ROOT", str(fetched_root / "models"))).expanduser()
    ica_dir = Path(os.environ.get("ICA_EXPLORER_ICA_DIR", str(ica_root / DEFAULT_MODEL_NAME))).expanduser()
    models = {
        model_name: ModelSettings(
            model_name=model_name,
            model_id=str(meta["model_id"]),
            display_name=str(meta["display_name"]),
            ica_dir=ica_root / model_name,
            context_length=int(meta["context_length"]),
            dtype=str(meta["dtype"]),
            dataset_path=str(meta["dataset_path"]),
            dataset_name=str(meta["dataset_name"]) or None,
            dataset_split=str(meta["dataset_split"]),
            dataset_text_column=str(meta["dataset_text_column"]),
            dataset_streaming=bool(meta["dataset_streaming"]),
        )
        for model_name, meta in DEFAULT_MODEL_REGISTRY.items()
    }
    if os.environ.get("ICA_EXPLORER_ICA_DIR"):
        default = models[DEFAULT_MODEL_NAME]
        models[DEFAULT_MODEL_NAME] = ModelSettings(
            model_name=default.model_name,
            model_id=default.model_id,
            display_name=default.display_name,
            ica_dir=ica_dir,
            context_length=default.context_length,
            dtype=default.dtype,
            dataset_path=default.dataset_path,
            dataset_name=default.dataset_name,
            dataset_split=default.dataset_split,
            dataset_text_column=default.dataset_text_column,
            dataset_streaming=default.dataset_streaming,
        )
    return Settings(
        db_path=db_path,
        ica_dir=ica_dir,
        ica_root=ica_root,
        artifact_repo=os.environ.get("ICA_EXPLORER_ARTIFACT_REPO", DEFAULT_ARTIFACT_REPO),
        db_repo=os.environ.get("ICA_EXPLORER_DB_REPO", DEFAULT_DB_REPO),
        hf_revision=os.environ.get("ICA_EXPLORER_HF_REVISION") or None,
        model_id=os.environ.get("ICA_EXPLORER_MODEL_ID", DEFAULT_MODEL_ID),
        model_name=os.environ.get("ICA_EXPLORER_MODEL_NAME", DEFAULT_MODEL_NAME),
        device=os.environ.get("ICA_EXPLORER_DEVICE", "auto"),
        dtype=os.environ.get("ICA_EXPLORER_DTYPE", "bfloat16"),
        context_length=int(os.environ.get("ICA_EXPLORER_CONTEXT_LENGTH", "1024")),
        download_missing=os.environ.get("ICA_EXPLORER_DOWNLOAD_MISSING", "1").strip().lower() not in {"0", "false", "no"},
        models=models,
    )
