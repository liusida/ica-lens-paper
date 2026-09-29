"""Resolve and lazily prepare pinned SAEBench backends."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from icalens.cli._status import log

CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
CUDA_TORCH_VERSION = "2.11.0"
CUDA_TORCHVISION_VERSION = "0.26.0"


@dataclass(frozen=True)
class BackendResolution:
    """One exact benchmark backend selected for a model."""

    name: str
    repository: str
    commit: str
    python_version: str
    model_id: str
    saebench_model_name: str
    status: str


@dataclass(frozen=True)
class PreparedBackend:
    """A checkout and interpreter ready to execute the benchmark worker."""

    resolution: BackendResolution
    root: Path
    python: Path
    managed: bool


def load_registry() -> dict[str, Any]:
    path = files("icalens.experiments").joinpath("saebench_backends.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise RuntimeError("unsupported SAEBench backend registry schema")
    return cast(dict[str, Any], value)


def resolve_backend(model_id: str) -> BackendResolution:
    registry = load_registry()
    model = registry["models"].get(model_id)
    if model is None:
        raise ValueError(
            f"No SAEBench sparse-probing backend is registered for {model_id!r}. "
            "Only explicitly validated model IDs are accepted."
        )
    backend = registry["backends"][model["backend"]]
    resolution = BackendResolution(
        name=str(model["backend"]),
        repository=str(backend["repository"]),
        commit=str(backend["commit"]),
        python_version=str(backend["python"]),
        model_id=model_id,
        saebench_model_name=str(model["saebench_model_name"]),
        status=str(model["status"]),
    )
    if resolution.status != "verified":
        raise ValueError(
            f"{model_id} is registered but not verified with backend "
            f"{resolution.name}@{resolution.commit[:8]}."
        )
    return resolution


def default_cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    return Path(root).expanduser() / "icalens" if root else Path.home() / ".cache" / "icalens"


def prepare_backend(
    resolution: BackendResolution,
    *,
    cache_dir: Path | None = None,
    saebench_path: Path | None = None,
    refresh: bool = False,
) -> PreparedBackend:
    """Return a pinned backend, downloading and isolating it only when needed."""
    if saebench_path is not None:
        root = saebench_path.expanduser().resolve()
        _validate_checkout(root)
        python = _existing_python(root)
        if python is None or not _environment_ready(python):
            python = _create_environment(root, resolution.python_version)
        _ensure_cuda_torch(python)
        _ensure_modern_transformers(python)
        _ensure_gb10_loader(python)
        return PreparedBackend(resolution, root, python, False)

    cache = (cache_dir or default_cache_dir()).expanduser().resolve()
    root = cache / "experiments" / "saebench" / resolution.name / resolution.commit
    if refresh and root.exists():
        shutil.rmtree(root)
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{resolution.commit[:8]}-", dir=root.parent))
        try:
            subprocess.run(
                ["git", "clone", "--filter=blob:none", resolution.repository, str(stage)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(stage), "checkout", "--detach", resolution.commit],
                check=True,
            )
            os.replace(stage, root)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    _validate_checkout(root)
    actual = _git_revision(root)
    if actual != resolution.commit:
        raise RuntimeError(
            f"cached SAEBench revision mismatch: expected {resolution.commit}, got {actual}"
        )
    python = _existing_python(root)
    if python is None or not _environment_ready(python):
        python = _create_environment(root, resolution.python_version)
    _ensure_cuda_torch(python)
    _ensure_modern_transformers(python)
    _ensure_gb10_loader(python)
    return PreparedBackend(resolution, root, python, True)


def backend_description(
    resolution: BackendResolution, cache_dir: Path | None = None
) -> dict[str, Any]:
    cache = (cache_dir or default_cache_dir()).expanduser().resolve()
    root = cache / "experiments" / "saebench" / resolution.name / resolution.commit
    return {
        "name": resolution.name,
        "repository": resolution.repository,
        "commit": resolution.commit,
        "status": resolution.status,
        "cached": root.is_dir(),
        "cache_path": str(root),
        "saebench_model_name": resolution.saebench_model_name,
    }


def _validate_checkout(root: Path) -> None:
    if not (root / "sae_bench").is_dir() or not (root / "pyproject.toml").is_file():
        raise ValueError(f"not a SAEBench checkout: {root}")


def _existing_python(root: Path) -> Path | None:
    candidates = (root / ".venv" / "bin" / "python", root / ".venv" / "Scripts" / "python.exe")
    return next((path for path in candidates if path.is_file()), None)


def _create_environment(root: Path, python_version: str) -> Path:
    log(f"Preparing isolated SAEBench environment with Python {python_version}...")
    environment = root / ".venv"
    if environment.exists():
        shutil.rmtree(environment)
    subprocess.run(["uv", "venv", "--python", python_version, str(environment)], check=True)
    python = _existing_python(root)
    assert python is not None
    subprocess.run(["uv", "pip", "install", "--python", str(python), "-e", str(root)], check=True)
    return python


def _environment_ready(python: Path) -> bool:
    result = subprocess.run(
        [str(python), "-c", "import sae_bench"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _ensure_cuda_torch(python: Path) -> None:
    """Replace a backend's CPU-only PyTorch resolution with the CUDA build."""
    available = (
        subprocess.run(
            [
                str(python),
                "-c",
                "import torch; raise SystemExit(0 if torch.version.cuda else 1)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if available:
        return
    log(
        f"Installing CUDA-enabled PyTorch {CUDA_TORCH_VERSION} in the isolated "
        "SAEBench environment..."
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--upgrade",
            f"torch=={CUDA_TORCH_VERSION}",
            f"torchvision=={CUDA_TORCHVISION_VERSION}",
            "--index",
            CUDA_TORCH_INDEX,
        ],
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            (
                "import torch; assert torch.version.cuda, "
                "'installed PyTorch build does not provide CUDA'"
            ),
        ],
        check=True,
    )


def _ensure_gb10_loader(python: Path) -> None:
    available = (
        subprocess.run(
            [str(python), "-c", "import gb10_load_llm"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if not available:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "gb10-load-llm>=0.1.2",
            ],
            check=True,
        )


def _ensure_modern_transformers(python: Path) -> None:
    """Ensure the shared worker can load recent Hugging Face architectures."""
    compatible = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import transformers; "
                "major = int(transformers.__version__.split('.')[0]); "
                "raise SystemExit(0 if major >= 5 else 1)"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if compatible.returncode == 0:
        return
    log("Installing a Qwen3.5-capable Transformers release in the SAEBench environment...")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "transformers>=5.0",
        ],
        check=True,
    )


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
