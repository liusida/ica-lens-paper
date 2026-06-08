#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V6_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

install_common_env() {
  local repo_dir="$1"
  local transformer_lens_spec="$2"
  local transformers_spec="${3:-}"

  echo "==> Setting up ${repo_dir}"
  cd "${repo_dir}"
  uv venv --python 3.11
  uv pip install --python .venv/bin/python -e . --no-deps
  uv pip install --python .venv/bin/python \
    beartype==0.14.1 \
    collectibles==0.1.5 \
    einops==0.8.0 \
    jaxtyping==0.2.37 \
    matplotlib==3.10.0 \
    numpy==1.26.4 \
    openai==1.61.1 \
    pandas==2.2.3 \
    plotly==5.24.1 \
    pydantic==2.10.6 \
    scikit-learn==1.6.1 \
    seaborn==0.13.2 \
    tabulate==0.9.0 \
    tqdm==4.67.1 \
    safetensors \
    huggingface-hub \
    "${transformer_lens_spec}" \
    sae_lens==5.4.1

  if [[ -n "${transformers_spec}" ]]; then
    uv pip install --python .venv/bin/python "${transformers_spec}"
  fi
}

install_common_env "${V6_ROOT}/vendor/SAEBench" "transformer-lens==2.11.0"
install_common_env "${V6_ROOT}/vendor/SAEBench-qwen35" "transformer-lens" "transformers==5.10.2"

echo "==> SAEBench environments ready"
"${V6_ROOT}/vendor/SAEBench/.venv/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys
print("SAEBench python:", sys.version.split()[0])
for package in ["sae-bench", "torch", "transformer-lens", "transformers", "sae-lens"]:
    print(f"{package}=={metadata.version(package)}")
PY
"${V6_ROOT}/vendor/SAEBench-qwen35/.venv/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print("SAEBench-qwen35 python:", sys.version.split()[0])
for package in ["sae-bench", "torch", "transformer-lens", "transformers", "sae-lens"]:
    print(f"{package}=={metadata.version(package)}")
print("has qwen3_5", "qwen3_5" in CONFIG_MAPPING)
PY
