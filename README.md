<h1 align="center">
  <b>ICA Lens: A fast, compact lens for exploring interpretable directions in language-model activations.</b>
</h1>

<p align="center">
  <a href="https://liusida.github.io/ica-lens-paper/">
    <img src="https://img.shields.io/badge/Project-Page-2f80ed?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://huggingface.co/spaces/EEEAILab/ICAExplorer">
    <img src="https://img.shields.io/badge/🤗%20Space-ICA%20Explorer-ffcc4d?style=flat-square" alt="ICA Explorer">
  </a>
  <a href="https://huggingface.co/datasets/sida/ica-lens-paper">
    <img src="https://img.shields.io/badge/🤗%20Dataset-Checkpoints-f9a03f?style=flat-square" alt="Hugging Face Dataset">
  </a>
  <a href="docs/quickstart.md">
    <img src="https://img.shields.io/badge/Docs-Quickstart-4c8eda?style=flat-square&logo=readthedocs&logoColor=white" alt="Docs">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-supported-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-explorer-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/uv-package%20manager-6f42c1?style=flat-square" alt="uv">
  <img src="https://img.shields.io/badge/Reproducibility-workflows-2ea44f?style=flat-square" alt="Reproducibility">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Mechanistic%20Interpretability-ICA-111827?style=flat-square" alt="Mechanistic Interpretability">
  <img src="https://img.shields.io/badge/Independent%20Component%20Analysis-FastICA-7c3aed?style=flat-square" alt="ICA">
  <img src="https://img.shields.io/badge/Sparse%20Autoencoders-comparison-f97316?style=flat-square" alt="SAE Comparison">
  <img src="https://img.shields.io/badge/Models-GPT--2%20%7C%20Gemma--2%20%7C%20Qwen3.5-0f766e?style=flat-square" alt="Models">
</p>

<p align="center">
  <a href="docs/quickstart.md">Quickstart</a> ·
  <a href="docs/fit_one_layer_qwen36_27b.md">Fit a New Model</a> ·
  <a href="docs/reproduction.md">Reproduction</a> ·
  <a href="docs/troubleshooting.md">Troubleshooting</a> ·
  <a href="https://huggingface.co/spaces/EEEAILab/ICAExplorer">Try the Demo</a> ·
  <a href="https://huggingface.co/datasets/sida/ica-lens-paper">Download Artifacts</a>
</p>

---

**ICA Lens** is the standalone code release for the ICA Lens paper. It provides tools for fitting, browsing, annotating, and evaluating Independent Component Analysis (ICA) directions in language-model activations.

Large fitted ICA models and explorer databases are hosted on Hugging Face and downloaded into `artifacts/fetched/`. Generated reproductions are written under `results/`.

---

## Quick Start

### For Explorer Users: Mini Database

Use this path if you want the explorer with the small released database (about 225-255 MB local SQLite). This is the default fetch/server path for readers.

```bash
uv sync
uv run python scripts/fetch_artifacts.py --models --databases
uv run python scripts/verify_artifacts.py
uv run python -m server.app --port 8001
```

Open `http://127.0.0.1:8001`.

**Default server inputs:**
```
artifacts/fetched/databases/ica_probe_mini.sqlite
artifacts/fetched/models/gpt2/
artifacts/fetched/models/gemma2_2b/
artifacts/fetched/models/qwen3_5_2b_base/
```

### For Explorer Users: Full Database

Use this path if you want the released explorer state with the full database (about 7 GB local SQLite) and all released fitted ICA artifacts. Compared with the mini database, the full database mainly adds more stored component examples and score-backed token coloring/context in the explorer.

```bash
uv sync
uv run python scripts/fetch_artifacts.py --models --databases --database-variant full
uv run python scripts/verify_artifacts.py --database-variant full
ICA_EXPLORER_DB_PATH=artifacts/fetched/databases/ica_probe_full.sqlite \
uv run python -m server.app --port 8001
```

Open `http://127.0.0.1:8001`.

---

## Reproduction

### Mini Reproduction

Use this path if you want to rebuild a small three-model reproduction locally. It captures 3,000 tokens for GPT-2 Small, Gemma 2 2B, and Qwen 3.5 2B Base, fits one 128-component ICA model per model, builds a demo database, and runs small SAEBench TPP and sparse-probe checks.

```bash
uv sync
git submodule update --init --recursive
bash scripts/setup_saebench_envs.sh
uv run python scripts/reproduce_all.py --mode demo --clean --force --erf-limit 1
```

Outputs are written under `results/demo/`. To inspect the reproduced demo database in the explorer while keeping the released ICA artifact layout:

```bash
ICA_EXPLORER_DB_PATH=results/demo/databases/ica_probe_demo.sqlite \
ICA_EXPLORER_ICA_ROOT=artifacts/fetched/models \
uv run python -m server.app --port 8001
```

The mini sparse-probe workflow compares ICA with PCA, public SAE baselines, and ITDA for all three demo models. For Gemma 2 2B layer 12 it also includes Matryoshka SAE prefixes at widths 128 and 512.

### Full-Scale Reproduction

The full-scale paper workflow is modular rather than a single command. Use the numbered scripts with the model configs in `configs/`; the activation configs default to 1,000,000 tokens and the ICA configs default to all hidden layers with hidden-dimensional ICA.

For one model, the core capture-and-fit steps look like:

```bash
uv sync
git submodule update --init --recursive
bash scripts/setup_saebench_envs.sh
uv run python workflows/01_capture_activations.py --config configs/activations/gpt2.toml --output-root results/reproduced/activations --include-embedding
uv run python workflows/02_fit_ica.py --config configs/fit_ica/gpt2.toml --activation-root results/reproduced/activations --output-root results/reproduced/ica
```

Then run the analysis workflows as needed:

```bash
uv run python workflows/03_compute_nongaussianity.py --models gpt2 --token-budget 1000000 --activation-root results/reproduced/activations --ica-root results/reproduced/ica --output-root results/reproduced/nongaussianity
uv run python workflows/04_build_explorer_db.py --models gpt2 --token-budget 1000000 --activation-root results/reproduced/activations --ica-root results/reproduced/ica --output-db results/reproduced/databases/ica_probe.sqlite
uv run python workflows/05_populate_erf.py --models gpt2 --token-budget 1000000 --db-path results/reproduced/databases/ica_probe.sqlite --ica-root results/reproduced/ica
uv run python workflows/06_compare_ica_sae_overlap.py --models gpt2 --token-budget 1000000 --ica-root results/reproduced/ica --output-root results/reproduced/ica_sae_overlap
uv run python workflows/07_run_saebench_tpp.py --model gpt2 --token-budget 1000000 --ica-root results/reproduced/ica
uv run python workflows/08_run_saebench_sparse_probe.py --model gpt2 --token-budget 1000000 --ica-root results/reproduced/ica --methods all
```

Repeat with `gemma2_2b` and `qwen3_5_2b_base` configs for the other released models. Full-scale runs require substantial GPU time and disk space.

---

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `configs/` | Model, activation, ICA fitting, and analysis settings |
| `workflows/` | Numbered reproduction and analysis entrypoints |
| `scripts/` | Artifact fetching, verification, environment setup, and demo orchestration |
| `server/` | FastAPI explorer and static UI |
| `src/ica_lens/` | Reusable library code |
| `artifacts/` | Artifact manifest, checksums, and fetched artifact placeholders |
| `results/` | Generated output placeholders; actual outputs are ignored by git |

---

## Documentation

For more detailed information, see:
- [`docs/quickstart.md`](docs/quickstart.md) - Quick start guide
- [`docs/fit_one_layer_qwen36_27b.md`](docs/fit_one_layer_qwen36_27b.md) - Fit one ICA layer for a new LLM, using Qwen3.6-27B as a worked example
- [`docs/reproduction.md`](docs/reproduction.md) - Detailed reproduction instructions
- [`docs/troubleshooting.md`](docs/troubleshooting.md) - Troubleshooting guide

---

## License

This project is licensed under [MIT License](LICENSE).

---

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.


## Contact

We welcome feedback and collaboration from researchers interested in ICA, sparse autoencoders, mechanistic interpretability, and related directions.

We have also summarized several possible next steps and open directions here: [Future Projects](https://liusida.github.io/ica-lens-paper/future-projects.html). 
Please feel free to reach out if you are interested in collaborating or exploring new ideas.

<p align="center">
  <img src="https://liusida.github.io/ica-lens-paper/assets/wechat_QR.png" alt="WeChat QR code" width="160">
</p>

| Name             | Affiliation            | Email                                     |
| ---------------- | ---------------------- | ----------------------------------------- |
| **Sida Liu**     | Independent Researcher | [me@liusida.com](mailto:me@liusida.com)   |
| **Feijiang Han** | CS PhD @ University of Maryland | [feijhan@umd.edu](mailto:feijhan@umd.edu) |

