# ICA Lens

Standalone code release for the ICA Lens paper.

This repository is code-only. Large fitted ICA models and explorer databases are
downloaded from the `sida/ica-lens-paper` Hugging Face dataset into
`artifacts/fetched/`. Generated reproductions are written under `results/`.

## For Explorer Users: Mini Database

Use this path if you want the explorer with the small released database
(about 225-255 MB local SQLite). This is the default fetch/server path for
readers.

```bash
uv sync
uv run python scripts/fetch_artifacts.py --models --databases
uv run python scripts/verify_artifacts.py
uv run python -m server.app --port 8001
```

Open `http://127.0.0.1:8001`.

Default server inputs:

```text
artifacts/fetched/databases/ica_probe_mini.sqlite
artifacts/fetched/models/gpt2/
artifacts/fetched/models/gemma2_2b/
artifacts/fetched/models/qwen3_5_2b_base/
```

## For Explorer Users: Full Database

Use this path if you want the released explorer state with the full database
(about 7 GB local SQLite) and all released fitted ICA artifacts.
Compared with the mini database, the full database mainly adds more stored
component examples and score-backed token coloring/context in the explorer.

```bash
uv sync
uv run python scripts/fetch_artifacts.py --models --databases --database-variant full
uv run python scripts/verify_artifacts.py --database-variant full
ICA_EXPLORER_DB_PATH=artifacts/fetched/databases/ica_probe_full.sqlite \
uv run python -m server.app --port 8001
```

Open `http://127.0.0.1:8001`.

## For Mini Reproduction Users

Use this path if you want to rebuild a small three-model reproduction locally.
It captures 3,000 tokens for GPT-2 Small, Gemma 2 2B, and Qwen 3.5 2B Base,
fits one 128-component ICA model per model, builds a demo database, and runs
small SAEBench TPP and sparse-probe checks.

```bash
uv sync
git submodule update --init --recursive
bash scripts/setup_saebench_envs.sh
uv run python scripts/reproduce_all.py --mode demo --clean --force --erf-limit 1
```

Outputs are written under `results/demo/`. To inspect the reproduced demo
database in the explorer while keeping the released ICA artifact layout:

```bash
ICA_EXPLORER_DB_PATH=results/demo/databases/ica_probe_demo.sqlite \
ICA_EXPLORER_ICA_ROOT=artifacts/fetched/models \
uv run python -m server.app --port 8001
```

The mini sparse-probe workflow compares ICA with PCA, public SAE baselines, and
ITDA for all three demo models. For Gemma 2 2B layer 12 it also includes
Matryoshka SAE prefixes at widths 128 and 512.

## For Full-Scale Reproduction Users

The full-scale paper workflow is modular rather than a single command. Use the
numbered scripts with the model configs in `configs/`; the activation configs
default to 1,000,000 tokens and the ICA configs default to all hidden layers
with hidden-dimensional ICA.

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

Repeat with `gemma2_2b` and `qwen3_5_2b_base` configs for the other released
models. Full-scale runs require substantial GPU time and disk space.

## Repository Map

- `configs/`: model, activation, ICA fitting, and analysis settings.
- `workflows/`: numbered reproduction and analysis entrypoints.
- `scripts/`: artifact fetching, verification, environment setup, and demo orchestration.
- `server/`: FastAPI explorer and static UI.
- `src/ica_lens/`: reusable library code.
- `artifacts/`: artifact manifest, checksums, and fetched artifact placeholders.
- `results/`: generated output placeholders; actual outputs are ignored by git.

See `docs/quickstart.md`, `docs/reproduction.md`, and
`docs/troubleshooting.md` for more detail.
