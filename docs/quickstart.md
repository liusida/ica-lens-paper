# Quickstart

Install dependencies:

```bash
uv sync
```

Fetch released artifacts from `sida/ica-lens-paper`. By default this fetches
the mini explorer database:

```bash
uv run python scripts/fetch_artifacts.py --models --databases
```

Verify local artifacts:

```bash
uv run python scripts/verify_artifacts.py
```

Run the full miniature three-model reproduction:

```bash
git submodule update --init --recursive
bash scripts/setup_saebench_envs.sh
uv run python scripts/reproduce_all.py --mode demo --clean --force --erf-limit 1
```

This captures 3,000 random-token residual-stream activations for GPT-2 Small,
Gemma 2 2B, and Qwen 3.5 2B Base, fits one 128-component FastICA model per
model, builds a demo explorer database, computes small non-Gaussianity and
ICA-SAE overlap tables, and runs small SAEBench sparse-probe and TPP checks.
Outputs are written under `results/demo/`.

The sparse-probe demo runs ICA, PCA, public SAE baseline, and ITDA comparisons
for the selected layer. Gemma 2 2B layer 12 additionally runs Matryoshka SAE
prefixes at widths 128 and 512.

To debug only the capture/fit/SAEBench part for one model:

```bash
uv run python scripts/run_demo.py --models gpt2
```

Build a small explorer DB from reproduced demo artifacts:

```bash
uv run python workflows/04_build_explorer_db.py --models gpt2 --layers layer_06 --token-budget 3000 --activation-root results/demo/activations --ica-root results/demo/ica --output-db results/demo/databases/ica_probe_demo.sqlite --examples-per-region 5 --force
```

Launch the explorer:

```bash
uv run python -m server.app --port 8001
```

By default the explorer uses the published mini database and published ICA
model artifacts under `artifacts/fetched/`. To inspect the reproduced demo
database instead:

```bash
ICA_EXPLORER_DB_PATH=results/demo/databases/ica_probe_demo.sqlite \
ICA_EXPLORER_ICA_ROOT=artifacts/fetched/models \
uv run python -m server.app --port 8001
```
