# Reproduction

`v6` currently supports two publication-ready paths.

1. Artifact reuse: download fitted ICA models and explorer databases, verify
   them, and inspect components.
2. Miniature reproduction: run small GPT-2, Gemma, and Qwen capture-and-fit pipelines locally.

The full paper-scale reproduction is intentionally not wired as a one-command
workflow yet. The numbered workflow scripts are kept modular so the paper-scale
path can be added without changing the published artifact or demo interfaces.

## Miniature Reproduction

```bash
git submodule update --init --recursive
bash scripts/setup_saebench_envs.sh
uv run python scripts/reproduce_all.py --mode demo --clean --force --erf-limit 1
```

This runs the standalone miniature pipeline: it fetches and verifies published
artifacts, captures 3,000 tokens from `NeelNanda/pile-10k` for GPT-2 Small,
Gemma 2 2B, and Qwen 3.5 2B Base, fits a 128-component FastICA model for one
representative layer per model, builds a demo explorer database, computes small
non-Gaussianity and ICA-SAE overlap tables, and runs small SAEBench sparse-probe
and TPP checks on each fitted ICA artifact.

The sparse-probe check now evaluates ICA against PCA, public SAE baselines, and
ITDA for each demo model. Gemma 2 2B layer 12 also evaluates Matryoshka SAE
prefixes at widths 128 and 512.

The default demo layers are:

- `gpt2`: `layer_06`
- `gemma2_2b`: `layer_12`
- `qwen3_5_2b_base`: `layer_12`

To debug only the capture/fit/SAEBench part for one model:

```bash
uv run python scripts/run_demo.py --models gpt2
```

The underlying workflow scripts can also be run directly:

```bash
uv run python workflows/01_capture_activations.py --config configs/activations/gpt2.toml --token-budget 3000 --output-root results/demo/activations
uv run python workflows/02_fit_ica.py --config configs/fit_ica/gpt2.toml --activation-root results/demo/activations --token-budget 3000 --output-root results/demo/ica --layers layer_06 --n-components 128
uv run python workflows/03_compute_nongaussianity.py --models gpt2 --layers layer_06 --token-budget 3000 --activation-root results/demo/activations --ica-root results/demo/ica --output-root results/demo/nongaussianity --families ica random --random-directions 128
uv run python workflows/04_build_explorer_db.py --models gpt2 --layers layer_06 --token-budget 3000 --activation-root results/demo/activations --ica-root results/demo/ica --output-db results/demo/databases/ica_probe_demo.sqlite --examples-per-region 5
uv run python workflows/05_populate_erf.py --models gpt2 --layer layer_06 --limit 8 --db-path results/demo/databases/ica_probe_demo.sqlite --ica-root results/demo/ica --token-budget 3000
uv run python workflows/06_compare_ica_sae_overlap.py --models gpt2 --layers layer_06 --token-budget 3000 --ica-root results/demo/ica --output-root results/demo/ica_sae_overlap
uv run python workflows/07_run_saebench_tpp.py --model gpt2 --layer layer_06 --token-budget 3000 --ica-root results/demo/ica
uv run python workflows/08_run_saebench_sparse_probe.py --model gpt2 --layer layer_06 --token-budget 3000 --ica-root results/demo/ica
```

Use `--methods` to restrict sparse-probe methods, for example
`--methods ica pca sae_baseline itda`. The default `--methods all` skips
Matryoshka automatically unless the target is `gemma2_2b` `layer_12`.

`01_capture_activations.py` can also store the model input embedding matrix as
an `embedding` layer. It does this automatically when all layers are captured;
for selected-layer runs, pass `--include-embedding` explicitly.

`03_compute_nongaussianity.py` writes per-direction excess-kurtosis CSVs and
layer summaries for ICA, random, and optionally SAE directions.

`04_build_explorer_db.py` builds the base explorer SQLite database from
local activations and ICA artifacts. It creates `components`, `examples`,
`context_tokens`, and `import_meta`.

`05_populate_erf.py` enriches that database with `effective_receptive_fields`.
Use `--limit` while testing; a full pass loads the model and probes each
selected component.

`06_compare_ica_sae_overlap.py` writes JSON/CSV overlap results comparing ICA
component directions to public SAE decoder directions.
