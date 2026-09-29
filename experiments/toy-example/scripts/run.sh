#!/usr/bin/env bash
set -euo pipefail

experiment_dir="experiments/toy-example"
work_dir="$experiment_dir/work"

if [[ ! -f "$work_dir/source/output/activations.safetensors" ]]; then
  uv run python "$experiment_dir/scripts/capture_gpt2_vocab.py"
fi

if [[ ! -f "$work_dir/source/ica-fit/lens/icalens.json" ]]; then
  uv run python "$experiment_dir/scripts/fit_gpt2_vocab_ica.py"
fi

uv run python "$experiment_dir/scripts/make_ica_direction_toy.py" --force
uv run --with scipy python "$experiment_dir/scripts/analyze.py" \
  --b-selection concept \
  --b-concept-rank 10 \
  --ica-lens "$work_dir/source/ica-fit/lens" \
  --figure-output "$experiment_dir/figures" \
  --force

uv run --with scipy python "$experiment_dir/scripts/make_overview_figure.py" \
  --version 1 \
  --force
uv run --with scipy python "$experiment_dir/scripts/make_overview_figure.py" \
  --version 2 \
  --activation-index 138 \
  --force

cp "$work_dir/render/results.json" "$experiment_dir/results/results.json"
