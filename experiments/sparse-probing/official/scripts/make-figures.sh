#!/usr/bin/env bash
set -euo pipefail

uv run icalens experiment figure sparse-probing \
  experiments/sparse-probing/official/results/gpt2 \
  experiments/sparse-probing/official/results/gemma2 \
  experiments/sparse-probing/official/results/qwen35-9b \
  --panel-titles "GPT-2 Small,Gemma 2 2B,Qwen 3.5 9B Base" \
  --output experiments/sparse-probing/official/figures \
  --format png,pdf \
  --force
