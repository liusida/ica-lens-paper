#!/usr/bin/env bash
set -euo pipefail

ROOT="experiments/reconstruction/official"

uv run icalens experiment figure reconstruction \
  "$ROOT/results/gpt2-context64-all-eval64" \
  "$ROOT/results/gemma2" \
  "$ROOT/results/qwen35-9b" \
  --panel-titles "GPT-2 Small,Gemma 2 2B,Qwen 3.5 9B Base" \
  --output "$ROOT/figures" \
  --format png,pdf \
  --force

for spec in \
  "gpt2-context64-all-eval64|gpt2|GPT-2 Small" \
  "gemma2|gemma2|Gemma 2 2B" \
  "qwen35-9b|qwen35-9b|Qwen 3.5 9B Base"
do
  source="${spec%%|*}"
  remainder="${spec#*|}"
  output="${remainder%%|*}"
  title="${remainder#*|}"
  uv run icalens experiment figure reconstruction \
    "$ROOT/results/$source" \
    --panel-titles "$title" \
    --output "$ROOT/figures/$output-preview" \
    --format png \
    --force
done
