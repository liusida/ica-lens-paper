#!/usr/bin/env bash
set -euo pipefail

uv run icalens experiment saebench-sparse-probing \
  --lens sida/icalens-qwen3.5-9b-base-pile10k \
  --layers 12,20 \
  --preset paper \
  --baselines all \
  --output experiments/sparse-probing/official/results/qwen35-9b
