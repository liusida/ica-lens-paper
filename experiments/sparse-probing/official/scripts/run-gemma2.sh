#!/usr/bin/env bash
set -euo pipefail

uv run icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gemma-2-2b-pile10k \
  --layers 12,20 \
  --preset paper \
  --baselines all \
  --output experiments/sparse-probing/official/results/gemma2
