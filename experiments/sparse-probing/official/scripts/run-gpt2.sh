#!/usr/bin/env bash
set -euo pipefail

uv run icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6,10 \
  --preset paper \
  --baselines all \
  --output experiments/sparse-probing/official/results/gpt2
