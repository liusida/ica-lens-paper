#!/usr/bin/env bash
set -euo pipefail

ACTIVATIONS="$HOME/Expansion/research/ICA-data/icalens-reconstruction-activations/gpt2"

uv run icalens experiment reconstruction measure \
  --lens sida/icalens-gpt2-small-pile10k \
  --activations "$ACTIVATIONS" \
  --layers all \
  --baselines all \
  --k-values 1,3,10,30,100,300 \
  --output experiments/reconstruction/pilot-runs/upgrade-full-run/results/gpt2
