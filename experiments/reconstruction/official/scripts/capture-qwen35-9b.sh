#!/usr/bin/env bash
set -euo pipefail

ACTIVATIONS="$HOME/Expansion/research/ICA-data/icalens-reconstruction-activations/qwen35-9b"

uv run icalens experiment reconstruction capture \
  --lens sida/icalens-qwen3.5-9b-base-pile10k \
  --layers all \
  --preset paper \
  --capture-layers-at-once all \
  --output "$ACTIVATIONS"
