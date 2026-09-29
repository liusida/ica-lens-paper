#!/usr/bin/env bash
set -euo pipefail

ACTIVATIONS="$HOME/Expansion/research/ICA-data/icalens-reconstruction-activations/gpt2"

uv run icalens experiment reconstruction capture \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers all \
  --preset paper \
  --capture-layers-at-once all \
  --output "$ACTIVATIONS"
