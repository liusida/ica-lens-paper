#!/usr/bin/env bash
# Recreate the RAVEL figures from official/results.
# Usage (from repo root): bash experiments/ravel/official/scripts/make-figures.sh
set -euo pipefail
uv run python experiments/ravel/trial/plot_default.py \
  experiments/ravel/official/results experiments/ravel/official/figures
