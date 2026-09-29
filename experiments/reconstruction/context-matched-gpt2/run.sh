#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
uv run python experiments/reconstruction/context-matched-gpt2/fit.py "$@"
