#!/usr/bin/env bash
# Render only after the complete fit array has succeeded.
#SBATCH --job-name=ica-size-plot
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/ica-fitting-sample-size/logs/plot-%j.out

set -euo pipefail
REPO=/cmlscratch/feijhan/code/ica-lens-paper
EXP="$REPO/experiments/ica-fitting-sample-size/work"
ENV=/cmlscratch/feijhan/envs/looped_transformer
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$EXP/cache/matplotlib"
"$ENV/bin/python" "$REPO/experiments/ica-fitting-sample-size/plot.py" \
  --input "$EXP/runs" \
  --output "$EXP/figures"
