#!/usr/bin/env bash
# Fit each activation budget as an independent GPU task after capture completes.
# Submit with: sbatch --dependency=afterok:<CAPTURE_JOB_ID> --array=0-4%4 .../fit_array.sbatch
#SBATCH --job-name=ica-size-fit
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%4
#SBATCH --output=experiments/ica-fitting-sample-size/logs/fit-%A_%a.out

set -euo pipefail
REPO=/cmlscratch/feijhan/code/ica-lens-paper
EXP="$REPO/experiments/ica-fitting-sample-size/work"
ENV=/cmlscratch/feijhan/envs/looped_transformer
ACTIVATIONS="$EXP/activations/gpt2-pile10k-layer11-1m"
mkdir -p "$EXP/runs" "$EXP/logs" "$EXP/cache/huggingface"
export HF_HOME="$EXP/cache/huggingface"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

case "$SLURM_ARRAY_TASK_ID" in
  0) BUDGET=1k ;;
  1) BUDGET=5k ;;
  2) BUDGET=10k ;;
  3) BUDGET=100k ;;
  4) BUDGET=1m ;;
  *) echo "Unexpected array task ID: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

"$ENV/bin/python" "$REPO/experiments/ica-fitting-sample-size/run.py" \
  --activations "$ACTIVATIONS" \
  --output "$EXP/runs" \
  --budget "$BUDGET" \
  --layer 11 \
  --steps 50 \
  --seed 0
