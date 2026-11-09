#!/usr/bin/env bash
# Run all five revised budgets concurrently on the two GPUs of one allocation.
set -euo pipefail
REPO=/cmlscratch/feijhan/code/ica-lens-paper
EXP="$REPO/experiments/ica-fitting-sample-size/work"
PYTHON_BIN=/cmlscratch/feijhan/envs/looped_transformer/bin/python
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$EXP/runs" "$EXP/logs"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u "$REPO/experiments/ica-fitting-sample-size/run.py" \
  --activations "$EXP/activations/gpt2-pile10k-layer11-1m" \
  --output "$EXP/runs" --budget 1m > "$EXP/logs/fit-1m.log" 2>&1 &
LARGE_PID=$!

SMALL_STATUS=0
for BUDGET in 1k 5k 10k 100k; do
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -u "$REPO/experiments/ica-fitting-sample-size/run.py" \
    --activations "$EXP/activations/gpt2-pile10k-layer11-1m" \
    --output "$EXP/runs" --budget "$BUDGET" > "$EXP/logs/fit-$BUDGET.log" 2>&1 || SMALL_STATUS=$?
done
LARGE_STATUS=0
wait "$LARGE_PID" || LARGE_STATUS=$?
if (( SMALL_STATUS != 0 || LARGE_STATUS != 0 )); then
  echo "At least one fit failed: small=$SMALL_STATUS large=$LARGE_STATUS" >&2
  exit 1
fi
export MPLCONFIGDIR="$EXP/cache/matplotlib"
"$PYTHON_BIN" "$REPO/experiments/ica-fitting-sample-size/plot.py" \
  --input "$EXP/runs" --output "$REPO/experiments/ica-fitting-sample-size/figures" --force
