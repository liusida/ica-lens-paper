#!/usr/bin/env bash
# Evaluate all saved states on the common new-token pool using two GPUs.
set -euo pipefail
REPO=/cmlscratch/feijhan/code/ica-lens-paper
PYTHON_BIN=/cmlscratch/feijhan/envs/looped_transformer/bin/python
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u "$REPO/experiments/ica-fitting-sample-size/evaluate_extra_tokens.py" \
  --budgets 1k 5k 10k > "$REPO/experiments/ica-fitting-sample-size/eval-extra-small.log" 2>&1 &
FIRST_PID=$!
SECOND_STATUS=0
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -u "$REPO/experiments/ica-fitting-sample-size/evaluate_extra_tokens.py" \
  --budgets 100k 1m > "$REPO/experiments/ica-fitting-sample-size/eval-extra-large.log" 2>&1 || SECOND_STATUS=$?
FIRST_STATUS=0
wait "$FIRST_PID" || FIRST_STATUS=$?
if (( FIRST_STATUS != 0 || SECOND_STATUS != 0 )); then
  echo "At least one evaluation failed: first=$FIRST_STATUS second=$SECOND_STATUS" >&2
  exit 1
fi
