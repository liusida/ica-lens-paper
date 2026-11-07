#!/usr/bin/env bash
# Run the four methods in parallel, one RTX 4090 pod each (~35-50 min), on the shared dataset,
# then copy the compact results into official/results/seed<SEED>/.
# Usage (from repo root): bash experiments/ravel/official/scripts/run-methods.sh [SEED]
# SEED defaults to 42 (SAEBench default). It seeds pair sampling, the train/test split and
# mask training; the dataset and the random basis stay fixed.
set -euo pipefail
SEED=${1:-42}
RUNPOD=${RUNPOD:-$HOME/research/runpod}
JOB=$(realpath experiments/ravel/pilot-runs/pod-job)
DEST=$(realpath -m experiments/ravel/official/results/seed$SEED)
mkdir -p "$DEST"
cd "$RUNPOD"
for m in sae ica pca random; do
  uv run python run_job.py --name ravel-default-$m-seed$SEED \
    --files "$JOB/run_trial.py" "$JOB/sae_bench" "$JOB/dataset_default" "$JOB/requirements.txt" \
    --cmd "pip install -q --no-deps -r requirements.txt" \
    --cmd "python run_trial.py --default-config --method $m --seed $SEED --output \$OUTPUT_DIR --dataset-dir dataset_default" \
    --sync-minutes 10 --max-hours 4 &
  sleep 20
done
wait
uv run python pods.py
for m in sae ica pca random; do
  run=$(ls -d runs/*_ravel-default-$m-seed$SEED | tail -1)
  cp "$run"/outputs/${m}_layer12.json "$DEST/"
  echo "$m: $run"
done
