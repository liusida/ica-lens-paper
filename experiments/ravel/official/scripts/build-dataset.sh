#!/usr/bin/env bash
# Build SAEBench's default filtered RAVEL datasets for Gemma 2 2B once, on one RTX 4090 pod
# (~40 min). All method runs share these files.
# Usage (from repo root): bash experiments/ravel/official/scripts/build-dataset.sh
set -euo pipefail
RUNPOD=${RUNPOD:-$HOME/research/runpod}
JOB=$(realpath experiments/ravel/pilot-runs/pod-job)
cd "$RUNPOD"
uv run python run_job.py --name ravel-default-dataset \
  --files "$JOB/run_trial.py" "$JOB/sae_bench" "$JOB/requirements.txt" \
  --cmd "pip install -q --no-deps -r requirements.txt" \
  --cmd 'python run_trial.py --default-config --dataset-only --output $OUTPUT_DIR --dataset-workdir /workspace/ravel_work' \
  --sync-minutes 10 --max-hours 4
run=$(ls -d runs/*_ravel-default-dataset | tail -1)
rm -rf "$JOB/dataset_default" && cp -r "$run/outputs/dataset" "$JOB/dataset_default"
echo "dataset from $run -> $JOB/dataset_default"
