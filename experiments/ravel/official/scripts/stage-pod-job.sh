#!/usr/bin/env bash
# Stage the self-contained RunPod job folder (ignored by Git) from the pinned SAEBench backend.
# Usage (from repo root): bash experiments/ravel/official/scripts/stage-pod-job.sh
set -euo pipefail
SAEBENCH=${SAEBENCH:-$HOME/.cache/icalens/experiments/saebench/paper-standard/2a7480bf67f9f82f79e742be55fc1f5ec404bfa9}
JOB=experiments/ravel/pilot-runs/pod-job
mkdir -p "$JOB"
cp experiments/ravel/trial/run_trial.py "$JOB/"
rsync -a --delete --exclude __pycache__ "$SAEBENCH/sae_bench" "$JOB/"
# Exact local package versions except torch (the pod image supplies torch 2.8 + CUDA 12.8).
# Installed with `pip install --no-deps` on the pod because the frozen set has a benign
# numpy pin conflict that uv tolerates.
uv pip freeze --python "$SAEBENCH/.venv/bin/python" 2>/dev/null \
  | grep -viE "^(torch|torchvision|torchaudio|triton|nvidia-|cuda-|sae-bench|-e )" \
  | grep -v "@ file" > "$JOB/requirements.txt"
echo "staged $JOB"
