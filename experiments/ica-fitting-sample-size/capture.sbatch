#!/usr/bin/env bash
# Capture one 1M-token GPT-2 layer from the Pile before submitting fit_array.sbatch.
#SBATCH --job-name=ica-size-capture
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=experiments/ica-fitting-sample-size/logs/capture-%j.out

set -euo pipefail
REPO=/cmlscratch/feijhan/code/ica-lens-paper
EXP="$REPO/experiments/ica-fitting-sample-size/work"
ENV=/cmlscratch/feijhan/envs/looped_transformer
mkdir -p "$EXP/cache/huggingface" "$EXP/activations" "$EXP/logs"
export HF_HOME="$EXP/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

"$ENV/bin/python" -m icalens.cli capture text \
  --model openai-community/gpt2 \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --layers 11 \
  --candidate-tokens 1000000 \
  --token-budget 1000000 \
  --capture-layers-at-once 1 \
  --seed 0 \
  --output "$EXP/activations/gpt2-pile10k-layer11-1m"
