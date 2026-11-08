# Autointerpretability twin data (GPT-2 Small, Gemma 2 2B)

These are frozen per-layer summaries of the existing stronger-tail ICA vs SAE
Tinker runs. They match the Qwen panel of the paper's Figure 4.

`prepare_twin_data.py` writes `autointerpretability-gpt2_small.npz` and
`autointerpretability-gemma2_2b.npz`.

## Source

- Runs: `icalens_pypi/experiments/autointerpretability/runs/{gpt2,gemma-2-2b}-tinker`.
- Evaluators: explainer Inkling, simulator Qwen3.8-27B; prompt hash
  `aac7ff94…`.

## Method

The extractor is a verbatim copy of the paper's `_autointerpretability_panel`
from `icalens_pypi/paper/overleaf-v2.1/scripts/prepare_data.py`. That paper
folder is not version-controlled, so the copy lives here.

Each file holds, per layer and method:
- the mean `combined_score`;
- a deterministic bootstrap 95% CI (10,000 resamples);
- the finite-score count.

## Check

Running the same extractor on the Qwen run reproduces the paper's
`ica-vs-sae.npz` `qwen_*` arrays exactly. The script exits with an error if it
does not.

## Arrays

`method_ids`, `layers`, `mean_scores[method, layer]`, `ci95_low`, `ci95_high`,
`finite_score_counts`, plus provenance strings.
