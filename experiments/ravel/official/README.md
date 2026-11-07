# Official RAVEL experiment

This folder holds the RAVEL runs accepted for the paper (rebuttal). It uses
SAEBench's default RAVEL protocol on Gemma 2 2B, residual stream layer 12, and
compares the ICA Lens (`sida/icalens-gemma-2-2b-pile10k`) with the Gemma Scope
SAE used elsewhere in the paper (16k, `average_l0_41`) and two controls: PCA and
a random orthogonal basis.

## How this differs from the other official experiments

RAVEL is not exposed through the `icalens` CLI. It runs as a standalone script,
[`../trial/run_trial.py`](../trial/run_trial.py), inside the pinned SAEBench
backend (`paper-standard`, commit `2a7480b`). Runs used RTX 4090 pods through the
RunPod job runner in `~/research/runpod`. Every run's inputs and outputs are
mirrored in the private HF dataset `sida/runpod-runs`.

## Protocol

- SAEBench `RAVELEvalConfig` defaults:
  - city: Country, Continent, Language;
  - Nobel laureate: Country of Birth, Field, Gender;
  - top 500 entities, 90 templates, 5000 pairs per attribute;
  - 2 epochs, lr 1e-3, 6 generated tokens.
- Batch size 8, bfloat16, unlimited mask, `add_error` off. These are SAEBench's
  CLI defaults.
- All methods share one filtered dataset, built once with seed 42.
- Seeds:
  - `--seed` controls pair sampling, the train/test split and mask training;
  - the random basis always uses seed 0.
- ICA, PCA and Random use split-sign features of an exactly invertible basis
  (2 × 2304 features). PCA comes from the lens's own whitening.
- The script patches SAEBench's RAVEL hooks for transformers 5, where decoder
  layers return a tensor, not a tuple. Without the patch, activation capture
  silently reads the wrong rows and the intervention hook raises.

## Reproduce

From the repository root:

```bash
bash experiments/ravel/official/scripts/stage-pod-job.sh
```

```bash
bash experiments/ravel/official/scripts/build-dataset.sh
```

```bash
bash experiments/ravel/official/scripts/run-methods.sh 42
```

```bash
bash experiments/ravel/official/scripts/make-figures.sh
```

To add a seed, run `run-methods.sh <seed>`. It writes `results/seed<seed>/`.
`make-figures.sh` then plots the mean across all `results/seed*/` folders, with
min-max whiskers, and updates the captions.

## Results

`results/seed<N>/<method>_layer12.json` holds, for each run:
- overall and per-attribute Cause, Isolation and Disentangle;
- mask sizes and the elapsed time;
- the resolved RAVEL and featurizer configs.

Seed 42:

| Method | Disentangle | Cause | Isolation |
|---|---|---|---|
| ICA | 0.768 | 0.753 | 0.782 |
| PCA | 0.719 | 0.690 | 0.748 |
| Random | 0.718 | 0.688 | 0.748 |
| SAE (Gemma Scope 16k `l0_41`) | 0.665 | 0.631 | 0.699 |

For reference, Chanin (2026) reports Disentangle 0.706 ± 0.001 under the same
config for the canonical 65k Gemma Scope SAE.

Figures:
- `figures/ravel-gemma2-layer12-overall.*` is the main-text panel;
- `figures/ravel-gemma2-layer12-attributes.*` has one panel per attribute test
  set.

## Provenance (seed 42)

| Run | RunPod run ID | Pod time | Cost |
|---|---|---|---|
| Dataset | `20261107-183047_ravel-default-dataset` | 40 min | $0.60 |
| SAE | `20261107-191146_ravel-default-sae` | 49 min | $0.72 |
| ICA | `20261107-191206_ravel-default-ica` | 36 min | $0.53 |
| PCA | `20261107-191226_ravel-default-pca` | 42 min | $0.62 |
| Random | `20261107-191246_ravel-default-random` | 43 min | $0.63 |
