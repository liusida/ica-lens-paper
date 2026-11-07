# RAVEL experiment

Evaluates whether ICA Lens features can isolate entity attributes, using the
SAEBench implementation of RAVEL (Huang et al., ACL 2024) as the main protocol.
MIB's RAVEL is a possible Gemma-only cross-check.

- `official/`: the runs and figures accepted for the paper. See
  `official/README.md`.
- `trial/`: experiment code. `run_trial.py` handles the RAVEL runs,
  `plot_default.py` the official figures and `plot_budget.py` the pilot budget
  sweep.
- `reference/`: the RAVEL paper and reference clones of SAEBench and MIB (not
  tracked).
- `pilot-runs/`: local and pilot outputs, ignored by Git.

The sections below record the exploratory pilots that led to the official
protocol.

## SAEBench RAVEL protocol (pinned commit `2a7480b`)

1. Download RAVEL prompts (`adamkarvonen/ravel_prompts`), optionally downsample,
   generate model completions, keep prompts the model answers correctly, and
   keep the top-N entities.
2. For each attribute A, build intervention pairs: *cause* pairs (base and
   source query A, so the output should switch to the source's value) and
   *isolation* pairs (base queries another attribute, so the output should not
   change).
3. Train a binary mask over featurizer latents (MDBM) at the last entity token
   with cross-entropy on both pair types. There is no sparsity penalty.
4. On held-out pairs, swap the masked latents from source to base, generate,
   and score Cause, Isolation and Disentangle = (Cause + Iso) / 2.

## Trial adaptations

- ICA uses split-sign features `[relu(z)·‖w‖, relu(−z)·‖w‖]` with decoder rows
  `±w/‖w‖`. That gives 2·d features and an exact inverse.
- The published lenses use no row normalization, so no norm handling is needed.
- The Gemma Scope SAE is the paper's sparse-probing baseline checkpoint
  (`layer_12/width_16k/average_l0_41`), with decoder norms folded into the
  features.
- The pinned SAEBench RAVEL hooks assume decoder layers return tuples. Under
  transformers 5 they return tensors, so activation capture silently reads the
  first batch row and the MDBM hook raises. The trial patches both.
- SAEBench never passes `add_error=True`, so the SAE's reconstruction error is
  dropped. The trial exposes this as `--add-error`.

## Running the trial

```bash
SAEBENCH=~/.cache/icalens/experiments/saebench/paper-standard/2a7480bf67f9f82f79e742be55fc1f5ec404bfa9
$SAEBENCH/.venv/bin/python experiments/ravel/trial/run_trial.py --method ica
$SAEBENCH/.venv/bin/python experiments/ravel/trial/run_trial.py --method sae
```

## Trial results (Gemma 2 2B, layer 12, city: Country + Continent)

About 230 pairs per attribute, roughly 70 held out, one epoch, single seed.

| Method | Disentangle | Cause | Iso | Mask size (Country / Continent) |
|---|---|---|---|---|
| Full-vector swap (no mask) | 0.44 | 0.81 | 0.07 | 2304 / 2304 of 2304 |
| Neuron DBM | 0.54 | 0.70 | 0.38 | 1656 / 1160 of 2304 |
| Gemma Scope SAE | 0.51 | 0.66 | 0.36 | 836 / 626 of 16384 |
| Gemma Scope SAE + error | 0.51 | 0.58 | 0.45 | 862 / 657 of 16384 |
| ICA split ± | 0.65 | 0.78 | 0.52 | 3132 / 1978 of 4608 |
| ICA split ± + error | 0.61 | 0.67 | 0.54 | 3245 / 1963 of 4608 |
| ICA split ± (repeat) | 0.60 | 0.67 | 0.53 | 3248 / 1977 of 4608 |
| PCA split ± (from lens whitening) | 0.61 | 0.75 | 0.46 | 3495 / 2098 of 4608 |
| Random orthogonal basis split ± (seed 0) | 0.58 | 0.70 | 0.46 | 3423 / 2095 of 4608 |

ICA's reconstruction is exact (relative error about 1e-5 on real activations),
so all three ICA rows test the same intervention. They range from 0.60 to 0.65,
so mask training alone moves the score by about ±0.05. At this sample size ICA
and PCA are indistinguishable. Both appear ahead of the SAE and the neuron mask,
but only by about two noise widths.

## Mask-budget sweep (RunPod RTX 4090, single seed)

`--budget k` restricts the mask to the k highest-scoring features. In
training, the forward pass uses only the top-k sigmoid weights, while a
straight-through gradient still reaches every feature. In evaluation, the mask
is the top-k features whose logits are positive. All bases except the SAE are
split by sign and centered at the lens center, giving 4608 features. Each cell
shows Cause / Isolation. Disentangle is about 0.5 whenever nothing changes, so
it is not shown.

| Method | k=1 | k=10 | k=100 | k=1000 | unlimited |
|---|---|---|---|---|---|
| ICA | 0.00 / 0.99 | 0.01 / 0.97 | 0.08 / 0.86 | 0.41 / 0.61 | 0.69 / 0.55 |
| PCA | 0.00 / 0.99 | 0.00 / 0.99 | 0.00 / 0.98 | 0.15 / 0.92 | 0.73 / 0.49 |
| Random orthogonal | 0.00 / 0.99 | 0.00 / 0.99 | 0.00 / 0.98 | 0.10 / 0.94 | 0.70 / 0.47 |
| Neurons | 0.00 / 0.99 | 0.00 / 0.98 | 0.02 / 0.97 | 0.37 / 0.73 | 0.70 / 0.41 |
| Gemma Scope SAE (16k) | 0.01 / 0.94 | 0.05 / 0.77 | 0.37 / 0.53 | 0.69 / 0.31 | 0.66 / 0.35 |

At k=1000 the SAE mask selects only 848 and 662 features, because just those
have positive logits. The unlimited ICA pod run scored Disentangle 0.62, inside
the local range of 0.60–0.65.

Cost: 5 pods for about 4–6.5 min each, roughly $0.40 in total, all terminated
and verified.

## Default SAEBench RAVEL (RunPod RTX 4090, single seed)

Configuration: SAEBench `RAVELEvalConfig` defaults (city: Country, Continent,
Language; Nobel laureate: Country of Birth, Field, Gender; top 500 entities,
90 templates, 5000 pairs per attribute, 2 epochs, lr 1e-3), batch size 8,
bfloat16, seed 42. Gemma 2 2B layer 12, unlimited mask, `add_error` off. All
four methods share one filtered dataset, built once on a separate pod.

| Method | Disentangle | Cause | Isolation | City D | Nobel D |
|---|---|---|---|---|---|
| ICA split ± | **0.768** | 0.753 | 0.782 | 0.69 | 0.85 |
| PCA split ± | 0.719 | 0.690 | 0.748 | 0.69 | 0.75 |
| Random orthogonal split ± (seed 0) | 0.718 | 0.688 | 0.748 | 0.69 | 0.75 |
| Gemma Scope 16k `average_l0_41` | 0.665 | 0.631 | 0.699 | 0.55 | 0.79 |

Per-attribute Disentangle:

| Method | Country | Continent | Language | Country of Birth | Field | Gender |
|---|---|---|---|---|---|---|
| ICA | 0.67 | 0.75 | 0.65 | 0.78 | 0.87 | 0.89 |
| PCA | 0.66 | 0.74 | 0.66 | 0.69 | 0.80 | 0.76 |
| Random | 0.65 | 0.75 | 0.67 | 0.67 | 0.81 | 0.77 |
| SAE | 0.55 | 0.56 | 0.53 | 0.76 | 0.85 | 0.75 |

The masks select about 1800–3000 of 4608 features for the split bases and
about 1700–2600 of 16384 latents for the SAE.

For reference, Chanin (2026) reports Disentangle 0.706 ± 0.001 over 5 seeds
for the canonical 65k Gemma Scope SAE (`average_l0_72`) under the same default
config. Our sparser 16k checkpoint scores 0.665.

Cost: dataset pod 40 min, $0.60; method pods 36–49 min each, $2.51 in total.
All pods were terminated and verified.
