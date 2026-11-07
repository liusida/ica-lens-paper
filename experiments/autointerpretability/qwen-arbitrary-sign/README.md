# Qwen arbitrary-sign recompute (rebuttal)

Goal: add an "ICA (arbitrary sign)" line to the Qwen 3.5 9B Base panel of
Figure 4B. That panel shows the stronger-tail (skewness-oriented) ICA only. The
new line uses the protocol as it was before the sign fix.

## Source runs (unchanged, in `icalens_pypi`)

- Preparation: `icalens_pypi/experiments/autointerpretability/runs/qwen3.5-9b`.
  This holds the fragments, `selection.json` and the oriented candidate
  activations.
- Evaluation: `icalens_pypi/experiments/autointerpretability/runs/qwen3.5-9b-tinker`.
  - Provider: Tinker.
  - Explainer: `thinkingmachines/Inkling`; simulator: `Qwen/Qwen3.8-27B`.
  - Protocol: `cunningham-modern`; prompt hash `aac7ff94…f8f1`.

## Plan

The protocol is not modified. Only the per-component sign changes, from
`tail_direction` (skewness) to +1, the raw fitted sign.

1. Components whose `tail_direction` is already `positive` give identical
   inputs under both signs, so their existing scores are reused.
2. Flipped (`negative`) components are recomputed with sign +1:
   - re-encode the same 50k fragments;
   - rerun `select_record_indices` (same seed: `seed + feature_id`);
   - run Tinker evaluation with the same models and prompts.

Flipped components among the 150 evaluated per layer:

| Layer | Reused (positive) | To recompute (flipped) |
|---|---|---|
| 7 | 77 | 73 |
| 15 | 83 | 67 |
| 23 | 81 | 69 |
| 31 | 81 | 69 |

That is 278 features in total, about 46% of the original ICA API budget.

## Eligibility check

- The candidate IDs are a seeded permutation that does not depend on sign.
- A feature is accepted if at least 20 of the 50k fragments activate it
  (`autointerpretability_protocol.select_record_indices`).
- The original preparation rejected 0 of 600 candidates at every layer, so the
  cohort is the first 150 candidates.
- Under sign +1 the protocol would skip an ineligible component. This is not
  expected for a weak tail over 50k fragments. If it happens, the component is
  reported as undefined, not replaced.

## Tinker API check (2026-11-08)

- With the pinned `tinker==0.25.0` (`src/icalens/experiments/_tinker_environment.py`),
  the server rejects requests: "Your Tinker SDK version is no longer supported".
  Log: `runs/tinker-smoke/logs/`.
- With `tinker==0.39.0` in a scratch venv, the API key works and both
  `thinkingmachines/Inkling` and `Qwen/Qwen3.8-27B` are still served.
- Before recomputing:
  1. Bump the SDK pin.
  2. Re-score one existing positive-skew feature and compare it with the stored
     score. This checks that the SDK change does not shift scores.

Done: the pin was bumped to `0.39.0`; tinker-cookbook stays at `0.5.5`.
Re-scoring layer 7 C2290 (positive skew, cohort position 0) into
`runs/tinker-check/` gave:

- the same prompt hash, models and record indices as the stored result;
- a similar explanation;
- combined score 0.794, against 0.699 stored (top 0.842 vs 0.764, random
  0.042 vs 0.059).

Explanation and simulation are sampled, so one feature cannot separate SDK
drift from sampling noise.

Drift check (`runs/tinker-drift20/`):

- Re-scored the first 5 cohort ICA features at each layer (20 in total) under
  the current protocol with SDK 0.39.0.
- Prompt hash and record indices match the stored results for all 20.
- Mean change in combined score (new − old) is −0.022, with a 95% CI of
  ±0.054 (bootstrap −0.076 to +0.029); the per-feature SD is 0.12.
- So there is no detectable systematic SDK shift. The per-feature changes are
  sampling noise of about ±0.12, which averages to about ±0.01 over 150
  features.

`runs/` is ignored by Git.

## Results (2026-11-08)

Preparation (`prepare_arbitrary_sign.py`, about 1 h 50 min on GB10):

- All 278 flipped features are eligible under sign +1, so none are undefined.
- The positive controls reproduce the stored activations exactly (relative
  error 0).

Tinker evaluation: 278 features in about 47 min with SDK 0.39.0, the same
models and the same prompt hash.

Mean combined score over the same 150 ICA components per layer, with
feature-bootstrap 95% CIs:

| Layer | ICA stronger-tail (Fig. 4B) | ICA arbitrary sign | Flipped only: strong → arbitrary | SAE |
|---|---|---|---|---|
| 7 | 0.465 [0.422, 0.507] | 0.335 [0.292, 0.379] | 0.457 → 0.190 (n=73) | 0.453 |
| 15 | 0.210 [0.173, 0.249] | 0.167 [0.136, 0.200] | 0.223 → 0.126 (n=67) | 0.290 |
| 23 | 0.449 [0.405, 0.495] | 0.353 [0.309, 0.398] | 0.442 → 0.233 (n=69) | 0.354 |
| 31 | 0.336 [0.291, 0.381] | 0.245 [0.202, 0.290] | 0.304 → 0.107 (n=69) | 0.304 |

The arbitrary-sign line combines stored scores for the positive-skew
components with new scores for the flipped ones. The per-layer table is in
`runs/summary.json`.

## Figure

`plot_arbitrary_sign.py` writes `figures/qwen-arbitrary-sign.{png,pdf,txt}`. It
shows SAE, ICA (arbitrary sign) and ICA (stronger-tail sign) in the same style
and colors as the Pythia panel. The existing Figure 4 files are unchanged.
