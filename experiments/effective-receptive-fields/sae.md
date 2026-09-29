# Official SAE suffix-sweep ERF

This experiment extends the validated Gemma pilot to GPT-2 Small (12 layers),
Gemma 2 2B (26), and Qwen 3.5 9B Base (32). It uses the pinned SAE baseline
registry and existing 1M-token Pile-10k activation caches, not new SAE training.

## Commands

From the repository root:

```bash
uv run python experiments/effective-receptive-fields/sae.py --dry-run
uv run python experiments/effective-receptive-fields/sae.py
```

For a layer-level validation run, in a separate output:

```bash
uv run python experiments/effective-receptive-fields/sae.py --models gemma2 --layers 12 --output experiments/effective-receptive-fields/runs/sae-validation
```

Defaults: 100 features sampled uniformly from the features active on the cached
token pool, up to 20 strongest positive examples per feature, seed 0, and
thresholds `1,3,5,10,15`. `--rank-thresholds 5`
selects the alternative top-5-only protocol and requires a different output.
All layer indices are zero-based. As in the ICA experiment, a stable hash of the
model label, layer, and seed gives each layer an independent deterministic
sample. Taking the first 100 active features in that random permutation is
uniform sampling without replacement from the eligible population.

## Definition and controls

- Score = existing checkpoint encoder activation multiplied by decoder norm.
  Rank uses strict competition ranking among the entire dictionary (32k, 16k,
  and 64k respectively), with positive activation required for recovery.
- Inputs preserve cached 1024-token framing: Gemma uses BOS, GPT-2 and Qwen
  use their recorded end-of-text document prefix. Targets exclude that prefix.
  Selection uses the 1M cached tokens, not every
  token in the larger candidate corpus.
- GPT-2's SAE was trained at context length 64. This experiment evaluates it
  at 1024, matching the released ICA ERF protocol, not the context-64
  reconstruction control. The comparison is not training-context matched.
- Reuse the official suffix engine: exact lengths 1–10, then the existing
  geometric schedule; exact or geometric-bracket first recovery estimates;
  available full-context length for unrecovered occurrences. Record recovery
  flags separately. A long assigned ERF is not proof of long-context dependence.
- As in ICA, the ERF population contains only features with at least one stored
  selected-tail occurrence. Dead-feature prevalence is a separate measurement
  and is not mixed into the ERF distribution.
- Checkpoint encoded features and full-context ranks are used as stored
  endpoints, as in the ICA suffix-sweep experiment.

## Resume and artifacts

The launcher owns one persistent progress box over all 70 layers. It starts one
worker per model, matching the ICA experiment's lifetime: each language model
loads once and is reused across its layers. One prepared bundle and one result
bundle are written per layer. The durable result boundary and displayed unit are
therefore both one layer; interruption repeats at most the current layer's
measurement. Prepared inputs survive an interrupted measurement, so their
activation scan remains reusable.

Repeating the same command validates identities and completed result bundles,
skips finished layers, and avoids loading a model whose layers are all complete.
Configuration changes require a distinct output. The default is
`runs/sae-suffix-sweep-v2`, which cannot mix with the fine-grained `v1` output.
Per-model summaries use the same row-oriented JSON and CSV organization as ICA.

Use feature fractions, not dictionary-size-extrapolated counts, for ICA–SAE
comparisons. Report threshold, recovery coverage, context protocol, and
dictionary width alongside ERF distributions. This experiment conditions on
active features; it does not measure dictionary-wide inactive-feature prevalence.
This implementation does not alter paper figures or launch the full experiment
automatically.

## Implementation validation (2026-09-05)

- 11 targeted CPU tests pass, including the existing ERF API and formula tests.
- The original pilot passed GPU smoke runs for all three SAE formats. The
  revised ICA-style runner additionally passed a GPT-2 layer smoke run and a
  two-layer run: it wrote only prepared/result layer bundles, loaded GPT-2 once
  across both layers, and skipped model loading on replay.
