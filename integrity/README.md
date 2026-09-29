# Repository integrity verifier

This directory contains the independent verifier described in
`notes/artifact-integrity.md`. It invokes or inspects production experiment paths,
but owns reference selection, comparison, dependency reporting, and source-aware
run isolation outside the installed `icalens` package.

The initial preflight validates the selected model canary's reference identities
and artifact relations. GPT-2 Small layer 6 is the quick default:

```bash
uv run python -m integrity.run --check reference-preflight
```

The representative base-model canaries are GPT-2 Small layer 6, Gemma 2 2B
layer 13, and Qwen 3.5 9B layer 16. Select one or run all three with:

```bash
uv run python -m integrity.run --model gemma --check capture
uv run python -m integrity.run --model qwen --check fit
uv run python -m integrity.run --model all --check all
```

Reference paths and the layer may be overridden when one model is selected.
They cannot be overridden with `--model all`.

The terminal prints only a short pass/fail summary. Detailed production output is
saved in each check's `runtime.log`. Pass `--verbose` to expose that output and
print the complete JSON report. A flushed `START` line with a timestamp appears
before every check, so long GPU operations never leave the current stage
ambiguous. Its `PASS` or `FAIL` line is printed immediately when that check
finishes and includes its elapsed duration; the final summary
includes the invocation completion timestamp. The JSON report retains start
time, finish time, and duration for every check.

Every invocation recomputes the checks. Model- and check-specific reports are
written below `integrity/runs/<source-fingerprint>/`; a completed report from
another commit or dirty-worktree state cannot be reused for the current source.

Replay `C10` capture on 32 deterministic rows from the accepted layer-6 cache:

```bash
uv run python -m integrity.run --check capture
```

This rebuilds the recorded candidate-token population and position sample using
the production tokenization/selection functions, recaptures only the sampled
documents through the production residual hook, and requires both sample metadata
and bfloat16 activations to match `D10` exactly. The compact expected/actual arrays
are retained as `fragment.npz` below the source-specific run directory.

Replay `C11` fitting for one complete layer from the accepted activation cache:

```bash
uv run python -m integrity.run --check fit
```

FastICA couples all rows and components, so this check deliberately reruns the
complete accepted layer configuration. It compares a deterministic fragment of
64 components and 1,024 probe rows with `D11`, then discards the temporary fitted
Lens and retains only `fragment.npz`, the numerical report, and the runtime log.

Replay the population-statistics portion of `C12` profiling:

```bash
uv run python -m integrity.run --model gpt2 --check profile
```

This recomputes score moments, sign/energy statistics, and tail orientation for
all components over the complete accepted activation population. It also uses
the hash-verified official R-lens from `local-r-lens-models/official/` to
recompute logit-lens and R-lens vocabulary readouts. It compares 64 deterministic
components with `D12`. Finally, it replays pinned token/context recovery over the
complete cache, selected-tail occurrence search, and absolute-score ranks. Only
the compressed 64-component comparison fragment is retained.

Replay the shared SAE adapter and the middle-layer activation-pattern result for
all three representative models:

```bash
uv run python -m integrity.run --model all --check sae
```

This covers the three checkpoint/activation families actually used by the
project: GPT-2 TopK-32, Gemma JumpReLU, and Qwen TopK-50. In addition to matching
the accepted token-level outputs, the verifier loads checkpoint tensors through
an independent path and checks decoder normalization, coefficient rescaling,
and reconstruction numerically. These checks are intended to guard a relocation
or refactor of `SAEFeatureEncoder` in `icalens.experiments._sae` without merely
repeating its implementation.

Replay deterministic downstream aggregations, representative twin-data
preparation, and representative figure rendering:

```bash
uv run python -m integrity.run --check downstream
```

The downstream suite also directly recomputes the complete GPT-2 layer-6
ICA-to-SAE directional-overlap arrays, including nearest feature identities and
the matched-random null. This protects a second consumer of the shared SAE
checkpoint/orientation helpers rather than checking only a stored summary.

The downstream command writes a temporary skipped-check note for C14 and C30.
C14 still needs an independent reconstruction-dataset recapture implementation;
C30 depends on stochastic generation and external language judgments.
