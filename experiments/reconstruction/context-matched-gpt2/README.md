# Context-matched GPT-2 ICA fit

This one-purpose experiment fits a GPT-2 Small ICA lens for comparison with
`jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs`. It uses the SAE configuration's
pretokenized training dataset directly:

- dataset: `apollo-research/Skylion007-openwebtext-tokenizer-gpt2`
- dataset revision: `f02886b54795e8acabceb637ca119f9ae8f19d3f`
- context length: 64 tokens
- framing: no prepended BOS
- activation site: each transformer block's `resid_post`
- fitting sample: 1,000,000 dataset-provided token IDs
- ICA: full rank, no ICALens preprocessing, seed 0, 50 iterations

Stored 1,024-token dataset rows are divided into contiguous 64-token examples;
the rows are first deterministically streaming-shuffled. The exact selected
token IDs are checkpointed before model loading. They are never decoded or
retokenized.

Validate the pinned inputs without starting a GPU fit:

```bash
./experiments/reconstruction/context-matched-gpt2/run.sh --validate-only
```

Run or resume the full fit by repeating the same command:

```bash
./experiments/reconstruction/context-matched-gpt2/run.sh
```

Operational checkpoints and complete logs are written to `run/` beside this
README. The fitted artifact is written separately to
`local-icalens-models/experimental/icalens-gpt2-openwebtext-context-length-64/`.
Changing a scientific or operational parameter for an existing run fails
before model loading; use a distinct `--run-output` and `--lens-output` for a
different configuration.
