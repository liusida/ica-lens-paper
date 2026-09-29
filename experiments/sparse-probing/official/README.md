# Official sparse-probing experiment

These are the sparse-probing runs accepted for the paper. Run scripts invoke
the public CLI with the exact model, layer, preset, and baseline selections used
to produce the stored results.

From the repository root, run one model with:

```bash
bash experiments/sparse-probing/official/scripts/run-gpt2.sh
```

The corresponding Gemma 2 and Qwen scripts have the same interface. Recreate
the paper figure after all three runs finish with:

```bash
bash experiments/sparse-probing/official/scripts/make-figures.sh
```

Repeating a run command resumes its existing output. `run.json` records the
fully resolved configuration and source provenance. The tracked `results.json`
and `results.csv` are compact final summaries; large logs, activation caches,
and environment checkpoints are intentionally not part of the official release.
