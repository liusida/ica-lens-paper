# Language steering

This experiment searches for SAE features and ICA components that steer Gemma 2 2B
from English toward Chinese, French, Japanese, and Spanish, then records steered
generations for each condition.

With no selection flags, the runner evaluates both methods, all four target languages,
and every layer available in the Lens:

```bash
uv run experiments/steering-language-control/run.py
```

The full default is a large, resumable run. Results are checkpointed separately under
`experiments/steering-language-control/runs/`, so rerunning the same command skips valid
completed conditions. Automatically generated directory names include the selected layer
set and the `all-positions` steering convention; for example, `--layers 19-21` uses the
suffix `layers19-21-all-positions`.

Use `--layers` to restrict the search. It accepts one layer, inclusive ranges,
comma-separated combinations, or `all`:

```bash
uv run experiments/steering-language-control/run.py --layers 19-21
uv run experiments/steering-language-control/run.py --layers 0,4-6,10
```

`--layer 20` remains available as a compatibility alias for a single layer. Likewise,
`--method sae` or `--method ica` and `--target-language french` restrict the method or
language grid.

After the Layer-20 run is available, evaluate every saved candidate with OpenAI and
regenerate `RESULTS.md` with:

```bash
uv run experiments/steering-language-control/report.py
```

The evaluator uses `gpt-4.1-mini-2025-04-14` and reads `OPENAI_API_KEY` from the project
`.env`. Judgments are cached under `evaluation-cache/`, so unchanged outputs are not billed
again. It follows the same OpenAI client and environment setup as the project's
autointerpretability experiment.

For each method and language, the evaluator scores every top-three contrast candidate on
four prompts for language adherence, quality, relevance, and degeneracy. It selects the
candidate with the most passing outputs, then displays that candidate's best passing sample.
Use `--dry-run` to print without modifying the document.

The paper does not document its prefill/KV-cache position semantics. This reproduction uses
`all-positions` consistently: every prompt position during prefill and every decoding step.
