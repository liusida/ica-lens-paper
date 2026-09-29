# Random 150-component manual-annotation sample

This experiment samples 50 ICA components from each base ICA Lens for GPT-2,
Gemma 2 2B, and Qwen3.5 9B. Sampling is uniform without replacement over all
available `(layer, component)` pairs within each model. The base seed is 0, and
a stable model-specific seed prevents model ordering from changing a sample.

Generate the recorded sample from the repository root:

```bash
uv run python experiments/manual-annotation-random-150/run.py
```

The selected layer and component IDs are stored in model-specific
`results/<model>/components.csv` files; complete sampling provenance is stored
in `results/sampling.json`. Existing outputs require `--force` to replace.

Build a single review page containing all 150 package-rendered component
profile panels:

```bash
uv run python experiments/manual-annotation-random-150/build_review.py
```

The combined page is written to `results/review.html`. It embeds the package's
profile-panel markup directly with one shared stylesheet; it does not use
iframes. Before rendering, the builder computes suffix-sweep ERF for every
sampled component through `lens.erf.suffix_sweep(...)`. Full results for all
five rank thresholds are cached under each model's `results/<model>/erf/`
directory. The card subtitle shows the top-15 mean suffix ERF. Existing valid
caches are reused; pass `--refresh-erf` to recompute them. Existing review
output requires `--force` to replace.

Initialize the top-level annotation record once with:

```bash
uv run python experiments/manual-annotation-random-150/create_annotations.py
```

This creates `annotations.json` with a blank label and `null` confidence for
each of the 150 sampled components and refuses to overwrite an existing
annotation file.

Fill blank labels for components whose cached top-15 mean suffix ERF is exactly
1.0 using their selected-tail top occurrence token, with surrounding whitespace
removed:

```bash
uv run python experiments/manual-annotation-random-150/autofill_erf1_labels.py
```

The command preserves every existing nonblank label.

Plot the completed manual-confidence distribution and write its summary
statistics with:

```bash
uv run python experiments/manual-annotation-random-150/plot_annotations.py
```

This writes `figures/annotation-confidence.png`,
`figures/annotation-confidence.pdf`, `figures/annotation-word-cloud.png`, and
`results/annotation-summary.json`.
Word-cloud token weights are the sums of annotation confidence scores; unknown
labels are excluded. Existing outputs require `--force` to replace.

Generate the appendix-ready LaTeX table from the completed annotations with:

```bash
uv run python experiments/manual-annotation-random-150/make_latex_table.py
```

This writes one caption-free table fragment per model under `tables/`. Existing
outputs require `--force` to replace.

Run controlled counterfactual prompts for five selected components from the
refitted Qwen ICA Lens with:

```bash
uv run python experiments/manual-annotation-random-150/run_qwen_counterfactuals.py
```

The hypotheses and two activating/two non-activating prompts per component are
stored separately in `qwen-counterfactual-tests.json`. Each target must occur
exactly once and map to one tokenizer token. Results report the component's raw
signed score and its one-based rank by absolute ICA score at that target. A
completed component is the durable resume unit; outputs are written under
`results-counterfactual/`.
