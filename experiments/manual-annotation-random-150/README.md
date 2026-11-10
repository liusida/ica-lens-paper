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

## Second annotator

Annotator 2 labeled the same 150 components on the same 0–5 confidence scale.

- `raw/annotator2-upload-2026-11-09.csv` is the uploaded sheet, renamed only
  (uploaded as `ica-lens-two-annotators-2026-11-09 (1).csv`). It is annotator 1's
  sheet with `new_annotation` and `new_confidence` columns added, plus
  component context (skewness, kurtosis, top tokens and contexts).
- `annotations-annotator2.json` holds annotator 2 in the same schema as
  `annotations.json` (annotator 1). Regenerate it with:

```bash
uv run python experiments/manual-annotation-random-150/extract_annotator2.py --force
```

The extractor fails if the CSV's annotator-1 columns differ from
`annotations.json`; all 150 rows match.

Confidence agreement is computed with:

```bash
uv run python experiments/manual-annotation-random-150/agreement.py
```

It writes `agreement/agreement.json` and `agreement/agreement.md`
(exact and within-1 agreement, weighted Cohen's κ, Spearman ρ, and agreement on
confidence ≥ 4). Labels are not compared: annotator 1 uses short names and
annotator 2 uses longer descriptive phrases, so label agreement needs a
by-meaning judgment.

To record: how annotator 2's labels were produced, and whether annotator 1's
labels (present in the uploaded sheet) were visible during annotation.
