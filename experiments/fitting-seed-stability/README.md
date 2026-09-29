# FastICA fitting-seed stability

This pilot refits GPT-2 small layer 6 five times from the exact one-million-token
activation cache used for the official ICA Lens. The activation rows, model
revision, dataset revision, preprocessing, fitting algorithm, iteration count,
and batch size are held fixed. Only the FastICA initialization seed changes.

The settings are taken from the official layer-6 artifact: 768 components,
parallel FastICA with the logcosh contrast, unit-variance whitening, no ICA Lens
preprocessing or source scaling, 50 fixed iterations, objective recording every
iteration, and a fitting batch size of 32,768.

This isolates optimization stability from activation-sampling stability. ICA
components must later be compared up to permutation and sign.

Before launching a fit, the runner validates the reference Lens and activation
cache against these settings. It refuses incompatible inputs rather than
silently producing a mixed comparison.

Run all five fits from the repository root:

```bash
uv run python pilot-experiments/fitting-seed-stability/run.py
```

The ignored `output/` directory contains one independently resumable Lens per
seed:

```text
output/
├── seed-0-iter-50/
├── seed-1-iter-50/
├── seed-2-iter-50/
├── seed-3-iter-50/
└── seed-4-iter-50/
```

Repeating the command validates completed artifacts, skips valid fits, and
continues at the first missing fit. Use `--seeds` and `--iterations` to run
subsets. For the additional 20-, 200-, and 500-iteration fits for seeds 0, 2,
3, and 4, run:

```bash
uv run python pilot-experiments/fitting-seed-stability/run.py --seeds 0,2,3,4 --iterations 20,200,500
```

After all fits complete, run the matched comparison:

```bash
uv run python pilot-experiments/fitting-seed-stability/analyze.py
```

For each of the ten seed pairs, this computes all cross-run absolute cosine
similarities between reading directions and finds the optimal one-to-one
component assignment. Absolute cosine handles sign ambiguity; assignment
handles permutation ambiguity. Results are written under `results/` as a JSON
summary, component-level CSV, and similarity-by-reference-rank plot.
Existing analysis outputs are not overwritten unless `--force` is supplied.

Compare successive fits at 20–50, 50–200, and 200–500 iterations for all five
seeds with:

```bash
uv run python pilot-experiments/fitting-seed-stability/analyze_iterations.py
```

Pass `--force` to replace existing iteration-comparison outputs.
The mean/median summary figure averages the independently computed seed-level
statistics across the five seeds. The component-rank figure likewise averages
matched similarity across seeds at each rank in the earlier fit.
The seed-pair heatmaps report the mean and median optimally matched absolute
cosine similarity for every seed pair, separately at 20, 50, 200, and 500
iterations. A parallel threshold heatmap reports the percentage of matched
components with absolute cosine similarity strictly greater than 0.7.
The density figure computes a fixed-bin density for each of the ten unique seed
pairs and plots their average at each iteration count.
Plots are written under `figures/`; the median seed-pair heatmap is emitted as
both PNG and PDF for use in the paper.
