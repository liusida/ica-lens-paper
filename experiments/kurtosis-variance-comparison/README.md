# Kurtosis and variance comparison

This pilot compares ICA, SAE, PCA, and random directions at every residual-stream
layer of GPT-2 Small, Gemma 2 2B, and Qwen 3.5 9B Base. It streams the existing
one-million-token activation datasets; it does not run the language models again.

The comparison is deliberately direction-level. For every method and layer, it
uses one model hidden dimension's worth of directions and
normalizes each direction to unit L2 norm. ICA uses rows of the effective reading
matrix, SAE uses decoder rows, PCA uses covariance eigenvectors recovered using the
same definition as the project's SAEBench PCA baseline, and Random uses a seeded
orthonormal basis. SAE nonlinear encoding is not applied. The run retains the
population mean, variance, skewness, and excess kurtosis (`m4 / m2^2 - 3`) of the
same raw linear projection for every method. Per-direction measurements are retained
in each layer checkpoint.

The common count is 768 for GPT-2, 2,304 for Gemma, and 4,096 for Qwen. This is the
complete ICA/PCA/Random basis and an equally sized, deterministically sampled subset
of the wider SAE dictionary. Sampling with replacement is never used. The optional
`--directions N` argument applies a smaller cap for smoke runs only.

Resume validation hashes only the small ICA and activation manifests. Large ICA
tensor and SAE checkpoint files are identified by their pinned manifest/repository
revision, layer/checkpoint name, format, and byte size; the preflight does not scan
their tensor contents merely to compute hashes.

The full run is resumable at the model-layer boundary:

```bash
uv run python experiments/kurtosis-variance-comparison/run.py
```

A small implementation check can use a separate output path:

```bash
uv run python experiments/kurtosis-variance-comparison/run.py \
  --models gpt2 --layers 0 --activation-samples 10000 --directions 8 \
  --output experiments/kurtosis-variance-comparison/runs/smoke
```

At any point, including while the run is partial:

```bash
uv run python experiments/kurtosis-variance-comparison/plot.py
```

The preview discovers valid completed layer checkpoints directly and writes five
PNGs for projection mean, variance, skewness, ordinary kurtosis, and excess
kurtosis. The stored
excess kurtosis is shifted by three for the kurtosis plot, whose dashed reference at
three denotes a Gaussian. The excess-kurtosis row uses a symmetric-log scale and a
Gaussian reference at zero because valid excess kurtosis can be negative. Each
figure follows the
sparse-probing figure convention: one row of three model panels, a shared method
legend, and explicit partial-coverage labels. Curves show the median across the
sampled directions. Because direction signs are arbitrary for ICA, PCA, and Random,
the mean and skewness figures summarize absolute values. Layer `.npz` files retain
the signed per-direction measurements. The unprefixed figures aggregate directions
with the median. A parallel `mean-*.png` set uses the arithmetic mean across
directions. For sign-sensitive statistics, these are respectively median absolute
and mean absolute summaries. The two `*kurtosis-variance-over-layers.png` figures
combine the five statistics as rows and the three models as columns.

When Gemma layers 9, 12, and 14 are complete, the plotter also writes
`gemma-random-kurtosis-distribution-l09-l12-l14.png`, an empirical cumulative
distribution diagnostic for Random-direction excess kurtosis at those layers. It
also writes a same-layer `gemma-random-kurtosis-histogram-l09-l12-l14.png` using
common symmetric-log-spaced bins and direction fractions.
