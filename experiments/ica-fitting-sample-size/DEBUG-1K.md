# Why the 1K-token GPT-2 ICA has high training kurtosis

The checks support finite-sample overfitting by high-dimensional FastICA. No
error was found in the kurtosis formula, centering, whitening scale, final
orthogonality, or float32 projection precision. This conclusion is supported by
independent metric calculations, an independent FastICA implementation, pure
Gaussian controls, and a dimensionality sweep.

The setting uses 1,000 token-position activations and 768 fitted directions,
only about 1.30 samples per component. FastICA chooses directions on those
same samples; the resulting training kurtosis is therefore an adaptive fitting
statistic. It does not estimate the non-Gaussianity of an independently selected
population direction without fitting bias.

## Numerical and implementation checks

- `max_abs_torch_vs_numpy`: 4.5474735e-12
- `max_abs_torch_vs_scipy`: 4.5474735e-12
- `median_kurtosis_fp64_projection`: 514.93859
- `max_abs_kurtosis_fp32_vs_fp64_projection`: 0.00046226144
- `max_abs_whitened_covariance_minus_identity`: 1.0728836e-06
- `max_abs_unmixing_orthogonality_error`: 1.7881393e-06
- `max_abs_source_variance_minus_one`: 2.0003986e-06
- `max_abs_source_mean`: 4.0680356e-07

Whitening alone, before optimizing ICA, has median excess kurtosis 0.158. After ICA it is about 515.

A complete float64 refit gives training excess kurtosis 511.95; independent scikit-learn FastICA with SVD whitening gives 513.20, versus 0.56 on unseen tokens. All runs use 50 updates; the scikit-learn run reports that it has not met its convergence tolerance at 50 updates.

## What causes the large number

For a centered source, define `p_i = s_i^2 / sum_j(s_j^2)`. The empirical excess
kurtosis is exactly `N * sum_i(p_i^2) - 3`. In the saved 1K run the median
single-token squared-energy share is 71.86%, and the median single-token share
of the fourth moment is 99.82%. A token carrying about 72% of the energy
contributes roughly `1000 * 0.72^2 - 3 = 515.4` by itself. The directions have
specialized to isolated training examples, creating large empirical kurtosis.

## Independent Gaussian control

Standard Gaussian data have zero population excess kurtosis in every fixed
linear direction. Fitting 768 ICA directions on 1,000 Gaussian samples produces:

| Gaussian draw | Training median excess kurtosis | Independent-sample median excess kurtosis |
| --- | ---: | ---: |
| 0 | 453.213 | 0.00057 |
| 1 | 450.279 | 0.00114 |
| 2 | 445.934 | -0.00008 |

Thus GPT-2 semantics, rare linguistic features, and the bfloat16 activation cache are unnecessary for the effect.

## Fix 1K tokens and reduce the number of components

| Components retained by PCA whitening | Training median excess kurtosis | Unseen-token median excess kurtosis |
| --- | ---: | ---: |
| 64 | 9.805 | 6.736 |
| 128 | 16.079 | 3.925 |
| 256 | 44.598 | 2.926 |
| 512 | 211.037 | 1.623 |
| 768 | 517.226 | 0.560 |

These controls evaluate 10K unseen token positions. The earlier evaluation of
the saved full-width 1K fit on 100K unseen positions gives 0.603, versus 514.939
on its training set. The token positions are disjoint; documents are not
necessarily disjoint. Reducing the number of retained PCA dimensions also changes
the fitted subspace, so the sweep measures this concrete reduction in degrees of
freedom rather than holding the original subspace fixed.

The original main plot is a correct description of training-set kurtosis.
For claims about sample efficiency or recovery, add a common independent
assessment corpus and report training and assessment curves together. A high
training curve alone should not be interpreted as a better ICA representation.

- [Diagnostic plot (PDF)](figures/ica-small-sample-debug.pdf)
- [Diagnostic plot (PNG)](figures/ica-small-sample-debug.png)
- [Full measurements](work/debug-small-sample/report.json)
- [Reproduction script](debug_small_sample.py)
- [Figure renderer](plot_small_sample_debug.py)
