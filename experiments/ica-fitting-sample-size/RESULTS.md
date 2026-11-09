# GPT-2 ICA fitting versus token count

All five settings completed 50 fixed FastICA updates on two RTX A5000 GPUs.
The figure shows the median excess kurtosis across all 768 components after
GPT-2 block 11. Shading shows component quartiles. Every state (0–50) is measured
on the entire training subset for that setting, using raw residual activations
without row L2 normalization. Subsets are nested samples from one seeded random
permutation of a common 1M-token activation pool (seed 0).

| Fitting tokens | Step 0 median excess kurtosis | Step 50 median excess kurtosis | Step 50 mean excess kurtosis | Fit and recording time (s) |
| --- | ---: | ---: | ---: | ---: |
| 1K | 0.010 | 514.939 | 488.699 | 7.5 |
| 5K | 0.217 | 28.323 | 44.077 | 4.6 |
| 10K | 0.253 | 17.425 | 27.431 | 5.5 |
| 100K | 0.298 | 15.100 | 21.735 | 13.6 |
| 1M | 0.299 | 14.618 | 21.094 | 93.4 |

The 1K condition develops much larger training-set excess kurtosis than the
other settings. With 768 fitted components and only 1,000 token samples, this
is consistent with a strong finite-sample fitting effect; the training statistic
alone cannot establish better recovery or interpretation. The 100K and 1M final
medians differ by about 3.3%, and the corresponding curves are close. This is
one seed and one layer; it measures these fitted training distributions.

The final statistics were independently replayed over every training row. The
check also verified all 51 states, finite component statistics, exact budget
sizes, unique and nested token indices, and the final native Lens transforms.
The audit passed for all five settings.

- [Comparison PNG](figures/ica-fitting-sample-size.png)
- [Vector PDF](figures/ica-fitting-sample-size.pdf)
- [Figure data twin](figures/ica-fitting-sample-size.npz)
- [Raw per-step statistics and native lenses](work/runs/)
- [Audit and summary](work/runs/audit.json)

Model: `openai-community/gpt2`, revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. Corpus:
`NeelNanda/pile-10k`, with exact revisions and framing recorded in
`work/activations/gpt2-pile10k-layer11-1m/activations.json`. Activations were
captured from 1,877 documents using the documented capture CLI. Each native
Lens and each intermediate unmixing matrix is saved under its budget directory.
Fitting used float32, TF32 disabled, and blockwise float64 moment accumulation.

## Why 1K training kurtosis is high

A follow-up evaluation used 100K token positions excluded from every fit through 100K. These positions are held out at token level, not document level.

| Fitting tokens | Training median excess kurtosis | Unseen-token median excess kurtosis | Median top-token squared-energy share | Median top-token share of fourth moment |
| --- | ---: | ---: | ---: | ---: |
| 1K | 514.939 | 0.603 | 71.86% | 99.82% |
| 5K | 28.323 | 1.991 | 4.56% | 32.63% |
| 10K | 17.425 | 4.817 | 1.84% | 15.54% |
| 100K | 15.100 | 13.672 | 0.23% | 2.70% |

For a centered component, define p_i = s_i^2 / sum_j(s_j^2). Its empirical excess kurtosis is exactly N * sum_i(p_i^2) - 3. At N=1000 a single token carrying roughly 72% of squared energy contributes about 518 to ordinary kurtosis. The observed concentration and unseen-token result support fitting to isolated training examples as the explanation of the large 1K statistic.
