# Generalization of GPT-2 ICA checkpoints to an additional 1M tokens

All five fitting budgets (1K, 5K, 10K, 100K, 1M) were evaluated at every saved
state, including initialization and updates 1–50: 255 states in total. Every
state is evaluated on every one of the same additional 1,000,000 token positions
and all 768 components. The underlying GPT-2 layer is block 11 (`resid_post`).
The original fitted directions, center, and whitening are unchanged.

## Assessment-set construction

The original 1M-token training pool was reconstructed from the pinned
`NeelNanda/pile-10k` revision. Its full token-ID sequence matched the saved
training metadata exactly. All 1,877 original source documents were excluded.
The next eligible documents were selected after excluding exact-text hashes
seen in training or already included in assessment. The new 1M token positions
come from 1,852 source documents. Both source-document overlap and exact-text
hash overlap are zero. Raw source records, source-row identities, hashes,
model/dataset revisions, framing, and counts are retained in the selection
audit and the separate ActivationDataset manifest.

This is document-disjoint, same-corpus held-out assessment. It does not make a
cross-domain OOD claim. Repeated vocabulary tokens across corpora are expected;
what is new is each token occurrence and its document context.

## Final-state comparison

All kurtosis values below are medians across components of population central
excess kurtosis `m4/m2^2 - 3`, evaluated over the entire stated token set.

| Fitting tokens | Training excess kurtosis | New 1M-token excess kurtosis | New standardized logcosh deviation | New absolute skewness |
| --- | ---: | ---: | ---: | ---: |
| 1K | 514.9386 | 0.5314 | 0.007421 | 0.0702 |
| 5K | 28.3228 | 1.3689 | 0.012431 | 0.2415 |
| 10K | 17.4246 | 3.1982 | 0.019803 | 0.4719 |
| 100K | 15.1000 | 11.0410 | 0.042866 | 1.3947 |
| 1M | 14.6179 | 11.8302 | 0.046205 | 1.5377 |

The 1K fit's training statistic of about 515 falls to about 0.53 on new
same-corpus documents. Increasing the fitting budget yields higher new-document
non-Gaussianity in this run; 100K and 1M are much closer to each other than to
1K/5K/10K. The same ordering appears in standardized logcosh deviation and
absolute skewness. These are one-seed, one-layer results; a larger training
statistic alone does not establish better recovery.

For logcosh, raw scores and scores standardized by their assessment mean and
variance are both stored. The plotted deviation uses the standardized score
and the standard-Gaussian logcosh expectation, so differences in score variance
do not masquerade as non-Gaussianity. This is moment/contrast evaluation only;
no directions or whitening are estimated anew from assessment data.

## Verification and outputs

The audit checked the 1M-token count, 51 states per fit, finite statistics for
all 768 components, shared assessment-manifest identity, and all original
checkpoint hashes. For each final checkpoint, eight recorded components were
independently projected over the entire new 1M-token set and checked against
SciPy kurtosis/skewness and NumPy standardized logcosh. All checks passed.

Bands in the figures show quartiles across components, not uncertainty across
seeds. The training/new-token kurtosis comparison shares symmetric-log y axes
with a linear interval from -1 to 1. All curves show measured states; none are
extrapolated.

- [Training versus new-token kurtosis (PDF)](figures/ica-fitting-extra-token-kurtosis.pdf)
- [Training versus new-token kurtosis (PNG)](figures/ica-fitting-extra-token-kurtosis.png)
- [New-token logcosh and skewness (PDF)](figures/ica-fitting-extra-token-nongaussianity.pdf)
- [New-token logcosh and skewness (PNG)](figures/ica-fitting-extra-token-nongaussianity.png)
- [Selection audit](work/extra-evaluation-selection.json)
- [Measurement audit](work/extra-evaluation/audit.json)
- [Raw per-component trajectories](work/extra-evaluation/)

Each figure has a same-stem NPZ data twin and TXT caption. To reproduce the
assessment inside GPU allocations, run `capture_extra_tokens.py`, then
`evaluate_extra_parallel.sh` (two GPUs), `audit_extra_tokens.py`, and
`plot_extra_tokens.py`. All capture and evaluation jobs for this extension
completed within a short interactive allocation.
