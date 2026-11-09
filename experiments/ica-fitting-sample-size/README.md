# ICA fitting versus activation sample size

Fits GPT-2 Small residual-stream activations with the same 50-step parallel
FastICA configuration at 1K, 5K, 10K, 100K, and 1M token activations. The default target
is residual stream after block 11 (`layer=11`), with all 768 dimensions.
Every run uses nested subsets from one seeded permutation of the common 1M-token
pool and the same FastICA initialization seed. At each
iteration, the current unmixing matrix is evaluated on all N rows used for
fitting. Stored diagnostics include per-component skewness and excess
kurtosis, their median/mean and quartiles, logcosh contrast, the FastICA update
limit and orthogonality error, plus fit settings and elapsed time. Initial state
0 and states after updates 1–50 are retained. Unmixing matrices at all states,
the whitening matrix, the mean center, sampled row indices, and the final native
ICA Lens are saved for each budget.

The output curve is median excess kurtosis across the 768 components; shading
is the component interquartile range. This tracks non-Gaussianity over fitting
steps without hiding component spread. It is a diagnostic of the current
whitened FastICA sources, not a claim that every component should have positive
kurtosis.

Capture uses the documented `openai-community/gpt2` model and
`NeelNanda/pile-10k` training split. Capture 1M token activations once at block
11, then fit nested random subsets at all five budgets. The bfloat16 activation
file occupies 1.536 GB, plus metadata and model/data caches; current scratch
space is sufficient.

The batch files target the `class` partition RTX A4000 profile. Submit from
this repository root:

```bash
mkdir -p experiments/ica-fitting-sample-size/logs
CAPTURE_JOB=$(sbatch --parsable experiments/ica-fitting-sample-size/capture.sbatch)
FIT_JOB=$(sbatch --parsable --dependency=afterok:${CAPTURE_JOB} --array=0-4%4 \
  experiments/ica-fitting-sample-size/fit_array.sbatch
)
sbatch --dependency=afterok:${FIT_JOB} \
  experiments/ica-fitting-sample-size/plot.sbatch
```

The five budgets are independent GPU array tasks and run up to four at a time.
The plot job runs only after every fit task succeeds.

For a short run inside an allocation with two visible GPUs, run
`bash experiments/ica-fitting-sample-size/fit_parallel.sh` after capture.
It uses one GPU for the 1M fit and the other for the four smaller fits.
The PDF, PNG, NPZ data twin, and caption are written to `figures/`.

Runs resume only at the whole-budget level (completed runs are skipped); the
iteration trace is atomically replaced after each callback. Do not launch
two workers for the same budget and output directory simultaneously.

The 1K result is investigated in [DEBUG-1K.md](DEBUG-1K.md). Reproduce its
short GPU controls with `debug_small_sample.py`; render their saved measurements
with `plot_small_sample_debug.py`. Controls include independent SciPy moments,
float64 projections and fitting, scikit-learn FastICA with SVD whitening,
three Gaussian draws, and a retained-component sweep at fixed 1K tokens.

For assessment on an additional 1M tokens from new documents, run
`capture_extra_tokens.py` inside a GPU allocation. It replays the original
training token identities, excludes all original document rows and exact-text
duplicates, and writes the source-selection audit. Then run
`evaluate_extra_parallel.sh` with two visible GPUs to evaluate all 255 saved
states. `audit_extra_tokens.py` independently checks the final moments and
standardized contrast with SciPy/NumPy on the full 1M-token assessment set for
eight recorded components per fit. `plot_extra_tokens.py` renders the training
versus new-document kurtosis comparison and the additional non-Gaussianity
metrics. No transform is refitted. New same-Pile documents constitute held-out
generalization; they are not a cross-domain OOD corpus.

