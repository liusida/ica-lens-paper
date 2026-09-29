# FastICA limit and fit-quality trajectory

This experiment repeats the four Qwen 3.5 9B Base FastICA trajectories used by the
fitting--autointerpretability convergence experiment.  It records the full
component population at every iteration:

- per-component FastICA update limit;
- per-component raw Logcosh objective and absolute contrast from the Gaussian
  reference; and
- per-component excess kurtosis.

The purpose is to test whether the usual maximum update limit is a useful
stopping criterion for LLM analysis.  The maximum describes only the single
least-stable component, while both maximum and mean limits describe change in
the fitted directions rather than the absolute non-Gaussian quality attained.

Run the measurement (repeating the command reuses completed layers):

```bash
uv run python experiments/record-mean-lim/run.py
```

Then render the pilot figure:

```bash
uv run python experiments/record-mean-lim/plot.py
```

Each layer is one durable unit.  An interrupted layer is recomputed; completed
layer results are validated and reused.
