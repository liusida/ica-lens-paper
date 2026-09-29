# ICA--SAE efficiency measurement

This experiment compares the current native ICA representation with the public
SAE registered for each model. It measures one middle layer per model because
tensor dimensions—and therefore persistent parameter memory—are invariant over
the compared layers. Timing uses identical synthetic CUDA inputs for both
methods, excludes model inference, warms up each operation, synchronizes CUDA,
and reports the median batch-amortized time per activation.

The primary quantities corresponding to the paper's earlier efficiency figure
are `parameter_bytes`, encode latency, and full encode--decode (`forward`)
latency. Decode-only latency and peak temporary CUDA allocation are retained in
the result for diagnosis.

The companion plot adds measured ICA fitting cost as a fourth panel. Its ranges
are direct wall-time measurements from two complete one-million-token,
50-iteration fits per model. The selected public SAE releases do not report
comparable checkpoint-specific per-layer wall times, so the plot deliberately
does not invent or display SAE fitting-time bars.

```bash
uv run python experiments/efficiency-measurement/run.py
uv run python experiments/efficiency-measurement/measure_fitting_time.py
uv run python experiments/efficiency-measurement/plot.py
```

`measure_fitting_time.py` runs the production activation-fitting CLI on two
middle layers per model and records actual wall time. It reads the existing
one-million-token activation caches and is resumable at the completed-layer
boundary. It deliberately does not infer fitting duration from artifact file
timestamps.
