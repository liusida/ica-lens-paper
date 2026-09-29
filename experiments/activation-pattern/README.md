# Tokenwise activation patterns

The measurement caches the union of the top five ICA components (ranked by
absolute score) and positive SAE features (ranked by activation) at every token.
One model pass therefore supports cumulative top-1 through top-5 plots.

```bash
uv run python experiments/activation-pattern/run.py \
  --output experiments/activation-pattern/results-top5

uv run python experiments/activation-pattern/plot.py \
  --results experiments/activation-pattern/results-top5 \
  --force
```

The renderer writes `activation-pattern-<model>-top-<n>.png` for each model and
each cumulative rank threshold from 1 through 5.
