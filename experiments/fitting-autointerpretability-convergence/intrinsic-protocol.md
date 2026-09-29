# Checkpoint-specific intrinsic-interpretability protocol

The primary convergence analysis measures each fitting checkpoint on its own
strongest occurrences. This avoids privileging the final checkpoint and avoids
penalizing a later direction merely because it no longer activates on an
early-checkpoint-specific fragment.

For every persistent component at every checkpoint, preparation:

1. ranks the 50,000 fragments by maximum positive token activation;
2. retains the strongest 25 fragments from distinct source documents;
3. uses a deterministic permutation, seeded by layer, persistent component ID,
   and iteration, to assign five fragments to explanation and twenty to held-out
   top validation; and
4. samples five positive random-validation fragments from additional distinct
   source documents.

The five explanation and twenty top-validation fragments are therefore
exchangeable samples from the same top-25 pool. Document-level separation avoids
near-duplicate leakage between explanation and validation. `top_score`, the
Pearson correlation pooled over the 20 × 64 held-out top-token positions, is the
primary statistic. The five random fragments retain the existing secondary
control but are not pooled into `top_score`.

The complete four-layer preparation is reproducible with:

```bash
uv run python experiments/fitting-autointerpretability-convergence/prepare_fixed_panel_run.py \
  --n-components 100
```

The historical script name and `runs/fixed-panel/` path refer to the unified
four-layer run container. The prepared selection itself is checkpoint-specific,
not a fixed fragment panel.

Preparation capacity and evaluation extent are separate. The cohort is a
prefix of a deterministic random permutation, so increasing the extent keeps
all earlier component IDs and positions. Evaluate any prefix one component at
a time with:

```bash
uv run python experiments/fitting-autointerpretability-convergence/evaluate.py \
  --layers 7,15,23,31 \
  --n-components 50 \
  --input experiments/fitting-autointerpretability-convergence/runs/fixed-panel/prepared \
  --output experiments/fitting-autointerpretability-convergence/runs/fixed-panel/evaluated-tinker
```

Rerunning that command later with `--n-components 100` reuses the first 50 and
continues with positions 51--100. The target cannot exceed the capacity passed
to `prepare_fixed_panel_run.py`.
