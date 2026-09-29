# ICA–SAE directional overlap

This experiment compares each ICA **writing direction** with all SAE decoder
directions from the same model and transformer layer.  It records the largest
absolute cosine similarity and the index of the nearest SAE direction.

For each layer, it also samples an isotropic random dictionary containing the
same number of directions as the ICA basis and measures those directions
against the same SAE decoder dictionary. This matched null controls the model's
residual dimension, the SAE dictionary width and geometry, and the number of
query directions. Random dictionaries are reproducible from `--random-seed`
(default 0) and a stable model-layer-specific derived seed.

Writing directions are used because both they and SAE decoder rows live in the
model residual space and describe how a feature maps back into that space.
ICA reading vectors are not interchangeable with decoder directions for a
non-orthogonal basis.

Run all registered paper models:

```bash
uv run python experiments/ica-sae-directional-overlap/run.py
```

Run one model or selected layers:

```bash
uv run python experiments/ica-sae-directional-overlap/run.py \
  --models gpt2 --layers 0,6,11
```

Outputs are written under `results/`.  Each completed layer has a compressed
NPZ checkpoint containing nearest-SAE cosines and SAE feature IDs for both the
ICA and matched-random directions. The script also writes aggregate JSON/CSV
summaries and diagnostic PDF/PNG figures.

The output directory identifies a resumable run. Repeating the same command
validates its configuration and dependency fingerprints, skips valid completed
model-layer checkpoints, and continues at the first missing layer. A changed
model selection, layer selection, Lens artifact, SAE release, or measurement
definition is rejected instead of being mixed into an existing output. Use
`--force` only to deliberately recompute the same compatible set of layers.
The durable and user-facing progress unit is one model-layer comparison.

Nearest-neighbor cosine depends on SAE dictionary width: a larger dictionary
offers more chances for a high accidental match. The matched random reference
therefore provides the relevant within-model, within-layer baseline; raw
nearest cosines should not be compared across models as width-controlled
scores.
