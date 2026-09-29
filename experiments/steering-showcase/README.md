# ICA steering showcase

This experiment reproduces three qualitative ICA steering demonstrations:

- GPT-2 Small: steer a weekend continuation from going to a bar toward playing
  with children.
- Gemma 2 2B: steer English to Chinese and ablate the same component to shift
  Chinese back toward English.
- Qwen 3.5 9B Base: steer English to French and ablate the same component to
  shift French back toward English.

Gemma and Qwen components are recovered by reproducible language-contrast
searches. The GPT-2 component comes from the hand-made weekend demonstration and
does not require a language search.

## Reproduce from scratch

Remove or move aside any existing folders under `runs/`, then run the two searches:

```bash
uv run python experiments/steering-showcase/search.py \
  --models gemma2 \
  --layers gemma2=18-22 \
  --target-language chinese \
  --output experiments/steering-showcase/runs/component-search-chinese-gemma2

uv run python experiments/steering-showcase/search.py \
  --models qwen9b \
  --layers qwen9b=23-27 \
  --target-language french \
  --output experiments/steering-showcase/runs/component-search-french-qwen9b
```

These searches reproduce Gemma L20 C105 as the strongest Chinese–English
component and Qwen L26 C9 as the strongest French–English component. Each search
samples 1,000 ManyThings translation pairs with seed 0, measures every ICA
component at the last non-padding text token, and ranks components by the absolute
target-minus-English mean-score difference.

Then reproduce all three demonstrations in one run:

```bash
uv run python experiments/steering-showcase/steer.py --models all
```

The resulting run layout is:

```text
runs/
├── component-search-chinese-gemma2/
├── component-search-french-qwen9b/
└── steering-demonstrations-model-framing-all/
```

The corpus archives are downloaded into `data/` on first use and reused later.
Corpus ZIPs and run outputs are excluded from Git. Search results are checkpointed
per layer; steering results are checkpointed per model, so repeating an interrupted
command reuses completed units.

## Run the matched steering demonstrations

The demonstration script mirrors the final hand-made notebook examples:

- `misc/language-steering-gpt.ipynb`: GPT-2 L7 C317, using a strong initial
  intervention to change “go to a local bar” into “play with the kids.”
- `demo/language-steering.ipynb`: Gemma L20 C105, English–Chinese steering and
  ablation.
- `misc/language-steering-qwen.ipynb`: Qwen L26 C9, English–French steering and
  ablation.

Models are loaded sequentially. Use `--models gpt2`, `gemma2`, or `qwen9b` to run
one model, and `--debug` to print token-aligned component scores. All generations
use `document_framing="model"`, matching each tokenizer's ordinary direct-generation
behavior: Gemma receives its normal BOS, while GPT-2 and Qwen receive no extra prefix.
