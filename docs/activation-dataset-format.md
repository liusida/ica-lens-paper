# ActivationDataset format

`ActivationDataset` is the portable, disk-backed format consumed by
`icalens fit activations`. An external inference system can write this format directly; it does
not need to use ICA Lens for model loading or activation capture.

This page specifies format version 1, identified by:

```json
{
  "format": "icalens.activations",
  "format_version": 1
}
```

## Directory layout

```text
my-activations/
├── activations.json
├── samples.safetensors
└── layers/
    ├── layer_00.bfloat16.bin
    ├── layer_01.bfloat16.bin
    └── layer_02.bfloat16.bin
```

File names under `layers/` are chosen by the manifest and need not follow the example naming
pattern. Paths in the manifest are relative to the dataset directory.

## Activation values

Capture the post-block residual stream (`resid_post`) at zero-based transformer-block indices.
For each layer, write one contiguous, headerless, row-major tensor with shape:

```text
[sample_count, hidden_size]
```

The supported element types are `bfloat16`, `float16`, `float32`, and `float64`. ICA Lens's own
capture commands use `bfloat16`. The raw file must contain exactly
`sample_count * hidden_size` elements in the dtype declared by the manifest.

Row `i` must represent the same sampled token in every layer file and in row `i` of every tensor
in `samples.safetensors`. The sampling order itself is unrestricted as long as this alignment is
preserved.

`resid_post` means the transformer block output before the model's final normalization. Do not
capture the embedding output as layer 0: layer 0 is the output of transformer block 0.

## `samples.safetensors`

The sample metadata is a Safetensors file containing three required one-dimensional `int64`
tensors of length `sample_count`:

| Tensor | Meaning |
| --- | --- |
| `document_index` | Zero-based index of the source document or rendered conversation in the capture stream |
| `position` | Zero-based token position in that document's complete model input |
| `token_id` | Tokenizer ID at that position |

`position` refers to the input actually passed to the model, including any native chat-template,
BOS, EOS, or other framing tokens. A context token that is excluded from sampling remains in the
model input but has no activation row and therefore no entry in this file.

Additional aligned tensors are permitted. Readers must not depend on them unless separately
documented.

## `activations.json`

The manifest must be named `activations.json`. A complete three-layer example is:

```json
{
  "activation_site": "resid_post",
  "dtype": "bfloat16",
  "format": "icalens.activations",
  "format_version": 1,
  "hidden_size": 3072,
  "layer_indexing": "transformer_blocks_zero_based",
  "layers": {
    "0": {
      "file": "layers/layer_00.bfloat16.bin",
      "shape": [1000000, 3072],
      "status": "complete"
    },
    "23": {
      "file": "layers/layer_23.bfloat16.bin",
      "shape": [1000000, 3072],
      "status": "complete"
    },
    "47": {
      "file": "layers/layer_47.bfloat16.bin",
      "shape": [1000000, 3072],
      "status": "complete"
    }
  },
  "model": {
    "repo_id": "Qwen/Qwen3.5-122B-A10B",
    "revision": "FULL_COMMIT_HASH",
    "type": "instruct"
  },
  "provenance": {
    "dataset": {
      "description": "Neuronpedia fitting corpus"
    },
    "fitting_tokens": 1000000,
    "sampling_seed": 0,
    "token_scope": "all"
  },
  "sample_count": 1000000,
  "samples_file": "samples.safetensors",
  "status": "complete"
}
```

The fields have these requirements:

| Field | Requirement |
| --- | --- |
| `format` | Must be `icalens.activations` |
| `format_version` | Must be `1` |
| `status` | Must be `complete` before fitting |
| `model.repo_id` | Model identifier used in the resulting Lens |
| `model.revision` | Exact model revision; a full commit hash is recommended |
| `model.type` | `base` or `instruct` |
| `activation_site` | Must be `resid_post` for the standard ICA Lens workflow |
| `layer_indexing` | Use `transformer_blocks_zero_based` |
| `sample_count` | Number of activation rows in every captured layer |
| `hidden_size` | Number of values in every activation row |
| `dtype` | One of `bfloat16`, `float16`, `float32`, or `float64` |
| `samples_file` | Relative path to the sample-metadata Safetensors file |
| `layers` | Object keyed by decimal, zero-based layer number |
| `layers.*.file` | Relative path to the layer's raw binary tensor |
| `layers.*.shape` | `[sample_count, hidden_size]` |
| `layers.*.status` | Must be `complete` |
| `provenance` | JSON object describing the source data, selection, token framing, and sampling |

The provenance object is intentionally extensible. Record enough information to reproduce the
model input and sampled rows, including the dataset identity and revision, input field, chat
template or document framing, token scope, context length, sampling policy, token budget, and
seed where applicable.

## Validate and fit

Load the directory once before starting a long fit:

```python
from icalens import ActivationDataset

dataset = ActivationDataset("/path/to/my-activations")
print(dataset.available_layers)
print(dataset.sample_count, dataset.hidden_size, dataset.dtype)
print(dataset.layer(dataset.available_layers[0]).shape)
print(dataset.samples().keys())
```

This checks the format version, completion status, supported dtype, required sample metadata,
metadata lengths, and layer-file size when a layer is opened. Then fit without loading the
language model:

```bash
icalens fit activations \
  --input /path/to/my-activations \
  --layers all \
  --icalens-preprocessing none \
  --max-iter 50 \
  --fit-batch-size 16384 \
  --output /path/to/icalens-model
```

The fit memory-maps one layer at a time. At least `hidden_size + 1` activation rows are required
for a full-width Lens; production releases normally use substantially more samples.
