# ICA Lens artifact format

## Scope

Format version 4 uses one Hugging Face Model repository for each model checkpoint. A
repository contains a manifest plus fitted ICA tensors for its available
layers. The format is identical when created with `save()`, stored in a local
directory, or uploaded with `push_to_hub()`.

A profiled artifact typically has this layout:

```text
README.md
icalens.json
artifacts/
└── resid_post/
    ├── layer_00.safetensors
    ├── layer_01.safetensors
    └── layer_06.safetensors
component_profiles/
└── resid_post/
    ├── layer_00.json.gz
    ├── layer_01.json.gz
    └── layer_06.json.gz
```

The activation-site directory is retained even when a release contains only
`resid_post`. This leaves room for future sites such as `mlp_out` or
`attn_out` without changing the format.

## Manifest

`icalens.json` is the machine-readable source of truth. It must include:

- The artifact format name and integer format version.
- The minimum ICA Lens package version required to read the artifact.
- The model repository ID, exact fitted checkpoint revision, and checkpoint type.
- The activation site and layer-indexing convention.
- The hidden size and input preprocessing steps.
- The available layers, tensor filename, and component count for each layer.
- Essential fitting provenance, including algorithm, dtype, and random seed.
- Per-iteration objective percentiles for parallel FastICA, from minimum through
  deciles to maximum.
- Final per-component contrast objectives and absolute deviations from the
  contrast's standard-Gaussian baseline.
- Dataset/token sampling provenance supplied by the fitter.
- The package version and post-ICA source-scaling policy.
- Optional per-layer component-profile paths and optional model-level R-lens
  provenance.

New v0.3 fits order component IDs by descending absolute contrast deviation
from the standard-Gaussian baseline. Thus `C0` is the most non-Gaussian
component under the fitted contrast. The manifest stores the Gaussian baseline
plus aligned `component_objectives` and `component_strengths`. This ranking is
local to one fitted layer and does not imply correspondence across layers or
artifacts. Older artifacts retain their recorded component-ID convention.

Example:

```json
{
  "format": "icalens",
  "format_version": 4,
  "minimum_package_version": "0.3.4",
  "package_version": "0.3.6",
  "model": {
    "repo_id": "openai-community/gpt2",
    "revision": "FULL_COMMIT_HASH",
    "type": "base"
  },
  "activation_site": "resid_post",
  "layer_indexing": "transformer_blocks_zero_based",
  "hidden_size": 768,
  "input_preprocessing": {
    "icalens_preprocessing": "l2",
    "row_normalization": "l2",
    "pre_normalization_center": "none",
    "norm_eps": 1e-12
  },
  "layers": {
    "6": {
      "file": "artifacts/resid_post/layer_06.safetensors",
      "n_components": 768,
      "component_profile": "component_profiles/resid_post/layer_06.json.gz",
      "fitting": {
        "algorithm": "fastica",
        "implementation": "icalens.torch",
        "random_state": 0,
        "source_scaling": "none"
      }
    }
  }
}
```

The example omits additional fitting diagnostics and provenance for brevity.
The precise model revision is part of compatibility validation. A model
name alone is not sufficient to identify the checkpoint that produced the
fitting activations. `model.type` is either `base` or `instruct`; it describes
the checkpoint rather than the input format. Format-version 1 manifests using
`base_model` remain readable and are interpreted as type `base`.

The current reader supports format versions 1 through 4. Version 3 requires
`minimum_package_version` `0.3.2`; version 4 requires `0.3.4`. New artifacts
are always written as version 4. Older formats remain readable for backward
compatibility, but this document describes the version-4 writing contract.

## Layer tensors

Layer files use Safetensors rather than pickle. The public artifact contract is
defined in terms of the operations needed by ICA Lens, not the attribute names
of the fitting library.

The tensor names are:

```text
center               [hidden_size]
reading_matrix       [n_components, hidden_size]
writing_matrix       [hidden_size, n_components]
preprocessing_center [hidden_size]  # present for geometric-median-l2
```

Conceptually, the operations are:

```python
processed = preprocess(activations)
scores = (processed - center) @ reading_matrix.T
reconstructed = scores @ writing_matrix.T + center
```

The final exporter and implementation must document the exact preprocessing
and equations. In particular, inverse transformation is approximate when input
preprocessing, such as row normalization, discards information.

## Loading and versioning

`ICALens.from_pretrained()` accepts either a Hugging Face Model repository ID
or a local directory. For a Hub repository, it loads `icalens.json` first and
may download individual layer files lazily. It accepts a `revision` argument so
users can pin a release tag or full commit hash.

Artifact releases should receive immutable tags such as `v0.3.6`. Readers must
reject unsupported `format_version` values with a clear compatibility error.

## Fitting and publishing

`ICALens.fit(activations, layer=...)` fits one layer at a time. Repeated calls
build a model-level collection; fitting an existing layer replaces that layer
only. All layers in a collection must have compatible model identity,
hidden size, activation site, and preprocessing settings.

Fitting must be reproducible when the same input, parameters, and
`random_state` are supplied. The manifest records the effective fitting
configuration and provenance. ICA Lens cannot verify that user-supplied
activations actually came from the declared model or layer, so this limitation
must be clear in the API and model card.

The v0.1 fitter is an internal PyTorch implementation of FastICA with
unit-variance whitening. It does not depend on scikit-learn or SciPy. NumPy
inputs default to CPU; callers can select a different fitting device. Mean,
covariance, fixed-point updates, and source variance are accumulated in
repeated batches, so the complete activation matrix need not be placed on the
GPU. The effective batch size, fitting device, and
`blockwise_multi_pass` memory strategy are recorded per layer.

New v0.2 fits store direct FastICA coordinates without a separate post-ICA
source-standard-deviation pass (`source_scaling: "none"`). For each token,
component energy share is `score² / sum(score²)`.

`save(path)` writes the manifest and layer files locally. It must use an atomic
or staging-directory strategy so an interrupted save does not leave an
apparently valid partial artifact.

`push_to_hub(repo_id, ...)` is the explicit network operation for publishing
the saved representation to a Hugging Face Model repository. It should support
at least `private`, `token`, `revision`, and `commit_message` options. It must
not overwrite unrelated remote files silently. Official `sida/icalens-*`
artifacts are generated with this same public fitting and publishing API rather
than a private exporter.

## Model card

The repository `README.md` is its Hugging Face model card. Its YAML metadata
should identify `icalens` as the library, name the analyzed model and license, and
include interpretability and ICA tags. The body should document the activation
site, layer convention, fitting data and method, intended use, limitations,
package version, and paper citation.
