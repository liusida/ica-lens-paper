# ICA Lens API notes

## v0.2 interface

Version 0.2 adds explicit base-versus-instruct checkpoint metadata while
retaining the activation-level fitting and transformation interface.

```bash
pip install icalens
```

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")

scores = lens.transform(
    activations,
    layer=6,
)

reconstructed_activations = lens.inverse_transform(
    scores,
    layer=6,
)
```

Users can fit a lens one layer at a time:

```python
from icalens import ICALens

lens = ICALens(
    model_id="openai-community/gpt2",
    model_type="base",
    activation_site="resid_post",
)

lens.fit(
    activations,
    layer=6,
    random_state=0,
    device="cuda",
    batch_size=8192,
    progress=True,
)

lens.save("./icalens-gpt2-small")
lens.push_to_hub("username/icalens-gpt2-small")
```

`model_revision` is optional. Pass an exact commit when fitting activations
captured outside ICA Lens and checkpoint provenance matters. When `capture()`
or `analyze()` loads a model, ICA Lens resolves and records its current commit
automatically.

Calling `fit()` again with another layer adds or replaces that layer in the
same model-level collection. `fit()` returns `self`. The input is an activation
array whose final dimension is the model's hidden size; leading dimensions
are treated as samples.

Instruction-tuned checkpoints are declared explicitly:

```python
lens = ICALens(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    model_type="instruct",
)
```

`model_type` is checkpoint provenance, not an ICA algorithm choice. It accepts
`"base"` and `"instruct"`. Chat templating, conversation datasets, and token
scope are input-pipeline metadata and must not be inferred from this value.
The old `base_model` and `base_model_revision` constructor names remain
compatibility aliases for v0.1 callers, and v0.1 artifacts remain loadable.

Fitting is blockwise and multi-pass. `device` selects where whitening and
FastICA calculations run, while the activation tensor may remain in CPU
memory. `batch_size` bounds the number of activation rows moved to the fitting
device at once. GPU memory therefore scales primarily with batch size rather
than the total token count. The input must remain available for repeated passes
through mean estimation, covariance accumulation, and fixed-point iterations.

`save()` writes the standard artifact layout to a local directory. Network
upload is a separate, explicit operation: `push_to_hub()` creates or updates a
Hugging Face Model repository using that same layout. The official
`sida/icalens-*` artifacts should be produced through these public methods.

The argument to `ICALens.from_pretrained()` is a Hugging Face Model repository
ID. Each repository contains the fitted ICA artifacts for one model and
may contain multiple layers. `transform()` selects the layer to apply.

`from_pretrained()` should load the repository manifest first. Layer tensor
files may then be downloaded lazily when a layer is used. Hugging Face's cache
is used by default. The same method should accept a local directory with the
same artifact structure.

A `revision` argument should allow callers to pin a Hub tag or commit:

```python
lens = ICALens.from_pretrained(
    "sida/icalens-gpt2-small-pile10k",
    revision="COMMIT_OR_TAG",
)
```

The v0.1 interface should support:

- Fitting an ICA Lens for one or more layers from user-supplied activations.
- Saving fitted artifacts in the standard local format.
- Creating or updating a user-owned Hugging Face Model repository.
- Loading and caching ICA Lens artifacts from Hugging Face.
- Inspecting available layers, activation sites, model identity and type, and fitting metadata.
- Validating activation shape and artifact compatibility.
- Transforming hidden-state activations into ICA component scores.
- Approximately mapping component scores back into activation space.
- Loading the same artifact format from either the Hugging Face Hub or a local
  directory.

The low-level fitting API remains activation-only and cannot verify that the
caller captured the declared model, layer, or site. The optional
The high-level integration included with `icalens` handles model loading and
capture for analysis.

An analysis result can be exported without retaining the model or tokenizer:

```python
result = lens.analyze("She deposited the check at the bank.", layer=6)
result.to_html("analysis.html", metric="score", top_k=5)
```

The generated HTML is self-contained and can be opened directly from disk.
The following are intentionally deferred:

- In-model interventions and paper-reproduction workflows.

The fitting API does not capture activations or infer whether they came from
the declared model, revision, layer, or activation site. The caller supplies
that provenance, and ICA Lens records it in the manifest.

Fitting uses the package's internal PyTorch FastICA implementation. NumPy
inputs fit on CPU, while PyTorch inputs fit on their current device. ICA Lens
does not depend on scikit-learn or SciPy.

The optional high-level API builds on the same artifact:

```python
result = lens.analyze(
    "She deposited the check at the bank.",
    layer=6,
)
```

The standard installation includes this high-level interface. The result
contains aligned tokens, token IDs, positions, activations, signed ICA
scores, and per-token energy shares. Energy is `score² / sum(score²)` across
components. New v0.2 fits use direct FastICA coordinates without post-ICA
source scaling. `fit(..., provenance={{...}})` records JSON-compatible dataset
and sampling metadata in each layer artifact.

Starting in v0.3, Qwen3.5 multimodal checkpoints are supported for text and conversation
analysis. Automatic loading through `AutoModelForCausalLM` selects the language
backbone and avoids materializing the vision encoder. When a caller supplies a
full conditional-generation model, ICA Lens locates transformer blocks under
`model.language_model.layers` and captures the same language `resid_post` site.
