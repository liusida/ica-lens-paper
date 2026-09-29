# Python API

The public package interface consists of `ICALens`, `CaptureResult`,
`AnalysisResult`, and the package exceptions.

```python
from icalens import ICALens
```

## Create or load a lens

### `ICALens(...)`

Create an empty lens before fitting activation tensors:

```python
lens = ICALens(
    model_id="openai-community/gpt2",
    model_type="base",
    activation_site="resid_post",
    icalens_preprocessing="none",
)
```

```python
ICALens(
    *,
    model_id=None,
    model_revision=None,
    model_type="base",
    activation_site="resid_post",
    layer_indexing="hidden_states",
    row_normalize=None,
    icalens_preprocessing=None,
    norm_eps=1e-12,
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `model_id` | `str`, required | Hugging Face repository of the language model being analyzed |
| `model_revision` | `str \| None = None` | Exact language-model revision; resolved on first capture if omitted |
| `model_type` | `"base" \| "instruct" = "base"` | Whether the checkpoint is a base or instruction/chat model |
| `activation_site` | `str = "resid_post"` | Named activation location; the built-in capture path supports post-block residual streams |
| `layer_indexing` | `str = "hidden_states"` | Convention recorded in the artifact for interpreting layer numbers |
| `icalens_preprocessing` | `"none" \| "l2" \| "geometric-median-l2" \| None = None` | Transform applied before standard FastICA centering and whitening; `None` preserves the legacy L2 default |
| `row_normalize` | `bool \| None = None` | Compatibility option; prefer `icalens_preprocessing` in new code |
| `norm_eps` | `float = 1e-12` | Numerical floor used during normalization |
| **Returns** | `ICALens` | A new, unfitted lens |

`icalens_preprocessing="none"` passes raw activations to standard FastICA
centering and whitening. `"l2"` first normalizes every token activation to unit
length. `"geometric-median-l2"` subtracts a robust center before that L2
normalization. The selected mode is stored in the artifact and automatically
reused by `transform()`, `inverse_transform()`, and `analyze()`.

`model_revision` identifies the analyzed language-model weights. It is
different from the `revision` argument of `from_pretrained()`, which selects a
branch, tag, or commit of the **ICA Lens repository**.

### `ICALens.from_pretrained(...)`

Load a published lens:

```python
lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
```

Or load a local artifact directory:

```python
lens = ICALens.from_pretrained("icalens-output/my-icalens")
```

```python
ICALens.from_pretrained(
    repo_id_or_path,
    *,
    revision=None,
    cache_dir=None,
    token=None,
    local_files_only=False,
    force_download=False,
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `repo_id_or_path` | `str \| Path`, required | Hugging Face Model repository or local artifact directory |
| `revision` | `str \| None = None` | Branch, tag, or commit of the ICA Lens repository |
| `cache_dir` | `str \| Path \| None = None` | Optional Hugging Face cache directory |
| `token` | `str \| bool \| None = None` | Hugging Face authentication setting |
| `local_files_only` | `bool = False` | Refuse network access and use cached files only |
| `force_download` | `bool = False` | Download the manifest and lazy layer files again |
| **Returns** | `ICALens` | Lens initialized from the artifact manifest |

For a Hub repository, only `icalens.json` is fetched initially. The tensor file
for a fitted layer is downloaded when that layer is first used. `revision`
selects the lens artifact revision; `token` accepts a Hugging Face token or the
standard `huggingface_hub` boolean token setting.

## Analyze model inputs

### `analyze(...)`

Capture activations and immediately transform them into ICA scores and energy
shares:

```python
result = lens.analyze(
    "She deposited the check at the bank.",
    layer=6,
)
```

For an instruction model, pass a completed conversation:

```python
result = lens.analyze(
    [
        {"role": "user", "content": "Name an interesting science."},
        {"role": "assistant", "content": "Physics."},
    ],
    layer=16,
    token_scope="all",
)
```

```python
lens.analyze(
    inputs,
    *,
    layer,
    selected_components=None,
    model=None,
    tokenizer=None,
    token_scope="all",
    context_length=None,
    device="auto",
    verbose=False,
) -> AnalysisResult
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `inputs` | `str \| list[dict[str, str]]`, required | Raw text or a completed `{role, content}` conversation |
| `layer` | `int`, required | Zero-based transformer-block index stored in the lens |
| `selected_components` | `int \| list[int] \| tuple[int, ...] \| None = None` | Component to highlight automatically when the result is displayed; the current explorer accepts one component |
| `model` | `torch.nn.Module \| None = None` | Optional caller-supplied language model |
| `tokenizer` | tokenizer or `None` | Optional caller-supplied tokenizer |
| `token_scope` | `"all" \| "content" \| "user" \| "assistant" = "all"` | Conversation positions to return; raw text always returns all encoded tokens |
| `context_length` | `int \| None = None` | Optional maximum encoded input length |
| `device` | `str \| torch.device \| None = "auto"` | Model device; automatic mode prefers CUDA and otherwise uses CPU |
| `verbose` | `bool = False` | Print timed progress for model loading, activation capture, and score computation |
| **Returns** | `AnalysisResult` | Tokens, activations, signed scores, and energy shares |

The first call lazily loads the language model and tokenizer. Subsequent calls
on the same lens reuse them. For large models, pass `verbose=True` to make the
loading, capture, and transformation stages visible.

### `generate(...)`

Generate a continuation, optionally adding offsets to or clamping signed ICA
coordinates:

```python
baseline = lens.generate(messages, max_new_tokens=16)

steered = lens.generate(
    messages,
    layer=5,
    steer=(188, -12.0),
    max_new_tokens=16,
)
```

```python
lens.generate(
    prompt,
    *,
    layer=None,
    clamp=None,
    steer=None,
    steering_scope="current-position",
    max_new_tokens=64,
    device="auto",
    model=None,
    tokenizer=None,
    **generation_kwargs,
) -> str
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `prompt` | `str \| list[dict[str, str]]`, required | Raw prompt or chat messages; chat templates are applied automatically |
| `layer` | `int \| None = None` | Zero-based residual-stream layer to edit; required with `clamp` or `steer` |
| `clamp` | `tuple[int, float] \| Mapping[int, float] \| None = None` | One `(component, target_score)` pair or a mapping of simultaneous clamps, applied at every processed token position and generation step |
| `steer` | `tuple[int, float] \| Mapping[int, float] \| None = None` | One `(component, score_offset)` pair or a mapping of simultaneous additive score offsets; mutually exclusive with `clamp` |
| `steering_scope` | `"current-position" \| "all-positions" = "current-position"` | With `steer`, edit only the runtime current position, or every position processed during prefill and decoding |
| `max_new_tokens` | `int = 64` | Maximum number of continuation tokens |
| `device` | `str \| torch.device \| None = "auto"` | Model device; automatic mode prefers CUDA and otherwise uses CPU |
| `model` | `torch.nn.Module \| None = None` | Optional caller-supplied language model |
| `tokenizer` | tokenizer or `None` | Optional caller-supplied tokenizer |
| `**generation_kwargs` | keyword arguments | Additional arguments forwarded to `model.generate()` |
| **Returns** | `str` | Decoded continuation only, without the prompt |

Without `clamp` or `steer`, this is ordinary generation. A clamp replaces an
absolute score and restores the activation norm. Additive steering adds
`score_offset * writing_direction` directly to the residual stream. It requires
an unnormalized Lens without a preprocessing center. Greedy decoding is the
default; pass standard generation arguments to choose another decoding strategy.
The language model is loaded lazily and reused by later calls on the same lens.

### `capture(...)`

Use `capture()` when you need aligned activations without transforming them:

```python
captured = lens.capture("She deposited the check.", layer=6)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `inputs` | `str \| list[dict[str, str]]`, required | Raw text or a completed `{role, content}` conversation |
| `layer` | `int`, required | Zero-based transformer-block index to capture |
| `model` | `torch.nn.Module \| None = None` | Optional caller-supplied language model |
| `tokenizer` | tokenizer or `None` | Optional caller-supplied tokenizer |
| `token_scope` | `"all" \| "content" \| "user" \| "assistant" = "all"` | Conversation positions included in the result |
| `context_length` | `int \| None = None` | Optional maximum encoded input length |
| `device` | `str \| torch.device \| None = "auto"` | Model device |
| **Returns** | `CaptureResult` | Tokens, positions, IDs, and aligned activations |

### `unload_model()`

```python
lens.unload_model()
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| **Returns** | `None` | Releases the model and tokenizer cached by `capture()`, `analyze()`, or `generate()` |

This does not unload the fitted ICA layer matrices.

## Profile fitted components

### `profile_components(...)`

Profile one fitted layer from a stream of raw texts or completed conversations:

```python
profile = lens.profile_components(
    texts_or_conversations,
    layer=5,
    token_scope="all",
    max_tokens=100000,
    top_k_examples=20,
    provenance={"dataset": {"repo_id": "owner/dataset", "split": "train"}},
    device="auto",
    progress=True,
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `inputs` | iterable of text or conversations, required | Inputs streamed through the analyzed model |
| `layer` | `int`, required | Fitted layer to profile |
| `token_scope` | `str = "all"` | Eligible chat positions; raw text always uses all tokens |
| `max_tokens` | `int \| None = 100000` | Maximum token positions included; `None` consumes the iterable |
| `top_k_examples` | `int = 20` | Largest absolute-score occurrences retained on the selected tail |
| `logit_lens_top_k` | `int = 20` | Top and bottom vocabulary entries retained for each direction |
| `logit_lens_batch_size` | `int = 64` | Writing directions unembedded together; lower this to reduce peak memory |
| `r_lens` | `str \| Path \| dict \| None = None` | Compatible R-lens artifact used to add downstream-aware vocabulary readouts |
| `r_lens_top_k` | `int = 20` | R-lens vocabulary entries retained per direction |
| `r_lens_batch_size` | `int = 8` | R-lens directions processed together |
| `provenance` | `dict \| None = None` | JSON-compatible profiling provenance |
| `context_length` | `int \| None = 1024` | Maximum encoded length of each input |
| `device` | `str \| torch.device \| None = "auto"` | Language-model device |
| `progress` | `bool = False` | Display streaming progress |
| **Returns** | `dict` | Complete per-layer component profile, also attached to the lens |

This is a post-fit operation: it does not alter the fitted center, reading
matrix, or writing matrix. The returned layer profile contains `n_tokens`,
`n_inputs`, profiling `selection` and `provenance`, and a `components` list.
Each component entry contains:

| Field | Contents |
| --- | --- |
| `component` | Component ID |
| `tail_direction` | Tail selected by the sign of population skewness |
| `dominant_sign` | Compatibility alias for `tail_direction` |
| `sign_statistics` | Positive/negative position fractions and positive/negative energy fractions |
| `score_statistics` | Mean, variance, third central moment, skewness, excess kurtosis, and kurtosis rank |
| `examples` | Top-score occurrences on the selected tail and their token counts |
| `logit_lens` | Vocabulary associations for the positive, negative, and dominant writing directions |
| `r_lens` | Optional downstream-aware vocabulary associations when a compatible R-lens was supplied |

Call `save()` afterward to persist all attached profiles. When processing many
layers, use `checkpoint_component_profile()` after each layer so completed
profiles survive an interruption.

### `refresh_profile_statistics_from_activations(...)`

Recompute distribution statistics and skewness-based tail selection from
previously captured activations without rebuilding the rest of the profile:

```python
profile = lens.refresh_profile_statistics_from_activations(
    activations,
    layer=6,
    rows=None,
    batch_size=8192,
    provenance={"source": "saved activations"},
    device="auto",
    progress=True,
)
lens.checkpoint_component_profile("icalens-output/my-icalens", layer=6)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `activations` | `torch.Tensor`, required | Two-dimensional activation matrix captured at the fitted site |
| `layer` | `int`, required | Fitted layer with an existing component profile |
| `rows` | `torch.Tensor \| None = None` | Optional one-dimensional `int64` row selection |
| `batch_size` | `int = 8192` | Activation rows transformed per batch |
| `provenance` | `dict \| None = None` | JSON-compatible provenance for the refreshed statistics |
| `device` | `str \| torch.device \| None = "auto"` | Transform device |
| `progress` | `bool = False` | Display refresh progress |
| **Returns** | `dict` | Updated per-layer profile, also attached to the lens |

The method streams the selected activation rows through the already fitted ICA
transform. It updates sign fractions, squared-energy fractions, score moments,
skewness, excess kurtosis, and `tail_direction`; it also repoints the dominant
Logit Lens and R-lens readouts to the selected tail. Existing examples and
per-sign readouts are preserved. No language-model capture or FastICA fitting
is performed. Use the `icalens profile refresh-statistics` CLI for disk-backed
activation datasets and resumable layer-by-layer refreshes.

### `add_r_lens_profile(...)`

Add R-lens readouts to an existing component profile without replaying its
dataset:

```python
profile = lens.add_r_lens_profile(
    layer=6,
    r_lens="local-r-lens-models/model/lens.pt",
    top_k=20,
    batch_size=8,
    device="auto",
    progress=True,
    allow_base_model_transfer=False,
)
lens.checkpoint_component_profile("icalens-output/my-icalens", layer=6)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `layer` | `int`, required | Layer with an existing component profile |
| `r_lens` | `str \| Path \| dict`, required | Compatible fitted R-lens artifact |
| `top_k` | `int = 20` | Vocabulary entries retained per sign direction |
| `batch_size` | `int = 8` | Component directions processed together |
| `device` | `str \| torch.device \| None = "auto"` | Language-model device |
| `progress` | `bool = False` | Display R-lens projection progress |
| `allow_base_model_transfer` | `bool = False` | Explicitly permit a dimension-compatible base-model R-lens to be reused for an instruct ICA Lens and record the transfer provenance |
| **Returns** | `dict` | Updated per-layer profile, also attached to the lens |

The method preserves existing sign statistics, examples, and Logit Lens
entries. The R-lens must match the analyzed model and hidden size and provide a
source map for the requested `resid_post` layer. Use `save()` or
`checkpoint_component_profile()` afterward to persist the update.
Exact model provenance is required by default. Base-to-instruct reuse must be
explicitly enabled; hidden-size, activation-site, and layer-map checks still
apply, and the source and target model identities are stored in the profile.

### `component_profile(...)`

```python
component = lens.component_profile(layer=5, component=188)
component  # displays the profile panel in Jupyter or Colab
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `layer` | `int`, required | Profiled layer |
| `component` | `int`, required | Component ID |
| **Returns** | `ComponentProfile` | Dictionary-compatible profile with notebook display and `to_html()` |

Profiles are loaded lazily when reading a local or Hugging Face artifact.

The returned component profile supports ordinary dictionary indexing and has the same `tail_direction`,
`dominant_sign`, `sign_statistics`, `score_statistics`, `examples`, `logit_lens`, and optional `r_lens` fields
described above. When the fitted layer records logcosh component objectives,
the returned display object additionally contains `fitting_statistics`: the
objective, Gaussian baseline, absolute deviation, and one-based deviation rank.
This field is derived from layer fitting metadata at read time; it is not
written into the compressed component-profile JSON. A final `component`
expression displays the profile panel in a notebook;
`component.to_html("component-188.html")` saves it. Its
four sign statistics are:

| Field | Meaning |
| --- | --- |
| `positive_fraction` | Fraction of nonzero profiled positions with a positive score |
| `negative_fraction` | Fraction of nonzero profiled positions with a negative score |
| `positive_energy_fraction` | Fraction of this component's corpus-wide squared score magnitude on the positive side |
| `negative_energy_fraction` | Fraction of this component's corpus-wide squared score magnitude on the negative side |

### `top_occurrence(...)`

Retrieve one stored activating occurrence without navigating the complete
component-profile dictionary:

```python
occurrence = lens.top_occurrence(layer=7, component=452, id=0)
print(occurrence["context"], occurrence["text"], occurrence["score"])
```

Occurrence IDs are zero-based positions in the stored ranking. By default the
method uses the component's selected tail. Pass `sign="positive"` or
`sign="negative"` to choose a side explicitly.

The regular result contains the short context window stored in the artifact.
Set `full=True` to retrieve the complete original dataset sample using its
recorded repository, revision, split, text field, and `source_index`:

```python
occurrence = lens.top_occurrence(layer=7, component=452, id=0, full=True)
print(occurrence["full_text"])
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `layer` | `int`, required | Profiled layer |
| `component` | `int`, required | Component ID |
| `id` | `int = 0` | Zero-based stored occurrence ID |
| `sign` | `"positive" \| "negative" \| None = None` | Requested score side; `None` uses the selected tail |
| `full` | `bool = False` | Add the complete source sample as `full_text` |
| **Returns** | `dict` | Independent copy of the stored occurrence record |

### `erf.suffix_sweep(...)`

Measure how much left context a stored component occurrence needs to recover
its selected-tail absolute-score rank:

```python
result = lens.erf.suffix_sweep(
    layer=6,
    component=214,
    rank_thresholds=(1, 3, 5, 10, 15),
    occurrences=20,
)
```

The method uses the component's stored selected-tail occurrences, lazily loads
and then reuses the Lens model and tokenizer, and returns component summaries
for every requested rank threshold together with the occurrence-level sweep
records. It does not write files. Use `lens.unload_model()` to release the
cached language model afterward.

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `layer` | `int`, required | Profiled transformer-block layer |
| `component` | `int`, required | Stored component ID |
| `rank_thresholds` | `Sequence[int] = (1, 3, 5, 10, 15)` | Absolute-score ranks recorded in one sweep |
| `occurrences` | `int = 20` | Maximum stored selected-tail occurrences to measure |
| `exact_suffix_length` | `int = 10` | Test every suffix length through this value |
| `max_batch_size` | `int = 64` | Maximum occurrence batch size |
| `batch_token_budget` | `int = 64` | Approximate lexical-token budget per forward batch |
| `device` | `str \| torch.device \| None = "auto"` | Model device |
| `model` | `torch.nn.Module \| None = None` | Optional caller-supplied language model |
| `tokenizer` | tokenizer or `None` | Optional caller-supplied tokenizer |
| `verbose` | `bool = False` | Report model loading and reuse |
| **Returns** | `dict` | Threshold summaries and occurrence-level sweep records |

### `checkpoint_component_profile(...)`

Persist one completed in-memory layer profile immediately:

```python
path = lens.checkpoint_component_profile(
    "icalens-output/my-icalens",
    layer=5,
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `path` | `str \| Path`, required | Existing local ICA Lens artifact directory |
| `layer` | `int`, required | Layer whose newly created profile should be written |
| **Returns** | `Path` | Resolved artifact directory |

The layer must already have an in-memory profile created by
`profile_components()` or enriched by `add_r_lens_profile()`. This method writes the compressed profile under
`component_profiles/` and updates the manifest without rewriting unrelated
layer tensors. The `icalens profile` CLI uses it to checkpoint every completed
layer. For an ordinary one-shot Python workflow, calling `save()` is sufficient.

## Transform activation tensors

### `transform(...)`

```python
scores = lens.transform(activations, layer=6)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `activations` | floating `torch.Tensor` or `numpy.ndarray`, required | Activation vectors whose final dimension equals `lens.hidden_size` |
| `layer` | `int`, required | Fitted layer transformation to apply |
| **Returns** | same array family as `activations` | Signed ICA scores with the leading dimensions preserved |

`activations` may be a floating-point PyTorch tensor or NumPy array. Its final
dimension must equal `lens.hidden_size`; any leading dimensions are preserved.
The return value has the same array family, device, leading shape, and a final
dimension equal to the number of fitted components.

The operation applies the artifact's preprocessing, fitted center, whitening,
and ICA rotation. The result contains signed ICA scores.

### `energy(...)`

```python
energy = lens.energy(scores)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `scores` | floating `torch.Tensor` or `numpy.ndarray`, required | Signed component scores |
| **Returns** | same array family as `scores` | Non-negative per-vector component energy shares |

Returns `scores.square() / scores.square().sum(-1, keepdim=True)`. Energy is
non-negative and sums to one for every nonzero score vector. It is a viewing
metric, not an invertible coordinate system.

### `keep_topk(...)`

```python
top_scores = lens.keep_topk(scores, k=10)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `scores` | floating `torch.Tensor` or `numpy.ndarray`, required | Signed component scores |
| `k` | `int`, required | Number of largest absolute scores to retain per vector |
| **Returns** | same array family as `scores` | Copy with all other components set to zero |

### `ablate_topk(...)`

```python
ablated_scores = lens.ablate_topk(scores, k=10)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `scores` | floating `torch.Tensor` or `numpy.ndarray`, required | Signed component scores |
| `k` | `int`, required | Number of largest absolute scores to remove per vector |
| **Returns** | same array family as `scores` | Copy with the selected components set to zero |

Both methods rank components independently at every token position and do not
modify their inputs in place.

### `inverse_transform(...)`

```python
reconstructed = lens.inverse_transform(top_scores, layer=6)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `scores` | floating `torch.Tensor` or `numpy.ndarray`, required | Signed scores whose final dimension equals the fitted component count |
| `layer` | `int`, required | Fitted writing matrix and center to apply |
| **Returns** | same array family as `scores` | Reconstructed preprocessed activations |

Maps signed scores back through the writing matrix and fitted center. The
result reconstructs the preprocessed activation, so only signed scores—not
energy shares—are valid input.

### `restore_norm(...)`

```python
hidden_states = lens.restore_norm(
    reconstructed,
    reference=activations,
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `values` | floating `torch.Tensor` or `numpy.ndarray`, required | Reconstructed or modified activation directions |
| `reference` | same array type and shape, required | Activations whose per-vector norms should be restored |
| **Returns** | same array family as `values` | `values` rescaled to the reference norms |

Rescales each reconstructed vector to the norm of its corresponding reference
activation. `values` and `reference` must have identical shapes and must both
be PyTorch tensors or both be NumPy arrays.

## Inspect fitting

### `plot_fitting_curve(...)`

Plot the component-wise FastICA objective distribution recorded during fitting:

```python
figure = lens.plot_fitting_curve(layer=6)
figure  # displays inline in Jupyter

# Individual layer panels in one figure
figure = lens.plot_fitting_curve(layers=[0, 6, 11], columns=3)
figure = lens.plot_fitting_curve(layers="all", columns=4)
```

```python
lens.plot_fitting_curve(
    *,
    layer=None,
    layers=None,
    columns=None,
) -> matplotlib.figure.Figure
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `layer` | `int \| None = None` | One fitted layer; mutually exclusive with `layers` |
| `layers` | `list[int] \| tuple[int, ...] \| "all" \| None` | Multiple fitted layers; mutually exclusive with `layer` |
| `columns` | `int \| None = None` | Subplot columns; defaults to 2 and is capped at the selected layer count |
| **Returns** | Matplotlib `Figure` | Nested component-percentile bands and the median objective curve |

Pass exactly one of `layer` and `layers`. Multiple layers are shown as separate
subplots; they are not aggregated. Each panel shows the saved minimum, 10th
through 90th percentiles, maximum, and median across components at every
recorded iteration.

The plot is generated entirely from artifact metadata. It does not load the
language model, fitting activations, or layer tensors. The selected layers must
contain `objective_history`; current fitting commands record it by default, at
the interval selected by `--objective-every` or `objective_every`.

## Fit activation tensors

### `fit(...)`

```python
lens.fit(
    activations,
    layer=6,
    n_components=None,
    max_iter=20,
    random_state=0,
    progress=True,
    device="cuda",
    batch_size=8192,
    objective_every=1,
    provenance={
        "dataset": {"repo_id": "owner/dataset", "split": "train"},
        "fitting_tokens": int(activations.shape[0]),
    },
)
```

`fit()` fits or replaces one layer and returns the same lens object.

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `activations` | floating `torch.Tensor` or `numpy.ndarray`, required | Activations; leading dimensions are flattened into samples |
| `layer` | `int`, required | Label for the source layer; `fit()` does not capture activations |
| `n_components` | `int \| None = None` | Component count; defaults to the activation width |
| `algorithm` | `str = "parallel"` | FastICA update algorithm; currently only `"parallel"` |
| `fun` | `str = "logcosh"` | Contrast function; currently only `"logcosh"` |
| `max_iter` | `int = 200` | Fixed FastICA iteration count; no tolerance-based early stopping |
| `random_state` | `int \| None = 0` | FastICA initialization seed |
| `progress` | `bool = False` | Show preprocessing and optimization progress |
| `device` | `str \| torch.device \| None = None` | Fitting device; defaults to the activation tensor's device |
| `batch_size` | `int = 8192` | Activation rows processed on the fitting device at once |
| `objective_every` | `int = 1` | Record objective percentiles every N iterations |
| `provenance` | `dict \| None = None` | JSON-compatible fitting metadata stored verbatim in the artifact |
| **Returns** | `ICALens` | The same lens, with this layer fitted or replaced |

A full-width fit requires at least `hidden_size + 1` samples because centering
removes one degree of rank. Calling `fit()` again with another layer adds it to
the same artifact. See [Fit and publish](fit-and-publish.md) for end-to-end
dataset commands and a complete Python example.

## Save and publish

### `save(...)`

```python
path = lens.save("icalens-output/my-icalens")
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `path` | `str \| Path`, required | Destination artifact directory |
| **Returns** | `Path` | Resolved destination directory |

Writes the manifest, fitted layer tensors, and generated model card. It returns
the resolved output `Path`. Local output paths are not serialized, but values
placed in `provenance` are stored verbatim.

### `push_to_hub(...)`

```python
url = lens.push_to_hub(
    "owner/my-icalens",
    private=False,
    revision="main",
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `repo_id` | `str`, required | Destination in `owner/name` form |
| `private` | `bool \| None = None` | Requested visibility when creating the repository |
| `token` | `str \| bool \| None = None` | Hugging Face authentication setting |
| `revision` | `str = "main"` | Destination branch or revision |
| `commit_message` | `str = "Upload ICA Lens artifacts"` | Hub commit message |
| **Returns** | `str` | URL of the created Hub commit |

Creates or updates a Hugging Face **Model** repository and returns the commit
URL. Authentication follows `huggingface_hub`; use `hf auth login`, `HF_TOKEN`,
or pass `token=` explicitly.

## Results

### `CaptureResult`

`capture()` returns token-aligned model data:

| Field | Contents |
| --- | --- |
| `tokens` | Tokenizer-native token strings |
| `token_texts` | Readable token text used by reports |
| `token_labels` | Compact token-card labels |
| `token_tooltips` | Expanded text shown on hover |
| `token_groups` | Conversation group label for each selected token; empty for raw text |
| `token_ids` | Selected token IDs |
| `positions` | Positions in the complete encoded model input |
| `activations` | Captured activation vectors |

### `AnalysisResult`

`AnalysisResult` contains every `CaptureResult` field plus:

| Field | Contents |
| --- | --- |
| `scores` | Signed ICA component scores |
| `energy` | Per-token component energy shares |
| `model` | Model repository and exact revision |
| `layer` | Analyzed transformer-block index |
| `input_text` | Readable input representation |
| `token_scope` | Positions selected from the encoded input |
| `messages` | Original conversation messages, or an empty tuple for text |
| `component_profiles` | Compact per-component profile summaries for the analyzed layer, or `None` when that layer has no profile |
| `logit_effects` | Token-local Logit Lens effects added by `lens.add_logit_effects(...)` |

In Jupyter or Colab, leave `result` as the final expression to render it:

```python
result
```

`component_profiles` supplies the interactive result with sign statistics,
top-score occurrences, Logit Lens tokens, and optional R-lens tokens. Full
stored profiles remain available through `lens.component_profile(...)`.

### `add_logit_effects(...)`

After analyzing an input, scale its top component scores and attach the largest
direct Logit Lens changes to the result:

```python
result = lens.analyze("She deposited the check at the bank.", layer=6)
result = lens.add_logit_effects(
    result,
    components_per_token=3,
    multiplier=1.1,
    effect_tokens_per_component=10,
)
result
```

The method evaluates the leading absolute-score components independently at
every analyzed token. Its values appear in a separate **Local intervention
projection** panel. They use final normalization and unembedding directly and
do not run the remaining transformer layers.

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `result` | `AnalysisResult`, required | Existing analysis whose cached model and activations are reused |
| `components_per_token` | `int = 3` | Highest absolute-score components evaluated at each analyzed token |
| `multiplier` | `float = 1.1` | Multiplicative change applied to the original signed score |
| `effect_tokens_per_component` | `int = 10` | Vocabulary tokens retained for each local component edit, ranked by absolute logit change |
| `batch_size` | `int = 32` | Local edits unembedded together; lower this to reduce temporary memory |
| `vocabulary_batch_size` | `int = 16384` | Vocabulary rows projected together in float32; lower this to reduce temporary memory |
| **Returns** | `AnalysisResult` | A new result containing the attached local effect |

### `AnalysisResult.display(...)`

Configure the inline view explicitly with:

```python
result.display(metric="energy", top_k=5, height=720)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `metric` | `"score" \| "energy" = "score"` | Values shown in component bars |
| `top_k` | `int = 3` | Components displayed per token card |
| `title` | `str = "ICA Lens Explorer"` | Embedded report title |
| `height` | `int = 720` | Initial iframe height in pixels |
| **Returns** | `None` | Displays the interactive result through IPython |

### `AnalysisResult.to_html(...)`

Write a self-contained HTML explorer outside a notebook with:

```python
output = result.to_html(
    "analysis.html",
    metric="score",
    top_k=3,
    title="ICA Lens Explorer",
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `output_file` | `str \| Path`, required | Destination HTML file |
| `metric` | `"score" \| "energy" = "score"` | Values shown in component bars |
| `top_k` | `int = 3` | Components displayed per token card |
| `title` | `str = "ICA Lens Explorer"` | Report title |
| **Returns** | `Path` | Resolved path of the self-contained report |

## Inspect a lens

```python
print(lens.model_id)
print(lens.model_revision)
print(lens.model_type)
print(lens.activation_site)
print(lens.hidden_size)
print(lens.available_layers)
print(lens.metadata)
```

`metadata` returns a detached dictionary representing the portable manifest.

## `ActivationDataset`

Open reusable activations captured by `icalens capture text` or `icalens capture chat`:

```python
from icalens import ActivationDataset

captured = ActivationDataset("/mnt/external/icalens-activations/gpt2-pile10k-1m")
values = captured.layer(6)
```

| Member | Type | Meaning |
| --- | --- | --- |
| `path` | `Path` | Resolved activation-dataset directory |
| `available_layers` | `tuple[int, ...]` | Captured transformer layers |
| `sample_count` | `int` | Number of aligned token rows per layer |
| `hidden_size` | `int` | Width of every activation row |
| `dtype` | `torch.dtype` | On-disk activation dtype |
| `model` | `dict` | Recorded model identity, revision, and type |
| `provenance` | `dict` | Dataset, sampling, framing, and format provenance |
| `layer(layer)` | `torch.Tensor` | Disk-backed `[sample_count, hidden_size]` tensor |

## Exceptions

```python
from icalens import ArtifactError, ICALensError, NotFittedError
```

- `ICALensError` is the base package exception.
- `ArtifactError` reports missing, malformed, or incompatible artifacts.
- `NotFittedError` reports a requested layer that has not been fitted or
  loaded.
