# Reconstruction

ICA Lens can map component scores back into the activation coordinates used by
the fitted lens:

```python
normalized_reconstruction = lens.inverse_transform(result.scores, layer=6)
```

!!! note
    `inverse_transform()` requires signed scores (`result.scores`), not energy
    shares (`result.energy`).

If \(A\) is the fitted writing matrix and \(\mu\) is the fitted center, the
inverse mapping is:

\[
\hat{x} = sA^\top + \mu
\]

The returned tensor has the same leading dimensions and activation width as the
input used to produce the scores.

## What is reconstructed?

Following the ICA Lens paper, the default preprocessing applies L2
normalization independently at each token position:

\[
\tilde{x} = \frac{x}{\lVert x \rVert_2}
\]

The normalized activation is then centered, whitened, and rotated into ICA
coordinates. Whitening and the ICA rotation are composed in the stored reading
matrix; the writing matrix used by `inverse_transform()` maps back through that
combined transformation.

Therefore, `inverse_transform()` first reconstructs \(\tilde{x}\), the
normalized preprocessed activation. L2 normalization discards the scalar
\(\lVert x\rVert_2\), so scores alone cannot recover the original activation
norm.

```python
captured = lens.capture("She deposited the check.", layer=6)
scores = lens.transform(captured.activations, layer=6)
normalized_reconstruction = lens.inverse_transform(scores, layer=6)
```

With a full-rank lens, reconstruction should be close to the preprocessed
activation up to numerical error. With fewer ICA components than the activation
width, reconstruction is a lower-dimensional approximation and necessarily
loses information.

## Restore the hidden-state norm

Although the fitted lens cannot know the norms of future inputs, `analyze()`
retains the original captured hidden states in `result.activations`. Their norms
can therefore be recorded before transformation and restored afterward:

```python
normalized_reconstruction = lens.inverse_transform(result.scores, layer=6)
restored_hidden_states = lens.restore_norm(
    normalized_reconstruction,
    reference=result.activations,
)
```

For a full-rank lens, `restored_hidden_states` should closely reproduce the
captured hidden states, up to numerical fitting and floating-point error. This
rescaling is input-specific: the normalization method is stored in the lens,
while the norm values come from the particular hidden states being analyzed.

## Modify a component

Signed scores can be copied and edited before reconstruction. Energy is useful
for selecting a prominent component, but the actual modification and inverse
transformation must operate on `result.scores`:

```python
modified_scores = result.scores.clone()
modified_scores[:, 37] = 0

modified_normalized_reconstruction = lens.inverse_transform(modified_scores, layer=6)
modified_hidden_states = lens.restore_norm(
    modified_normalized_reconstruction,
    reference=result.activations,
)
```

This constructs activation vectors with component 37 removed in ICA score
space, then restores the original per-token hidden-state norms. You can
similarly increase, decrease, or replace a component score.

Using the original norm makes the intervention primarily directional and avoids
an unintended change in overall residual-stream magnitude. It is a sensible
default for actionable hidden-state modification, but it is still an
intervention choice: preserving the original norm may partially rescale the
effect of a large score edit.

The reconstructed vectors are not automatically inserted into a language
model. Doing so requires a separate intervention mechanism at the same model,
layer, activation site, and token positions used by the lens. Treat large or
out-of-distribution score changes cautiously: the inverse mapping is linear,
but the downstream model behavior is not.

## Keep or ablate the top-k components

ICA Lens provides per-token top-k score masks. Components are ranked by
absolute score, while the retained signed values are left unchanged.

Keep only the ten strongest components at each token position:

```python
top_scores = lens.keep_topk(result.scores, k=10)
top_normalized_reconstruction = lens.inverse_transform(top_scores, layer=6)
top_hidden_states = lens.restore_norm(
    top_normalized_reconstruction,
    reference=result.activations,
)
```

Ablate the ten strongest components and reconstruct from everything else:

```python
remaining_scores = lens.ablate_topk(result.scores, k=10)
ablated_normalized_reconstruction = lens.inverse_transform(remaining_scores, layer=6)
ablated_hidden_states = lens.restore_norm(
    ablated_normalized_reconstruction,
    reference=result.activations,
)
```

The top-k component set is selected independently for every token position.
Both methods return a new array and do not modify `result.scores`. They preserve
the input's shape, dtype, device, and NumPy or PyTorch type.

## Measure reconstruction quality

### Cosine similarity

Cosine similarity is useful when the fitted lens uses row normalization:

```python
import torch.nn.functional as F

top_scores = lens.keep_topk(result.scores, k=10)
top_normalized_reconstruction = lens.inverse_transform(top_scores, layer=6)

similarity = F.cosine_similarity(
    F.normalize(result.activations, dim=-1),
    top_normalized_reconstruction,
    dim=-1,
)
```

Values range from `-1` to `1`; `1` means the reconstructed and original
activation directions are identical. The result contains one value per token.

### Normalized MSE

Compute one value per token by dividing its reconstruction squared error by
the squared error from predicting the result's mean normalized activation.
Use the normalized output of `inverse_transform()`, not a hidden state returned
by `restore_norm()`:

For token \(i\), let

\[
z_i = \frac{x_i}{\lVert x_i \rVert_2},
\qquad
\mu = \frac{1}{n}\sum_i z_i,
\]

and let \(\hat{z}_i\) be its normalized-space reconstruction. The per-token
Normalized MSE is:

\[
\operatorname{NMSE}_i =
\frac{\lVert z_i - \hat{z}_i \rVert_2^2}
     {\lVert z_i - \mu \rVert_2^2}.
\]

```python
import torch.nn.functional as F

top_scores = lens.keep_topk(result.scores, k=10)
top_normalized_reconstruction = lens.inverse_transform(top_scores, layer=6)

normalized = F.normalize(result.activations, dim=-1)
baseline = normalized.mean(dim=0, keepdim=True)

token_squared_error = (
    normalized - top_normalized_reconstruction
).square().sum(dim=-1)
token_baseline_error = (normalized - baseline).square().sum(dim=-1)
token_normalized_mse = token_squared_error / token_baseline_error.clamp_min(1e-12)

print(token_normalized_mse)  # one value per token
```

Interpretation:

- `0` is exact reconstruction;
- `1` means the reconstruction has the same squared error as replacing that
  token's normalized activation with the mean normalized activation across all
  tokens in the result;
- values below `1` improve on that baseline;
- values above `1` are worse than that baseline.

The paper's aggregate Normalized MSE is a ratio of sums, not the sum or mean of
the per-token ratios:

\[
\operatorname{NMSE} =
\frac{\sum_i \lVert z_i - \hat{z}_i \rVert_2^2}
     {\sum_i \lVert z_i - \mu \rVert_2^2}.
\]

```python
normalized_mse = (
    token_squared_error.sum()
    / token_baseline_error.sum().clamp_min(1e-12)
)
```

Individual token values can be very large when a token is already extremely
close to the mean and therefore has a near-zero baseline error. The aggregate
ratio is generally more stable for reporting a text, model, or layer result.
The example assumes the lens uses its default row normalization; for
`row_normalize=False`, use `result.activations` as the target instead.

## Why energy cannot be reconstructed

Energy shares discard each component's sign and the score vector's overall
magnitude. Many different score vectors therefore have the same energy shares,
so `result.energy` cannot be inverted into a unique activation. Reconstruction
must start from signed ICA scores such as `result.scores`.
