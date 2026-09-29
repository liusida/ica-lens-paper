# Scores and energy

For every analyzed token, ICA Lens provides two views of the same component
coordinates: signed ICA scores and normalized energy shares.

## ICA scores

`result.scores` contains the standard signed ICA coordinates. If a token has
activation vector \(x\), ICA Lens first applies the preprocessing recorded by
the fitted lens and then projects it onto the ICA components:

\[
s = W(\tilde{x} - \mu)
\]

Following the ICA Lens paper, the default preprocessing is per-token L2
normalization:

\[
\tilde{x} = \frac{x}{\lVert x \rVert_2}
\]

Here, \(\mu\) is the fitted mean of the normalized activations, \(W\) is the
reading matrix, and \(s_j\) is the score for component \(j\).

### Where is whitening?

FastICA is fitted in whitened coordinates. Let \(K\) be the whitening matrix
computed from the centered, L2-normalized fitting activations, and let \(U\) be
the ICA rotation (unmixing matrix) learned in that whitened space. ICA Lens
stores their composition as the reading matrix:

\[
W = UK
\]

Therefore, the complete score transformation is:

\[
s = UK(\tilde{x} - \mu) = W(\tilde{x} - \mu)
\]

Whitening is therefore part of `lens.transform()` even though there is no
separate whitening call at analysis time. The fitted artifact stores the
composed reading matrix rather than requiring users to apply the two matrices
manually.

```python
scores = result.scores
print(scores.shape)  # [tokens, components]
```

A large positive or negative value means that the token lies strongly along
that component direction. Scores are appropriate when you need:

- activation direction as well as magnitude;
- positive-versus-negative comparisons;
- component distributions across tokens;
- reconstruction or component modification.

ICA component signs are conventional: an equally valid ICA solution can flip
a component and all of its scores. Once a fitted artifact is saved, its sign is
stable, so comparisons within that lens remain meaningful. Do not assume that
component IDs, signs, or raw score scales are aligned across independently
fitted lenses.

## Energy shares

For a component score \(s_j\), ICA Lens defines its energy share at one token
as:

\[
e_j = \frac{s_j^2}{\sum_k s_k^2}
\]

```python
energy = result.energy
print(energy.shape)          # [tokens, components]
print(energy.sum(dim=-1))    # approximately 1 for each nonzero row
```

Squaring measures each coordinate's contribution to the squared Euclidean
length of the score vector. Dividing by the total makes the values nonnegative
shares that sum to one for every nonzero token position.

An energy value of `0.20`, or 20%, means:

> This component accounts for 20% of the squared ICA-score magnitude at this
> token position.

Energy is useful for identifying which components dominate a token without
positive and negative scores cancelling each other. It is not a probability,
confidence score, or estimate of semantic importance. It also discards sign.

For an all-zero score vector, ICA Lens returns an all-zero energy vector rather
than dividing by zero.

## Score or energy?

| Question | Use |
| --- | --- |
| Is the component active positively or negatively? | Score |
| How unusually large is a signed component activation? | Score |
| Which components dominate this token's score vector? | Energy |
| What fraction of squared score magnitude belongs to a component? | Energy |
| Which side forms a component's asymmetric tail across the profiling corpus? | Profile skewness |
| Do I want to reconstruct or modify an activation? | Score |

The interactive result defaults to scores. Select **Energy** when relative
within-token concentration is easier to interpret.

## Inspect values in Python

Find the strongest signed components for each token:

```python
values, components = result.scores.abs().topk(3, dim=-1)
```

Find the components with the largest energy shares:

```python
shares, components = result.energy.topk(3, dim=-1)
```

The first dimension of both arrays follows `result.tokens`, so row `i` always
describes token `result.tokens[i]`.

For mapping scores back into activation space, continue to
[Reconstruction](reconstruction.md).
