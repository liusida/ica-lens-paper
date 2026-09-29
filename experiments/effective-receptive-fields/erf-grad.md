# Gradient effective receptive field

Gradient ERF estimates how far back in the text a component typically draws its
input influence. A value near 1 means that the target token itself dominates;
larger values mean that earlier tokens matter more.

## One occurrence

For one stored dominant-tail occurrence, we run the full prefix ending at the
target token through the language model. Let $s$ be the selected component's
score at the target position. For every input token $i$, we measure how sensitive
$s$ is to that token's input embedding $e_i$:

$$
a_i = \left\lVert \frac{\partial s}{\partial e_i} \right\rVert_2^2.
$$

We normalize these nonnegative influences so they sum to one:

$$
p_i = \frac{a_i}{\sum_j a_j}.
$$

The target token has distance $d_i=1$, the preceding token has distance 2, and
so on. The occurrence-level ERF is the influence-weighted geometric mean of
these distances:

$$
\operatorname{ERF}_{\mathrm{grad}}
= \exp\left(\sum_i p_i \log d_i\right).
$$

The geometric mean gives nearby influential tokens more weight than an
arithmetic average would. For example, if all influence is on the target token,
ERF is 1; if it is split equally between distances 1 and 4, ERF is 2.

## One component

We compute the occurrence-level ERF for each of the component's stored
dominant-tail examples, normally 20. The component's reported ERF is their
median:

$$
\operatorname{ERF}_{\mathrm{component}}
= \operatorname{median}(\operatorname{ERF}_1,\ldots,\operatorname{ERF}_n).
$$

Using the median makes the result less sensitive to an unusually short or long
example. Document-framing tokens are excluded from the influence normalization.

This is a gradient-based sensitivity summary, not a claim that exactly this many
tokens form a hard context window. It uses one forward and backward computation
per occurrence (with multiple occurrences batched together); it does not rerun
the model on many truncated suffixes.
