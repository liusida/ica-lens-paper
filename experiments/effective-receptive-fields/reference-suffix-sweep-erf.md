# Suffix-sweep effective receptive field

Suffix-sweep ERF asks how many tokens of left context are needed for a component
to remain prominent at a target token. It reruns the model with progressively
longer suffixes ending at that token. A small ERF means that the target token or
a short suffix is sufficient; a large ERF means that the component depends on a
broader preceding context.

## One occurrence

For one stored dominant-tail occurrence, we test every suffix length from 1 to
10, followed by 20, 40, 160, 640, and so on. At each length, we rerun the model
on only the last $k$ tokens ending at the target. The sweep is dense where
short-context differences matter and becomes coarser for long contexts, which
are grouped together in the analysis.

At the target position, we recompute all ICA component scores. For each rank
threshold $r\in\{1,3,5,10,15\}$, the component is considered
recovered at length $k$ when both conditions hold:

1. **Magnitude:** it is among the top $r$ components by absolute score.
2. **Sign:** its score remains on the selected tail.

The occurrence-level ERF is the shortest suffix that succeeds:

$$
\operatorname{ERF}_{\mathrm{suffix}}(r)
= \min\{k:\text{the component is recovered within the top }r\}.
$$

Lengths through 10 are exact. Beyond 10, recovery is bracketed by the last
failed and first successful tested lengths. If those lengths are $a$ and $b$,
the estimate is their geometric midpoint:

$$
\widehat{\operatorname{ERF}}_{\mathrm{suffix}}(r)=\sqrt{ab}.
$$

The stored full-context rank provides the final endpoint. If the component
still does not reach the threshold with its full available context, the
occurrence is marked unrecovered and assigned that full context length. The
recovery flag is retained separately from the numeric value.

## One component

At each threshold, we compute occurrence-level ERF for all stored top-scoring
examples on the component's selected tail, normally 20.

The component's reported ERF is their arithmetic mean:

$$
\operatorname{ERF}_{\mathrm{component}}(r)
= \frac{\operatorname{ERF}_1(r)+\cdots+\operatorname{ERF}_{n_r}(r)}{n_r}.
$$

For example, a top-3 ERF of 2 means that the component typically becomes one of
the three strongest components, with the correct sign, using about two tokens
ending at the target.

This is a rank-based recovery measure. It records the first successful suffix,
not how strongly each earlier token influences the score. Its mean is capped by
each occurrence's available full context. All five thresholds are recorded in
one run and can be plotted separately afterward.
