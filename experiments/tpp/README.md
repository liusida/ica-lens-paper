# TPP representation comparison

This experiment adapts SAEBench targeted probe perturbation (TPP) to compare
fitted ICA with a published trained SAE at the same residual-stream site. The
current protocol is method-definition version 4.

The default methods are:

1. fitted ICA with one profile-oriented ReLU feature per component; and
2. a published trained SAE.

Three optional controls remain available but are disabled by default:

1. unfitted ICA (the seeded, orthogonal FastICA initialization applied after
   whitening);
2. an untrained, tied random SAE control; and
3. PCA.

Use `--methods all` to evaluate all five representations, or pass an explicit
comma-separated subset such as `--methods ica,sae,pca`.

## ICA feature definition

For fitted ICA component $i$, let $z_i$ be its raw ICA coordinate, $s_i$ its
profile-derived dominant-tail sign, and $d_i$ its raw decoder vector. TPP
exposes one non-negative feature:

$$
a_i = \max(s_i z_i, 0)\lVert d_i\rVert.
$$

The decoder row is $s_i d_i / \lVert d_i\rVert$, so decoder normalization does
not change the represented positive-half-wave contribution. This definition
makes fitted ICA compatible with SAEBench's non-negative feature-selection and
zero-ablation protocol. It removes the profile-dominant positive half-wave; it
does not claim to remove the complete signed ICA coordinate. Negative-half-wave
contributions remain in the reconstruction error that SAEBench adds back.

The optional unfitted ICA and PCA controls remain signed linear coordinates.

## Intervention rule

SAEBench applies the same reconstruction-error-preserving intervention to ICA
and SAE wrappers:

$$
\widetilde{x}
= \operatorname{decode}(\operatorname{ablate}(\operatorname{encode}(x)))
+ x - \operatorname{decode}(\operatorname{encode}(x)).
$$

Consequently, no ablation leaves $x$ unchanged, while ablating feature $i$
removes that wrapper's decoded feature contribution. CPU-only regression tests
for both wrappers live in `tests/test_tpp_intervention.py`.

For each target class, SAEBench ranks features by the product of their clamped
positive class-conditional activation difference and alignment with the trained
probe. It zero-ablates the selected features and adds the original
reconstruction error back. The reported TPP score is the targeted probe's
accuracy drop minus the mean accuracy drop of the other probes. Higher is
better only when the intended change is large and unintended changes remain
small.

## Input framing

Evaluation reproduces the document framing recorded in each ICA Lens artifact:

- GPT-2 receives its fitted `<|endoftext|>` boundary token.
- Gemma uses the tokenizer-provided `<bos>` token.
- Qwen receives its fitted `<|endoftext|>` boundary token.

The boundary participates in the model forward pass but is excluded from
pooled evaluation activations. Padding tokens are excluded with an attention
mask. This is the BOS-conditioned protocol used by both TPP and Sparse Probing.

## Default paper configuration

The runner defaults to `--preset paper` and `--methods ica,sae`. The paper
preset uses:

- 4,000 training examples;
- 1,000 held-out test examples;
- 20 probe-training epochs; and
- feature budgets $k \in \{1,2,5,10,20,50,100\}$.

The preset's fallback seed is 42, but official results use explicit seeds 0,
1, and 2 because TPP can be seed-sensitive.

Run the three GPT-2 seeds sequentially:

```bash
for seed in 0 1 2; do
  uv run python experiments/tpp/run.py \
    --lens local-icalens-models/official/icalens-gpt2-small-pile10k \
    --layers 6,10 \
    --preset paper \
    --methods ica,sae \
    --seed "$seed" \
    --output "experiments/tpp/official/results/seed-${seed}/gpt2"
done
```

Official result directories follow:

```text
experiments/tpp/official/results/
  seed-0/<model>/
  seed-1/<model>/
  seed-2/<model>/
```

Use a new output directory whenever a result-affecting option changes. Repeating
an identical command resumes completed layers when its configuration matches.

## Activation caches

Large SAEBench activation caches are stored outside the repository under
`~/Expansion/research/ICA-data/tpp` by default. The runner namespaces them by
the resolved scientific configuration and layer and records the exact paths in
`<output>/storage.json`. Different seeds use different caches because dataset
sampling and trained probes are seed-dependent. ICA and SAE share the raw model
activations within one run.

Override the storage filesystem with `--activation-cache-root`. Do not point
`--activation-cache-run-dir` at a cache from a scientifically different
configuration. Keep a run's cache until its results are complete and validated;
old cache-key directories may then be removed to recover disk space.

## Explicit smoke check

Smoke mode remains available but is never implied by the default command:

```bash
uv run python experiments/tpp/run.py \
  --lens local-icalens-models/official/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --methods ica,sae \
  --seed 0 \
  --output experiments/tpp/runs/gpt2-l6-smoke
```

The original three-model smoke figures are archived under
`pilot-evidence/qr-unfitted-control/`. They used a superseded QR-based
unfitted-ICA control and are not scientific results.
