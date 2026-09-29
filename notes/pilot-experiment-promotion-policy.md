# Pilot experiment promotion checklist

Before moving a pilot to `experiments/` or scaling it, record each item as
**matched**, **intentional difference**, or **blocked**. Do not scale with an
unexplained mismatch. Discuss any mismatch with the user rather than silently
handling or compensating for it.

## Scientific definition

- State the question, metric, population, aggregation, and intended claim.
- Confirm that fitting, evaluation, plotting, and captions describe the same
  populations and operations.

## Semantic parity

- Pin model, tokenizer, dataset, and revisions.
- Match activation site and layer indexing, including the final layer.
- Compare native Transformers and wrapper hooks on one fixed batch.
- Match BOS/EOS framing, padding, masks, positions, packing, truncation,
  context length, and target-token indexing.
- Match centering, normalization, whitening, sign, biases, and SAE preprocessing.
- Give every method the same examples, layers, budgets, and metrics unless a
  difference is intrinsic and documented.

Replay one identical example through fitting-style and evaluation-style
preprocessing; compare token IDs, masks, positions, and captured tensors.

## Promotion smoke test

- Run one deterministic model--layer--example case and verify one metric by hand.
- Check special tokens, last-layer hooks, token alignment, aggregation, and resume.
- Record the command, findings, intentional differences, expected official-run
  cost, and source commit in the experiment README.

Follow [`long-run-policy.md`](long-run-policy.md) and
[`plot-style-policy.md`](plot-style-policy.md); keep large results ignored and
result-affecting configuration in the run manifest.
