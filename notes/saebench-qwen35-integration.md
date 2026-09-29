# SAEBench integration for Qwen3.5

ICA Lens evaluates sparse probing with the `stable_paper_version` of the original
[SAEBench repository](https://github.com/adamkarvonen/SAEBench/tree/stable_paper_version),
pinned to commit `2a7480bf67f9f82f79e742be55fc1f5ec404bfa9`. The checked-out SAEBench source is
not modified.

That historical revision installs Transformers 4, which predates Qwen3.5. ICA Lens therefore
upgrades Transformers to version 5 or newer inside SAEBench's isolated environment. Because
the accompanying TransformerLens 2.11 import expects the removed `TRANSFORMERS_CACHE`
attribute, ICA Lens temporarily maps that attribute to Hugging Face's current cache path while
importing SAEBench.

For sparse probing, ICA Lens replaces only SAEBench's model-loading boundary with a small
Hugging Face adapter. The adapter loads Qwen3.5 with Transformers, captures the requested
residual-stream layer, and exposes the limited hook interface used by sparse probing. The
SAEBench datasets, train/test splits, feature ranking, linear probes, metrics, and evaluation
logic remain those of the pinned upstream revision. The original and previously used Qwen
fork have byte-identical sparse-probing implementations, so a separate Qwen-specific
SAEBench environment is unnecessary.

In short: the benchmark implementation is pinned and unchanged; only its isolated dependency
environment and model-loading interface are adapted so that the same evaluation can run on
Qwen3.5.

## Shared multi-layer activation capture

For evaluations requesting more than one layer, ICA Lens keeps SAEBench's dataset loading,
class filtering, tokenization, masking, sequence pooling, feature selection, probe training,
and metrics unchanged, but captures all requested residual-stream layers in one model forward
pass. Execution is dataset-first: every requested method and layer consumes the shared raw
activations, and the dataset cache is deleted only after all corresponding result files are
durable.

Before using this path for the official Qwen 3.5 9B evaluation, we validate it on GPT-2 layers
6 and 10 against the completed original-path run, comparing sampled activation tensors,
selected features, per-class accuracies, and aggregate curves.
