# Vendor

Vendored dependencies are tracked as git submodules in the standalone `v6`
repository:

- `vendor/FastICA_torch`: FastICA implementation used by ICA fitting.
- `vendor/SAEBench`: SAEBench source used by sparse-probe and TPP workflows.
- `vendor/SAEBench-qwen35`: Qwen-specific SAEBench fork for full reproduction.

Generated vendor environments and artifacts should not be committed.

Do not commit downloaded artifact data here.
