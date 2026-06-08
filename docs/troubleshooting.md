# Troubleshooting

## Artifact Fetch Says `not_configured`

The Hugging Face dataset ID should be:

```text
sida/ica-lens-paper
```

If this error appears, check `artifacts/manifest.json` and any
`ICA_EXPLORER_ARTIFACT_REPO` or `ICA_EXPLORER_DB_REPO` environment overrides.

## Explorer Cannot Find a Database

Run:

```bash
uv run python scripts/fetch_artifacts.py --databases
uv run python scripts/verify_artifacts.py
```

Then check that `artifacts/fetched/databases/` contains either the miniature or
full SQLite database.

## Demo Explorer Uses Published ICA Artifacts

`scripts/reproduce_all.py --mode demo` builds a demo database under
`results/demo/databases/`, but the demo ICA refits are stored with token-budget
suffixes such as `results/demo/ica/gpt2_tok3000/`. The server expects
model-name directories such as `gpt2/`, so launch the demo database with the
published ICA artifact root:

```bash
ICA_EXPLORER_DB_PATH=results/demo/databases/ica_probe_demo.sqlite \
ICA_EXPLORER_ICA_ROOT=artifacts/fetched/models \
uv run python -m server.app --port 8001
```

## Paper-Scale Reproduction

The one-command paper-scale reproduction is not implemented yet. Use the
published artifact path for the released explorer state and the demo path for a
small end-to-end local reproduction.
