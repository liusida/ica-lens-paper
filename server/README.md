# ICA Lens Explorer and Annotator

This server is the public `v6` explorer/annotator for released ICA Lens
artifacts. It serves component statistics, examples, annotations, random
component samples, and live text probing when the corresponding model weights
and ICA artifacts are available.

Default local inputs:

```text
artifacts/fetched/models/
artifacts/fetched/databases/ica_probe_mini.sqlite
```

The fetched mini database is the default server database. Use
`ICA_EXPLORER_DB_PATH` only when you want to point the explorer at another
SQLite file, such as `artifacts/fetched/databases/ica_probe_full.sqlite` or a
locally reproduced demo database.

Run from `v6`:

```bash
uv run python -m server.app --port 8001
```

Pages:

```text
/           GPT-2 text probe explorer
/stats      component statistics overview
/component  component examples page
/annotate   manual annotation editor
/random-components  random 05.1 component sample list
```

Set `ICA_EXPLORER_DB_PATH`, `ICA_EXPLORER_ICA_ROOT`,
`ICA_EXPLORER_ICA_DIR`, `ICA_EXPLORER_DEVICE`, or `ICA_EXPLORER_DTYPE` to
override defaults. If local inputs are missing, the server attempts to fetch
released artifacts from the configured Hugging Face dataset unless
`ICA_EXPLORER_DOWNLOAD_MISSING=0`.
