# Artifact Dataset

Public artifacts live in one Hugging Face dataset:

```text
sida/ica-lens-paper
```

Dataset layout:

```text
models/
  gpt2/
  gemma2_2b/
  qwen3_5_2b_base/
databases/
  ica_probe_mini.sqlite
  ica_probe_full.sqlite
manifest.json
checksums.sha256
```

`artifacts/manifest.json` points both artifact sets at this dataset. The
fetcher narrows those patterns at runtime: `--models` downloads `models/**`,
and `--databases` defaults to only the mini database.

To fetch just one database variant, use:

```bash
uv run python scripts/fetch_artifacts.py --databases --database-variant mini
uv run python scripts/fetch_artifacts.py --databases --database-variant full
uv run python scripts/fetch_artifacts.py --databases --database-variant all
```
