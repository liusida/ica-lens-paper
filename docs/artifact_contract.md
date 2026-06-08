# Artifact Contract

Public artifacts live in the `sida/ica-lens-paper` Hugging Face dataset.

## Fitted ICA Models

The `models/` folder should contain model/layer ICA artifacts and JSON
metadata. Metadata must record:

- model short name and upstream model ID;
- layer name and hook/site convention;
- activation preprocessing mode;
- number of components;
- ICA seed, tolerance, max iterations, and convergence summary;
- tensor file path and checksum.

## Explorer Databases

The `databases/` folder should contain:

- a miniature SQLite database without full context rows;
- a full SQLite database for richer local browsing.

Database metadata must record which fitted-model artifact set it is compatible
with.
