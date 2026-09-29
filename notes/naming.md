## Project names

```text
Brand:              ICA Lens
PyPI distribution:  icalens
Python import:       icalens
Main class:          ICALens
GitHub repository:   icalens
Documentation:       icalens.readthedocs.io
HF artifacts:        sida/icalens-gpt2-small
```

## Hugging Face repositories

Fitted ICA artifacts are published as Hugging Face **Model repositories**, not
Dataset repositories. Although an ICA Lens is not a language model checkpoint,
it contains fitted parameters that are applied to model activations.

Use one artifact repository per base model:

```text
sida/icalens-gpt2-small
sida/icalens-gemma-2-2b
sida/icalens-qwen-3.5-2b
```

Repository IDs are lowercase and follow `icalens-<model-name>`.

Activation samples, annotations, evaluation examples, and explorer databases
are data rather than fitted parameters. If published on Hugging Face, they
belong in a separate Dataset repository such as
`sida/icalens-paper-data`.
