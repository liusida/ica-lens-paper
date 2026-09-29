# Official reconstruction experiment

This is the reconstruction experiment accepted for the paper. It separates durable
activation capture from reconstruction measurement. The language model is used only
by the capture scripts; measurement can be repeated from the HDD captures without
another model forward pass.

Durable activations are stored outside the repository under:

```text
~/Expansion/research/ICA-data/icalens-reconstruction-activations/
```

Run the three `capture-*.sh` scripts first. A repeated capture validates the existing
manifest and resumes missing layers. Then run the three `measure-*.sh` scripts and
`make-figures.sh`. Measurement uses requested budgets `1,3,10,30,100,300`; complete
linear-basis endpoints and native SAE reconstruction are added by the evaluator.

Official aggregate results, figures, captions, and reproduction scripts are retained
in Git. Activation binaries, operational logs, and temporary checkpoints are excluded.
