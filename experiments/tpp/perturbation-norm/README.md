# TPP perturbation norm (diagnostic, not used in the rebuttal)

**Status:** exploratory. These results are not used in the rebuttal or the paper.

Question: at the same k, do ICA and SAE top-k ablations perturb the activation by
similar amounts? `compute.py` recomputes TPP feature selection from cached SAEBench
activations and measures Δx = decode(f) − decode(f with top-k zeroed).

Run: Gemma 2 2B, layer 12, seed 0, 200 examples per class and split (light mode).
The TPP scores recomputed on this subset match the stored ones closely
(ICA k=20: 0.200 vs 0.190; SAE k=20: 0.055 vs 0.053).

| k | ICA ‖Δx‖/‖x‖ | ICA TPP | SAE ‖Δx‖/‖x‖ | SAE TPP |
|---|---|---|---|---|
| 10 | 0.054 | 0.097 | 0.014 | 0.027 |
| 20 | 0.081 | 0.190 | 0.036 | 0.053 |
| 50 | 0.125 | 0.360 | 0.069 | 0.100 |
| 100 | 0.170 | 0.375 | 0.099 | 0.154 |

- At equal k, ICA ablations are 2–6× larger in norm.
- At matched norm, ICA's target effect equals or exceeds the SAE's.
- At matched non-target (collateral) drop, the SAE removes more of the target,
  so this does not support a selectivity claim for ICA.

Usage (SAEBench venv, from the repo root):

    $SAEBENCH/.venv/bin/python experiments/tpp/perturbation-norm/compute.py \
        --run <official results>/seed-0/gemma2 --layer 12 --device cuda \
        --max-per-class 200 --output experiments/tpp/perturbation-norm/results/gemma2-layer12-seed0.json
