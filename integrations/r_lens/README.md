# R-lens integration

This directory contains tooling that produces R-lens artifacts consumed by
ICALens component profiling. R-lens fitting is a separate workflow from ICA
fitting, so its code lives outside `src/icalens`.

Generated weights belong under the gitignored root-level directory:

```text
local-r-lens-models/
├── official/
└── experiments/
```

## GPT-2 small

Install Anthropic's reference Jacobian Lens fitter once:

```bash
uv pip install "git+https://github.com/anthropics/jacobian-lens.git"
```

Fit the default 25-prompt pilot:

```bash
uv run python integrations/r_lens/fit_gpt2.py
```

GPT-2 targets its final transformer block (layer 11) by default. A
penultimate target can sometimes be better conditioned for larger models, but
it would unnecessarily omit GPT-2 layer 11 from the fitted transport maps.

Fit a stronger 100-prompt artifact:

```bash
uv run python integrations/r_lens/fit_gpt2.py \
  --prompts 100 \
  --dim-batch 32 \
  --output local-r-lens-models/official/gpt2-small/lens.pt
```

The GPT-2 implementation is architecture-specific. Do not use it for Gemma or
Qwen checkpoints: those models require separately validated RMSNorm and gated
MLP RelP rules.

## Pythia 70M

Pythia-70M uses GPT-NeoX LayerNorm and GELU blocks, so it follows the same
LayerNorm-only RelP protocol as GPT-2. The official command uses the pinned
model revision and the same 25-prompt Pile-10k sample as the other official
R-lenses:

```bash
uv run python integrations/r_lens/fit_pythia.py
```

The default target is Pythia-70M's final transformer block (layer 5), yielding
transport maps for all six residual layers. The fitter checkpoints after every
prompt and resumes automatically.

## Gemma 2 2B

Gemma uses the dense RelP LN, identity, and half rules. Its 2B checkpoint
targets the final transformer block by default:

```bash
uv run python integrations/r_lens/fit_gemma2.py
```

## Qwen 3.5 Base

The 2B Base checkpoint targets its final block; the larger 9B Base checkpoint
uses a penultimate target:

```bash
uv run python integrations/r_lens/fit_qwen35.py --model Qwen/Qwen3.5-2B-Base

uv run python integrations/r_lens/fit_qwen35.py --model Qwen/Qwen3.5-9B-Base
```
