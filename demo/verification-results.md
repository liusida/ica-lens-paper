# Colab demo verification

Verified locally on an NVIDIA GB10 with `icalens==0.4.0` on 2026-09-30.

## GPT-2 toy example

- Captured shape: `50256 x 768`
- FastICA: 50 iterations
- Selected component: C45
- Excess kurtosis: 294.5
- Related tokens among the top 20: 12/20
- The top tail contained the paper's `neither`, `either`, `both`, and `between` family.

## Collect, fit, profile, apply

- GPT-2 Small layer 8, 100,000 Pile-10k activations, 50 iterations
- Collect: 0.20 minutes
- Fit: 0.13 minutes
- Profile: 0.50 minutes
- Applied the freshly fitted lens to held-out text: 3.2 seconds
- Result shape: `15 x 768`

Times are specific to the verification machine and are not notebook runtime guarantees.

## GPT-2 L8/C396

Mean of the three highest scores on the stored negative semantic tail:

| Prompt | Score |
| --- | ---: |
| Programming / module specification | 6.21 |
| Programming / API | 9.72 |
| Programming / schema | 5.70 |
| Travel control | 0.31 |
| Cooking control | 1.17 |
| History control | 1.08 |

## Qwen Table 8 intervention

- Model: Qwen 3.5 9B Base
- Component: layer 26, C9
- English baseline continued in English.
- Applying `initial_steer=(9, -40, -40)` and `steer=(9, -20, -20)` continued in French.
- The French baseline continued in French.
- Clamping C9 to zero switched the French prompt to an English continuation.
