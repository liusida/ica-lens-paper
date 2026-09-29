# Experiment artifact integrity

## Purpose

The repository-level `integrity/` suite detects when current code no longer
reproduces accepted experimental artifacts. It is primarily a refactoring safety
system: after changing shared capture, fitting, adapter, experiment,
data-preparation, or plotting code, run the relevant check before deciding that
the existing results remain current.

Accepted artifacts are references; the verifier never silently replaces them.
A mismatch means that downstream artifacts may be stale. It does not decide
whether the old or new behavior is scientifically correct. Investigate the
difference, then either preserve the previous behavior or deliberately rerun the
affected experiment and update its descendants.

The verifier is deliberately outside `src/icalens`. It invokes production code
where appropriate but independently selects references, compares outputs, writes
reports, and maps failures to artifact dependencies.

## Verification levels

- **Numerical replay:** current production logic is rerun on a natural retained
  input slice and compared numerically with an accepted output.
- **Aggregation replay:** accepted fine-grained measurements are inputs; current
  aggregation or collection logic must reproduce the accepted summary.
- **Identity preflight:** revisions, hashes, schemas, shapes, and configuration
  relations are checked, but numerical production is not rerun.
- **Assumed valid:** the artifact is a declared trust root whose producer cannot
  currently be reproduced.
- **Skipped:** no honest independent check is currently available.

Aggregation replay protects summarization, collection, and serialization. It
does not validate the model computation that produced the retained fine-grained
measurements. Keep this distinction visible in documentation and reports.

## Running the verifier

The model canaries are GPT-2 Small layer 6, Gemma 2 2B layer 13, and Qwen 3.5 9B
Base layer 16.

```bash
# Fast identity check for the default GPT-2 canary
uv run python -m integrity.run --check reference-preflight

# Shared ICA pipeline stages
uv run python -m integrity.run --model all --check capture
uv run python -m integrity.run --model all --check fit
uv run python -m integrity.run --model all --check profile

# Shared SAE adapter plus activation-pattern replay
uv run python -m integrity.run --model all --check sae

# Downstream experiments, twin data, and representative rendering
uv run python -m integrity.run --check downstream

# Every implemented check
uv run python -m integrity.run --model all --check all
```

Each operation prints a flushed `START` line, timestamp, final status, dependency
IDs, and elapsed time. Detailed output goes to its `runtime.log`. Reports are
stored under `integrity/runs/<source-fingerprint>/`; the fingerprint includes
committed, modified, and untracked files, so results from another workspace state
are not silently reused.

The runner accepts `--verbose`, tolerances, verification sample sizes, and
single-model reference overrides. See `uv run python -m integrity.run --help`.

## Trust boundaries and artifact graph

For `A --B--> C`, data `A` is consumed by code `B` to produce artifact `C`.
Verify upstream producers first. If `A` has passed and `C` changes, the evidence
points more strongly to `B`; if `A` has not passed, a downstream mismatch cannot
be localized confidently.

```text
pinned model/tokenizer/dataset
  -- C10 capture --> D10 fitting activations
  -- C11 fitting --> D11 ICA Lens tensors
  -- C12 profiling + D10 + D15 --> D12 component profiles

shared artifacts + public SAE checkpoints
  -- C20...C30 --> D20...D29 experiment results
  -- C40 --> D30 paper twin data
  -- C41 --> D31 paper figures
  -- C42 --> D32 main.pdf
```

### Data artifacts

| ID | Artifact | Producer or resolver | Main consumers |
| --- | --- | --- | --- |
| `D01` | Pinned language-model weights/configuration | Recorded Hugging Face revision | Capture and model experiments |
| `D02` | Pinned tokenizer | Recorded model/tokenizer revision | Tokenization and text experiments |
| `D03` | Pinned text or task dataset | Recorded revision and selection policy | Capture and evaluation |
| `D04` | Public SAE checkpoint and metadata | Baseline registry/checkpoint resolver | C13 and SAE comparisons |
| `D05` | External evaluation backend or human protocol | Pinned backend/model/protocol identity | Evaluation experiments |
| `D06` | Cunningham reproduction artifacts | Separate reproduction repository | Historical reproduction figures |
| `D10` | Official ICA activation cache | `C10` | Fitting, profiling, stability, SAE ERF selection |
| `D11` | ICA center, reading matrix, writing matrix | `C11` | Most ICA experiments |
| `D12` | ICA component profiles | `C12` | Inspection, annotation, ERF context recovery |
| `D13` | SAE codes and decoder interface | `C13` | SAE comparisons |
| `D14` | Held-out reconstruction activations | `C14` | Reconstruction |
| `D15` | Official R-lens source maps | `C15` | R-lens profile readouts |
| `D20` | Toy-example results | `C20` | Figure 2 twin data |
| `D21` | Reconstruction results | `C21` | Reconstruction figures |
| `D22` | Sparse-probing results | `C22` | Sparse-probing figure |
| `D23` | Autointerpretability results | `C23` | Reproduction/robustness figures |
| `D24` | ICA and SAE ERF results | `C24`, `C25` | ERF figures |
| `D25` | Fitting-stability results | `C26` | Seed/iteration figures |
| `D26` | ICA-to-SAE overlap results | `C27` | Overlap figures |
| `D27` | Activation-pattern results | `C28` | Activation-pattern figures |
| `D28` | Manual-annotation results | `C29` | Confidence figure |
| `D29` | Steering/language-control results | `C30` | Steering analysis |
| `D33` | Targeted-probe-perturbation results | `C31` | TPP analysis and figures |
| `D30` | Paper twin data | `C40` | Figure renderers |
| `D31` | Paper PDF/PNG figures | `C41` | LaTeX paper |
| `D32` | Paper `main.pdf` | `C42` | Submission artifact |

### Code transformations

| ID | Transformation |
| --- | --- |
| `C10` | Tokenization, framing, sampling, and residual capture |
| `C11` | ICA fitting and component ordering |
| `C12` | Profiling, orientation, statistics, readouts, and occurrences |
| `C13` | Public SAE loading, preprocessing, encoding, and decoding in the shared `icalens.experiments._sae` module |
| `C14` | Held-out reconstruction tokenization, selection, and capture |
| `C15` | RelP R-lens fitting in `integrations/r_lens/` |
| `C20` | Toy-example construction and measurement |
| `C21` | Reconstruction measurement and aggregation |
| `C22` | Sparse-probing orchestration, adapters, and aggregation |
| `C23` | Autointerpretability selection, prompting, simulation, scoring |
| `C24` | ICA suffix-sweep and gradient ERF measurement |
| `C25` | SAE suffix-sweep ERF measurement |
| `C26` | Fitting-seed/iteration stability and matching analysis |
| `C27` | ICA-to-SAE overlap and matched-random calculation |
| `C28` | Tokenwise ICA/SAE activation-pattern measurement |
| `C29` | Annotation sampling/aggregation; judgments remain external |
| `C30` | Steering intervention, generation, and measurement |
| `C31` | SAEBench targeted probe perturbation, fitted/unfitted dictionary adapters, and aggregation |
| `C40` | Paper twin-data extraction and validation |
| `C41` | Figure rendering and style application |
| `C42` | LaTeX assembly |

## Implemented checks

| Relation | Dependency relation | Implemented check | Level and scope |
| --- | --- | --- | --- |
| ICA capture | `D01 + D02 + D03 --C10--> D10` | `C10-capture-{model}-layer{layer}` | Numerical replay: representative layer per model, 32 deterministic rows, exact metadata and bfloat16 activations |
| ICA fitting | `D10 --C11--> D11` | `C11-fit-{model}-layer{layer}` | Numerical replay: full layer using all fitting rows/components, deterministic comparison fragment |
| R-lens fitting | `D01 + D02 + D03 --C15--> D15` | Reference preflight | **Assumed valid:** profile-recorded D15 hashes must match; fitting environment/revision was not retained |
| ICA profiling | `D10 + D11 + D15 --C12--> D12` | Three `C12-profile-*` checks | Numerical replay: statistics, orientation, logit/R-lens readouts, context recovery, occurrence search, and ranks |
| Shared SAE adapter | `D01 + D02 + D04 --C13--> D13` | `C13-sae-adapter-{model}-layer{layer}` | Numerical replay for GPT-2 TopK-32, Gemma JumpReLU, and Qwen TopK-50 SAEs: accepted native activations and feature identities, plus independently computed decoder normalization, coefficient scaling, and decoding |
| Toy example | `D01 + D02 --C20--> D20` | `C20-toy-example` | Full numerical analysis replay from retained vocabulary activations |
| Reconstruction capture | `D01 + D02 + D03 --C14--> D14` | None | **Skipped:** independent recapture for six heterogeneous dataset loaders is not implemented |
| Reconstruction | `D14 + D11 + D04 --C13+C21--> D21` | `C21-reconstruction-aggregation-gpt2-context64` | Aggregation replay of every layer from retained per-dataset metrics; C13 covered separately |
| Sparse probing | `D01 + D02 + D03 + D11 + D04 + D05 --C13+C22--> D22` | `C22-sparse-probing-aggregation-gpt2` | Production row-collection replay plus an independent numerical check that unit decoder rows receive exact ICA score compensation; probing itself is not rerun |
| Autointerpretability | `D01 + D02 + D03 + D11 + D12 + D04 + D05 --C13+C23--> D23` | `C23-autointerpretability-aggregation-gpt2` | Aggregation replay; accepted external LLM responses are trusted inputs |
| ICA ERF | `D01 + D02 + D03 + D11 + D12 --C24--> D24` | `C24-ica-erf-aggregation-gpt2-layer6` | Component/threshold aggregation replay for GPT-2 layer 6 |
| SAE ERF | `D01 + D02 + D10 + D04 --C13+C25--> D24` | `C25-sae-erf-aggregation-gpt2-layer6` | Component/threshold validation for GPT-2 layer 6; C13 covered separately |
| Fitting stability | `D10 + D11 --C11+C26--> D25` | `C26-fitting-seed-stability-aggregation` | Successive-budget aggregation replay; fitting covered by C11 |
| Directional overlap | `D11 + D04 --C27--> D26` | `C27-directional-overlap-gpt2-layer6` | Direct numerical replay of every ICA-to-SAE nearest cosine and feature ID, plus the matched-random null, for GPT-2 layer 6 |
| Activation patterns | `D01 + D02 + D11 + D04 --C13+C28--> D27` | `C13-sae-adapter-{model}-layer{layer}` | Numerical replay of token IDs, selected ICA/SAE responses, and top-feature identities for all models |
| Manual annotation | `D12 + D05 --C29--> D28` | None | Not covered |
| Steering | `D01 + D02 + D11 --C30--> D29` | None | **Skipped:** stochastic generation and external judgments lack an independent numerical oracle |
| Targeted probe perturbation | `D01 + D02 + D03 + D11 + D04 + D05 --C13+C31--> D33` | None | Not yet covered; promoted experiment code has independent initialization checks, but no accepted official artifact exists yet |
| Twin-data preparation | `D20...D29 --C40--> D30` | `C40-twin-data-directional-overlap` | Representative exact-array replay for both directional-overlap twin-data files |
| Figure rendering | `D30 --C41--> D31` | `C41-render-directional-overlap` | Representative exact-raster replay for the overlap-distribution figure |
| LaTeX assembly | `D31 --C42--> D32` | None | Not covered |

`--check downstream` writes current C14/C30 skip explanations to a temporary
file and prints its path. A skipped relation does not fail implemented checks,
but it must remain visible here.

## Interpreting failures

- C10 failure: downstream activation-based checks are inconclusive until capture
  is resolved.
- C11 failure with C10 passing: existing ICA Lenses and descendants may be stale.
- C12 failure with C10/C11 passing: investigate profiling, readout, or occurrence
  logic.
- C13 failure: every experiment using `SAEFeatureEncoder` may be stale.
- Experiment aggregation failure with shared inputs passing: investigate that
  experiment's collection, aggregation, or serialization.
- C40 failure with raw results passing: raw results may remain valid, but paper
  twin data may be stale.
- C41 failure with twin data passing: investigate plotting/style code; the raw
  experiment generally need not be rerun.

External revisions are trust roots only while their immutable identities match.
Never describe schema or aggregation validation as a replay of external API or
human judgments.

## Maintenance rules

1. Find the changed path's `C` ID and downstream `D` artifacts in this map.
2. Run upstream checks first, then the affected check, then C40/C41 when paper
   artifacts consume the result.
3. Use a meaningful natural slice already present in accepted artifacts. Do not
   require remembering to create a special pre-refactor baseline.
4. Compare intermediate identities and arrays as well as final means whenever
   retained artifacts permit it.
5. Never update an accepted reference merely to make a failure disappear.
6. If implementation work reveals an incorrect dependency, update this map in
   the same change.
7. Keep pilots outside the integrity gate unless they become accepted evidence.

Ordinary unit tests remain necessary, but they do not establish that accepted
experimental numbers are still reproducible.
