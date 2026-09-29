# Long-running command policy

This policy applies to activation capture, ICA fitting, component profiling,
and experiments. A user should be able to stop any long command, repeat the
same command, and know what will happen without remembering command-specific
resume rules.

## User contract

### 1. Repeating the same command resumes automatically

There is no `--resume` flag. The output path identifies the run. When the
existing output has a compatible configuration, the command:

1. validates the existing metadata;
2. discovers durable completed work;
3. skips completed units;
4. continues with the first incomplete unit.

If everything requested is complete, the command reports that fact and exits
successfully without recomputing anything.

### 2. Configuration mismatches fail before expensive work

Resume is allowed only when inputs that affect the result match, including the
model and revision, dataset and revision, token selection, activation site,
preprocessing, algorithm settings, layers, and baseline definitions as
applicable.

If an output belongs to a different configuration, the command must stop with
a readable difference and ask for another output path. It must never mix two
configurations or silently replace the existing run.

### 3. Checkpoint at a reasonable durable boundary

A durable unit should balance restart cost against checkpoint overhead and
progress readability. Do not automatically choose the finest technically
possible unit. Prefer a boundary that saves a meaningful amount of expensive
work if interrupted, without producing excessive files, repeated
serialization, or a progress bar dominated by trivial operations. Internal
checkpoint granularity and the user-facing progress unit may differ; the
user-facing unit should still communicate meaningful overall progress.

Each command defines an independently reusable work unit:

| Command | Durable unit | Interrupted unit |
| --- | --- | --- |
| `icalens capture` | one captured layer | capture that layer again |
| `icalens fit` | one fitted layer | fit that layer again from iteration 1 |
| `icalens profile` | one profiled layer | profile that layer again |
| reconstruction experiment | one dataset × layer × method result | evaluate that unit again |
| sparse-probing experiment | one dataset × layer × method result | evaluate that unit again |
| targeted-probe-perturbation experiment | one layer result; SAEBench internally retains completed methods | repeat only the interrupted method(s), then assemble the layer result |
| autointerpretability preparation | one accepted fragment or one prepared layer | sample the current document again or prepare that layer again |
| autointerpretability evaluation | one explanation or one simulated fragment | request that explanation or simulation again |
| effective-receptive-field experiment | one completed layer | repeat the interrupted layer |

FastICA iteration state is not currently checkpointed. “Resume fitting” means
preserving completed layers and restarting only the interrupted layer.

Checkpoint writes should be atomic: write a temporary file, validate it, and
rename it into its final location. A partial file is not evidence of completed
work.

### 4. Existing completed work is skipped by default

The default is conservative and non-destructive. If a requested unit already
has a valid result for the same configuration, it is reused.

When deliberate recomputation is supported, `--force` means recompute only the
requested units and replace each old result only after its replacement is
complete. `--force` does not permit mixing incompatible run configurations;
use a new output directory for a different configuration.

Derived figures are the exception: they are cheap, so `--force` may directly
replace existing figure files.

### 5. Dependency changes invalidate downstream work

The tool should skip work only when its dependency fingerprint still matches.
For example:

- replacing a fitted ICA layer invalidates that layer's component profile;
- changing the profiling corpus or thresholds requires a new profile;
- changing an experiment method or preset requires a distinct experiment
  configuration;
- adding an R-lens readout may enrich an otherwise unchanged component profile
  without replaying its profiling corpus.

Presence of a file alone is insufficient: its recorded provenance and schema
must also be compatible.

### 6. Temporary data survives failure only when useful for resume

Reusable checkpoints and caches required to continue are retained after an
interruption. Temporary data is deleted only after the corresponding durable
result has been written successfully. A failed command must not delete the last
good checkpoint.

### 7. Progress starts from durable completed work

The compact terminal view reports:

- completed and total durable units;
- the current layer, dataset, and method when applicable;
- elapsed time for the current invocation;
- ETA estimated from work completed during the current invocation;
- the location of the complete log.

On resume, the progress bar begins at the number of units already complete. ETA
must not pretend that reused work was completed during the current invocation.

The shared terminal layout follows the experiment runner rather than exposing
raw nested progress bars:

1. an `Overall completed/total (percent)` row with elapsed time and ETA;
2. one progress bar for durable units;
3. the current phase and layer, dataset, or method;
4. a bounded `Recent output` window;
5. the path to the full log after success or failure.

Capture, fit, and profile use the same blue `ICA Lens` panel and layer-based
progress. Experiment panels may add dataset and method rows, but their timing,
resume, recent-output, and logging behavior follows the same rules.

### 8. Every long command keeps a complete log

Artifact-producing commands (`capture`, `fit`, and `profile`) keep logs outside
the artifact so `icalens publish` does not upload operational logs. Their logs
live under `.icalens-runs/`, whose exact path is printed when the command ends
or fails.

Experiment logs live under `<experiment-output>/logs/` because they are part of
the self-contained experiment record. Compact terminal output never replaces
the full log.

### 9. Lifecycle messages use one timestamp format

Capture, fit, profile, and experiment orchestration use local wall-clock
timestamps in the form `[YYYY-MM-DD HH:MM:SS ZONE]` for lifecycle messages:
configuration resolution, reuse/resume decisions, phase changes, checkpoints,
completion, warnings, and failures. The persistent run header separately keeps
an ISO-8601 `started_at` value and the exact command.

Progress-bar refreshes and verbatim third-party output are not prefixed with a
wall-clock timestamp. They already carry elapsed time or iteration state, and
timestamping every refresh would make logs noisy and can corrupt carriage-return
progress rendering. The compact overview likewise shows elapsed time and ETA
rather than changing its layout once per second to print wall-clock time.

### 10. Uncommitted source produces a visible, non-blocking warning

Before expensive work begins, every long command checks the Git checkout that
contains the running ICA Lens source. If tracked or untracked files are not
committed, the compact overview shows a dedicated `Source warning` panel
directly below the command, with the `WARNING` keyword highlighted in bright
yellow. The complete log retains the same warning as a timestamped lifecycle
message and records the dirty state in its run metadata. The run is still
allowed to proceed, but its exact source cannot be reconstructed from a commit
hash alone.

The warning applies consistently to capture, fit, profile, and experiment
runs. Installed wheels outside a Git checkout report the source state as
unknown and do not warn.

### 11. New implementations use the reusable framework

New long-running experiment implementations must use the reusable experiment
run/display framework in `src/icalens/experiments/_run.py` and
`src/icalens/experiments/_display.py`. They should extend those generic
interfaces when a new workflow shape is needed instead of creating another
command-specific run manifest, progress display, or logging wrapper.

This requirement does not implicitly migrate existing commands. Changes to an
existing runner remain separately reviewed so its resume and artifact semantics
are preserved.

## Current implementation audit

| Command | Current status | Required follow-up |
| --- | --- | --- |
| `capture text/chat` | Automatically captures missing layers and uses the shared compact view | Validate the final UI with long resumed runs |
| `fit activations` | Validates the artifact and automatically fits missing layers | Keep this as the reference behavior |
| `fit text/chat` | Checkpoints completed layers within one invocation | Make repeated invocations validate and skip completed layers, preferably by sharing capture/activation-fit machinery |
| `profile` from cached activations | Validates profile provenance, skips compatible completed layers, and supports `--force` for deliberate compatible regeneration | Keep this as the reference behavior |
| `profile refresh-statistics` | Validates statistics provenance and resumes from compatible refreshed layers | Keep aligned with the shared progress view |
| `profile` from a streamed dataset | Checkpoints each completed layer but recomputes it on rerun | Validate profile provenance before replay and skip compatible completed profiles |
| reconstruction | Durable capture resumes per dataset/layer; measurement validates `run.json` and resumes per dataset/layer/method with shared progress and logs | Validate the final UI with long resumed runs |
| sparse probing | Uses the reusable run/display framework, validates durable result schemas, resumes missing dataset--layer--method work, and atomically replaces summaries | Validate the final UI with a manually interrupted and resumed run |
| autointerpretability prepare/evaluate | Uses the reusable experiment run/display framework; checkpoints and validates fragments, prepared layers, explanations, and simulations; resumes with durable progress and complete logs | Validate the final UI on a manually interrupted and resumed pilot |
| experiment figures | Refuse replacement unless `--force` is supplied | Already aligned for derived outputs |

The reusable experiment framework is intentionally compatible with the other
long-run shapes in this table without migrating them yet. Its display accepts a
caller-defined unit label, arbitrary phase context (for example layer, dataset,
and method), an inherited worker start time, and optional hashable durable-unit
identities so a checkpoint is counted only once. Its run-state helper keeps result
configuration separate from lifecycle metadata and supports independently validated
configuration sections. Existing capture, fit, profile, and reconstruction runners
remain unchanged until a dedicated migration is reviewed.

Until the follow-up items are implemented, scripts must not assume that
`fit text/chat` or `profile` skips prior work merely because it checkpoints
within a single invocation.
