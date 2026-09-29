"""The unified ICA Lens command-line interface."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch an ICA Lens subcommand."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_TOP_HELP)
        return

    command = args.pop(0)
    if command == "fit":
        _dispatch_fit(args)
    elif command == "capture":
        _dispatch_capture(args)
    elif command == "publish":
        from .publish import main as publish_main

        publish_main(args)
    elif command == "profile":
        from ._live_output import compact_output
        from .profile import main as profile_main

        with compact_output("ICA Lens · component profiling", args):
            profile_main(args)
    elif command == "plot":
        _dispatch_plot(args)
    elif command == "smoke-test":
        from ..smoke_test import main as smoke_test_main

        smoke_test_main(args)
    elif command == "integrity":
        _dispatch_integrity(args)
    elif command == "experiment":
        _dispatch_experiment(args)
    else:
        raise SystemExit(f"icalens: unknown command {command!r}; run 'icalens --help'")


def _dispatch_integrity(args: list[str]) -> None:
    if not args or args[0] in {"-h", "--help"}:
        print(_INTEGRITY_HELP)
        return
    operation = args.pop(0)
    if operation != "reproduce":
        raise SystemExit(f"icalens integrity: unknown operation {operation!r}; use 'reproduce'")
    from .integrity import main

    main(args)


def _dispatch_experiment(args: list[str]) -> None:
    if not args or args[0] in {"-h", "--help"}:
        print(_EXPERIMENT_HELP)
        return
    kind = args.pop(0)
    if kind == "saebench-sparse-probing":
        from ..experiments.saebench_sparse_probing import main

        main(args)
    elif kind == "reconstruction":
        from ..experiments.reconstruction import main

        main(args)
    elif kind == "autointerpretability":
        from ..experiments.autointerpretability import main

        main(args)
    elif kind == "erf-gradient":
        from ..experiments.erf_gradient import main

        main(args)
    elif kind == "erf-suffix-sweep":
        from ..experiments.erf_suffix_sweep import main

        main(args)
    elif kind == "figure":
        _dispatch_experiment_figure(args)
    else:
        raise SystemExit(
            f"icalens experiment: unknown experiment {kind!r}; "
            "use 'saebench-sparse-probing', 'reconstruction', "
            "'autointerpretability', 'erf-gradient', 'erf-suffix-sweep', or 'figure'"
        )


def _dispatch_plot(args: list[str]) -> None:
    if not args or args[0] in {"-h", "--help"}:
        print(_PLOT_HELP)
        return
    kind = args.pop(0)
    if kind == "fitting-summary":
        from .plot_fitting import main as plot_fitting_main

        plot_fitting_main(args)
    else:
        raise SystemExit(f"icalens plot: unknown plot {kind!r}; use 'fitting-summary'")


def _dispatch_experiment_figure(args: list[str]) -> None:
    if not args or args[0] in {"-h", "--help"}:
        print(_EXPERIMENT_FIGURE_HELP)
        return
    kind = args.pop(0)
    if kind == "sparse-probing":
        from ..experiments.figure import main

        main(args)
    elif kind == "reconstruction":
        from ..experiments.reconstruction_figure import main

        main(args)
    elif kind == "autointerpretability":
        from ..experiments.autointerpretability_figure import main

        main(args)
    elif kind == "erf-gradient":
        from ..experiments.erf_gradient_figure import main

        main(args)
    elif kind == "erf-suffix-sweep":
        from ..experiments.erf_suffix_sweep_figure import main

        main(args)
    else:
        raise SystemExit(
            f"icalens experiment figure: unknown figure {kind!r}; "
            "use 'sparse-probing', 'reconstruction', 'autointerpretability', "
            "'erf-gradient', or 'erf-suffix-sweep'"
        )


def _dispatch_fit(args: list[str]) -> None:
    if not args or args[0] in {"-h", "--help"}:
        print(_FIT_HELP)
        return
    kind = args.pop(0)
    if kind == "text":
        from ._live_output import compact_output
        from .fit_text import main as fit_text_main

        with compact_output("ICA Lens · text fitting", args):
            fit_text_main(args)
    elif kind == "chat":
        from ._live_output import compact_output
        from .fit_chat import main as fit_chat_main

        with compact_output("ICA Lens · chat fitting", args):
            fit_chat_main(args)
    elif kind == "activations":
        from ._live_output import compact_output
        from .fit_activations import main as fit_activations_main

        with compact_output("ICA Lens · activation fitting", args):
            fit_activations_main(args)
    else:
        raise SystemExit(
            f"icalens fit: unknown input type {kind!r}; use 'text', 'chat', or 'activations'"
        )


def _dispatch_capture(args: list[str]) -> None:
    if not args or args[0] in {"-h", "--help"}:
        print(_CAPTURE_HELP)
        return
    kind = args.pop(0)
    if kind not in {"text", "chat"}:
        raise SystemExit(f"icalens capture: unknown input type {kind!r}; use 'text' or 'chat'")
    from ._live_output import compact_output
    from .capture import main as capture_main

    with compact_output(f"ICA Lens · {kind} activation capture", args):
        capture_main(kind, args)


_TOP_HELP = """usage: icalens COMMAND [OPTIONS]

Fit, profile, publish, and verify ICA Lens artifacts.

commands:
  capture text Capture reusable activations from raw text
  capture chat Capture reusable activations from conversations
  fit text     Fit from a raw-text dataset
  fit chat     Fit from a conversation dataset
  fit activations  Fit from a reusable activation dataset
  profile      Add component statistics, examples, and logit-lens tokens
  plot         Plot diagnostics from saved ICA Lens artifacts
  publish      Publish a local artifact to Hugging Face
  smoke-test   Verify installed text and chat analysis paths
  integrity    Reproduce and compare a reference artifact
  experiment   Run experiments and create paper-ready figures

Run 'icalens COMMAND --help' for command-specific options."""

_INTEGRITY_HELP = """usage: icalens integrity reproduce [OPTIONS]

Recapture, refit, reprofile, and compare one reference ICA Lens layer."""

_PLOT_HELP = """usage: icalens plot PLOT [OPTIONS]

Plot diagnostics from saved ICA Lens artifacts.

plots:
  fitting-summary  Compare layer-averaged FastICA fitting curves

Run 'icalens plot PLOT --help' for plot-specific options."""

_FIT_HELP = """usage: icalens fit {text,chat,activations} [OPTIONS]

Fit an ICA Lens from model activations.

input types:
  text         Raw-text dataset fitting
  chat         Conversation dataset fitting
  activations  Reusable disk-backed activation fitting

Run 'icalens fit INPUT_TYPE --help' for fitting options."""

_CAPTURE_HELP = """usage: icalens capture {text,chat} [OPTIONS]

Capture model activations once, stream them to disk, and reuse them across fits.

input types:
  text         Capture from a raw-text dataset
  chat         Capture from a conversation dataset

Run 'icalens capture INPUT_TYPE --help' for capture options."""

_EXPERIMENT_HELP = """usage: icalens experiment COMMAND [OPTIONS]

Run reproducible ICA Lens paper experiments.

commands:
  saebench-sparse-probing  Run SAEBench sparse probing for an ICA Lens
  reconstruction          Capture or evaluate held-out top-k reconstruction
  autointerpretability     Explain features and predict held-out activations
  erf-gradient             Measure gradient effective receptive fields
  erf-suffix-sweep         Measure suffix-sweep effective receptive fields
  figure                   Create a paper-ready figure from saved results

Run 'icalens experiment COMMAND --help' for command-specific options."""

_EXPERIMENT_FIGURE_HELP = """usage: icalens experiment figure FIGURE [OPTIONS]

Create paper-ready figures from saved experiment results.

figures:
  sparse-probing  Plot one or more SAEBench sparse-probing runs
  reconstruction  Plot held-out reconstruction curves
  autointerpretability  Plot feature-level explanation-prediction correlations
  erf-gradient          Plot layerwise gradient ERF distributions
  erf-suffix-sweep      Plot layerwise suffix-sweep ERF distributions

Run 'icalens experiment figure FIGURE --help' for figure-specific options."""
