"""Create paper-ready figures from saved ICA Lens experiment results."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="icalens experiment figure sparse-probing", description=__doc__
    )
    parser.add_argument(
        "experiment",
        nargs="+",
        help="Experiment directories. Use '-' for a pending panel.",
    )
    parser.add_argument("--format", default="png", help="Comma-separated formats: png,pdf.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <experiment>/figures).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--panel-titles",
        default=None,
        help="Comma-separated titles for a multi-panel figure.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiments = [
        None if item == "-" else Path(item).expanduser().resolve() for item in args.experiment
    ]
    formats = [item.strip().lower() for item in args.format.split(",") if item.strip()]
    if not formats or any(item not in {"png", "pdf"} for item in formats):
        raise ValueError("--format must contain png, pdf, or both")
    real_experiments = [item for item in experiments if item is not None]
    if not real_experiments:
        raise ValueError("at least one experiment directory is required")
    output = (
        real_experiments[0] / "figures"
        if args.output is None
        else args.output.expanduser().resolve()
    )
    titles = _parse_panel_titles(args.panel_titles, len(experiments))
    if len(experiments) == 1:
        payload = _load_payload(real_experiments[0])
        outputs = render_sparse_probing_figure(
            payload,
            output=output,
            stem=real_experiments[0].name,
            formats=formats,
            force=bool(args.force),
        )
    else:
        payloads = [_load_payload(item) if item is not None else None for item in experiments]
        outputs = render_sparse_probing_panels(
            payloads,
            titles=titles,
            output=output,
            stem="sparse-probing-model-comparison",
            formats=formats,
            force=bool(args.force),
        )
    for path in outputs:
        print(path)


def render_sparse_probing_figure(
    payload: dict[str, Any],
    *,
    output: Path,
    stem: str,
    formats: Sequence[str],
    force: bool,
) -> list[Path]:
    matplotlib_cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "plotting experiments requires matplotlib; install it with 'pip install matplotlib'"
        ) from error
    rows: list[dict[str, Any]] = payload.get("rows", [])
    if not rows:
        raise ValueError("experiment contains no completed sparse-probing rows")
    output.mkdir(parents=True, exist_ok=True)
    figure_paths = [output / f"{stem}.{extension}" for extension in formats]
    caption_path = output / f"{stem}.txt"
    for path in [*figure_paths, caption_path]:
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force to replace it")
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    layers = sorted({int(row["layer"]) for row in rows})
    methods = sorted({str(row.get("method", "ica")) for row in rows})
    if len(methods) > 1:
        for method in methods:
            method_points = _mean_rows_by_k(
                [row for row in rows if str(row.get("method", "ica")) == method]
            )
            axis.plot(
                [item[0] for item in method_points],
                [item[1] for item in method_points],
                marker="o",
                label={"ica": "ICA", "sae": "SAE", "pca": "PCA"}.get(method, method),
            )
    else:
        for layer in layers:
            layer_rows = sorted(
                (row for row in rows if int(row["layer"]) == layer),
                key=lambda row: int(row["k"]),
            )
            axis.plot(
                [int(row["k"]) for row in layer_rows],
                [float(row["mean_probe_accuracy"]) for row in layer_rows],
                marker="o",
                label=f"Layer {layer}",
            )
    axis.set_xscale("log")
    axis.set_xlabel("top-k features used by probe")
    axis.set_ylabel("mean probe accuracy")
    axis.set_title("SAEBench sparse probing")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    outputs: list[Path] = []
    for path in figure_paths:
        figure.savefig(path, dpi=180)
        outputs.append(path)
    plt.close(figure)
    caption_path.write_text(_sparse_probing_caption(payload, layers) + "\n", encoding="utf-8")
    outputs.append(caption_path)
    return outputs


def render_sparse_probing_panels(
    payloads: Sequence[dict[str, Any] | None],
    *,
    titles: Sequence[str],
    output: Path,
    stem: str,
    formats: Sequence[str],
    force: bool,
) -> list[Path]:
    """Render a formal, paper-style comparison across model families."""
    matplotlib_cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    figure_paths = [output / f"{stem}.{extension}" for extension in formats]
    caption_path = output / f"{stem}.txt"
    for path in [*figure_paths, caption_path]:
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force to replace it")

    legend_order = ("ica", "sae", "pca", "random")
    draw_order = ("random", "pca", "sae", "ica")
    styles = {
        "ica": ("ICA", "#3D5F99", "^"),
        "sae": ("SAE", "#B45F4D", "o"),
        "pca": ("PCA", "#5B8C6A", "^"),
        "random": ("Random", "#777777", "D"),
    }
    with plt.rc_context(_paper_style()):  # type: ignore[arg-type]
        figure, axes = plt.subplots(
            1, len(payloads), figsize=(7.15, 2.55), sharex=True, sharey=True
        )
        if len(payloads) == 1:
            axes = [axes]
        for index, (axis, payload, title) in enumerate(zip(axes, payloads, titles, strict=True)):
            axis.set_title(title, loc="left", fontweight="bold", pad=3)
            if payload is None:
                axis.text(
                    0.5,
                    0.5,
                    "Pending",
                    ha="center",
                    va="center",
                    color="#7a7a7a",
                    transform=axis.transAxes,
                )
            else:
                rows = payload.get("rows", [])
                for zorder, method in enumerate(draw_order, start=2):
                    points = _mean_rows_by_k(
                        [row for row in rows if str(row.get("method", "ica")) == method]
                    )
                    if not points:
                        continue
                    label, color, marker = styles[method]
                    axis.plot(
                        [item[0] for item in points],
                        [item[1] for item in points],
                        color=color,
                        marker=marker,
                        linewidth=1.35,
                        markersize=3.0,
                        label=label,
                        zorder=zorder,
                    )
                completed = sorted({int(row["layer"]) for row in rows})
                requested = payload.get("experiment", {}).get("layers", completed)
                if len(completed) < len(requested):
                    axis.text(
                        0.98,
                        0.04,
                        f"partial: layer {', '.join(map(str, completed))}",
                        ha="right",
                        va="bottom",
                        fontsize=6.4,
                        color="#6b7280",
                        transform=axis.transAxes,
                    )
            axis.set_xscale("log")
            tick_values = sorted(
                {
                    int(row["k"])
                    for payload in payloads
                    if payload is not None
                    for row in payload.get("rows", [])
                }
            )
            if not tick_values:
                tick_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
            axis.set_xticks(tick_values)
            axis.set_xticklabels([str(value) for value in tick_values], rotation=45, ha="right")
            axis.grid(axis="y", color="#e4e7eb", linewidth=0.45)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if index == 0:
                axis.set_ylabel("mean probe accuracy")
        figure.supxlabel("top-k features used by probe", y=0.04)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles, strict=True))
            ordered_labels = [styles[method][0] for method in legend_order]
            ordered_labels = [label for label in ordered_labels if label in by_label]
            figure.legend(
                [by_label[label] for label in ordered_labels],
                ordered_labels,
                loc="upper center",
                ncol=len(handles),
                frameon=False,
                bbox_to_anchor=(0.5, 1.01),
            )
        figure.subplots_adjust(left=0.085, right=0.995, bottom=0.25, top=0.79, wspace=0.16)
        for path in figure_paths:
            figure.savefig(path, dpi=300)
        plt.close(figure)

    caption = (
        "SAEBench sparse-probing performance for " + ", ".join(titles) + ". Curves average "
        "the completed evaluated layers for each model; panels marked partial exclude layers "
        "that are still running."
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return [*figure_paths, caption_path]


def _load_payload(experiment: Path) -> dict[str, Any]:
    results_path = experiment / "results.json"
    if results_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid results payload in {results_path}: expected an object")
    else:
        run_path = experiment / "run.json"
        if not run_path.is_file():
            raise FileNotFoundError(f"experiment results are missing: {results_path}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        resolved = run.get("resolved", {})
        from icalens.experiments.saebench_sparse_probing import collect_result_rows

        payload = {
            "experiment": resolved,
            "rows": collect_result_rows(experiment, resolved.get("layers", [])),
        }
    if payload.get("experiment", {}).get("experiment") != "saebench-sparse-probing":
        raise ValueError(f"unsupported experiment type in {experiment}")
    return payload


def _parse_panel_titles(value: str | None, count: int) -> list[str]:
    if value is None:
        return [f"Model {index + 1}" for index in range(count)]
    titles = [item.strip() for item in value.split(",")]
    if len(titles) != count or any(not item for item in titles):
        raise ValueError(f"--panel-titles must contain exactly {count} comma-separated titles")
    return titles


def _paper_style() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _mean_rows_by_k(rows: Sequence[dict[str, Any]]) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["k"]), []).append(float(row["mean_probe_accuracy"]))
    return [(k, sum(values) / len(values)) for k, values in sorted(grouped.items())]


def _sparse_probing_caption(payload: dict[str, Any], layers: Sequence[int]) -> str:
    experiment = payload.get("experiment", {})
    preset = experiment.get("preset", {})
    lens = experiment.get("lens", "unknown ICA Lens")
    layer_text = ", ".join(str(layer) for layer in layers)
    preset_name = preset.get("name", "custom")
    datasets = ", ".join(str(item) for item in preset.get("datasets", []))
    dataset_text = f" on {datasets}" if datasets else ""
    methods = sorted({str(row.get("method", "ica")).upper() for row in payload.get("rows", [])})
    method_text = ", ".join(methods)
    return (
        f"SAEBench sparse-probing performance for {lens}, layer(s) {layer_text}, using the "
        f"{preset_name} preset{dataset_text}. Mean probe accuracy is shown against the number "
        f"of top-ranked features used by the linear probe. Methods: {method_text}. ICA "
        "coordinates are profile-oriented and signed; linear dictionaries use one feature per axis."
    )


if __name__ == "__main__":
    main()
