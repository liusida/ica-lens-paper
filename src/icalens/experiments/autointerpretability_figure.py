"""Create paper-ready figures from saved autointerpretability results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

METHODS = ("ica", "sae")
STYLES = {
    "ica": ("ICA", "#3D5F99", "D"),
    "sae": ("SAE", "#B45F4D", "o"),
}
MODEL_TITLES = {
    "openai-community/gpt2": "GPT-2 small",
    "google/gemma-2-2b": "Gemma 2 2B",
    "Qwen/Qwen3.5-9B-Base": "Qwen 3.5 9B Base",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="icalens experiment figure autointerpretability", description=__doc__
    )
    parser.add_argument("experiment", nargs="+", help="Evaluation result directories.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", default="png,pdf", help="Comma-separated formats: png,pdf.")
    parser.add_argument("--panel-titles", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiments = [Path(value).expanduser().resolve() for value in args.experiment]
    payloads = _merge_model_payloads([_load(path) for path in experiments])
    formats = [value.strip().lower() for value in args.format.split(",") if value.strip()]
    if not formats or any(value not in {"png", "pdf"} for value in formats):
        raise ValueError("--format must contain png, pdf, or both")
    output = experiments[0] / "figures" if args.output is None else args.output.resolve()
    titles = _titles(args.panel_titles, payloads)
    outputs = render(payloads, titles=titles, output=output, formats=formats, force=args.force)
    for path in outputs:
        print(path)


def render(
    payloads: Sequence[dict[str, Any]],
    *,
    titles: Sequence[str],
    output: Path,
    formats: Sequence[str],
    force: bool,
) -> list[Path]:
    if len(payloads) != len(titles):
        raise ValueError("each result must have exactly one panel title")
    panels = [_panel_rows(payload) for payload in payloads]
    if not panels or not any(row["scores"].size for panel in panels for row in panel):
        raise ValueError("autointerpretability results contain no finite feature scores")
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"autointerpretability.{suffix}" for suffix in formats]
    companion = output / "autointerpretability.txt"
    for path in [*paths, companion]:
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force to replace it")

    cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    all_scores = np.concatenate(
        [row["scores"] for panel in panels for row in panel if row["scores"].size]
    )
    single_layer = all(len({row["layer"] for row in panel}) == 1 for panel in panels)
    lower, upper = _shared_limits(all_scores) if single_layer else (0.0, 1.0)
    width = 3.35 if len(panels) == 1 else 6.9
    height = 2.65 if single_layer else 2.85
    with plt.rc_context(_paper_style()):  # type: ignore[arg-type]
        figure, axes = plt.subplots(
            1, len(panels), figsize=(width, height), sharey=True, squeeze=False
        )
        axes_list = list(axes.ravel())
        for panel_index, (axis, rows, title) in enumerate(
            zip(axes_list, panels, titles, strict=True)
        ):
            if single_layer:
                _draw_distribution_panel(axis, rows, panel_index)
            else:
                _draw_layer_panel(axis, rows, panel_index)
            axis.set_title(title, loc="left", fontweight="bold", pad=3)
            axis.set_ylim(lower, upper)
            axis.axhline(0, color="#777777", linewidth=0.65, zorder=0)
            axis.grid(axis="y", color="#E1E1E1", linewidth=0.5)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if panel_index == 0:
                axis.set_ylabel("Activation prediction correlation")

        if not single_layer:
            figure.supxlabel("Layer", y=0.055)
        handles = [
            Line2D(
                [0],
                [0],
                color=STYLES[method][1],
                marker=STYLES[method][2],
                linewidth=1.5,
                markersize=3.5,
                label=STYLES[method][0],
            )
            for method in METHODS
            if any(
                row["method"] == method and row["scores"].size for panel in panels for row in panel
            )
        ]
        if handles:
            figure.legend(
                handles=handles,
                loc="upper center",
                ncol=len(handles),
                frameon=False,
                bbox_to_anchor=(0.5, 1.015),
            )
        bottom = 0.2 if single_layer else 0.24
        figure.subplots_adjust(
            left=0.17 if len(panels) == 1 else 0.09,
            right=0.985,
            bottom=bottom,
            top=0.82,
            wspace=0.18,
        )
        for path in paths:
            figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)

    companion.write_text(_companion(payloads, titles, panels), encoding="utf-8")
    return [*paths, companion]


def _draw_distribution_panel(axis: Any, rows: list[dict[str, Any]], seed: int) -> None:
    positions = {"ica": 0.0, "sae": 1.0}
    for row in rows:
        scores = row["scores"]
        if not scores.size:
            continue
        method = row["method"]
        label, color, marker = STYLES[method]
        rng = np.random.default_rng(_seed(seed, row["layer"], method))
        jitter = rng.uniform(-0.10, 0.10, size=len(scores))
        axis.scatter(
            positions[method] + jitter,
            scores,
            color=color,
            marker=marker,
            s=12,
            alpha=0.42,
            linewidths=0,
            zorder=2,
        )
        mean, low, high = _bootstrap_mean(scores, _seed(17 + seed, row["layer"], method))
        axis.errorbar(
            positions[method],
            mean,
            yerr=[[mean - low], [high - mean]],
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=5,
            elinewidth=1.25,
            capsize=3,
            zorder=4,
        )
        if row["defined"] != row["selected"]:
            axis.annotate(
                f"n={row['defined']}/{row['selected']}",
                (positions[method], high),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color=color,
            )
    axis.set_xticks([positions[method] for method in METHODS])
    axis.set_xticklabels([STYLES[method][0] for method in METHODS])
    axis.set_xlabel(f"Layer {rows[0]['layer']}")


def _draw_layer_panel(axis: Any, rows: list[dict[str, Any]], seed: int) -> None:
    layers = sorted({row["layer"] for row in rows})
    for method in METHODS:
        method_rows = sorted(
            (row for row in rows if row["method"] == method), key=lambda row: row["layer"]
        )
        observed = [row for row in method_rows if row["scores"].size]
        if not observed:
            continue
        label, color, marker = STYLES[method]
        means, lows, highs = zip(
            *[
                _bootstrap_mean(row["scores"], _seed(seed, row["layer"], method))
                for row in observed
            ],
            strict=True,
        )
        x = np.asarray([row["layer"] for row in observed], dtype=float)
        means_array = np.asarray(means)
        axis.errorbar(
            x,
            means_array,
            yerr=[means_array - np.asarray(lows), np.asarray(highs) - means_array],
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=3.5,
            capsize=2.5,
            zorder=4,
        )
        for row in observed:
            if row["defined"] != row["selected"]:
                mean = float(row["scores"].mean())
                axis.annotate(
                    f"n={row['defined']}/{row['selected']}",
                    (row["layer"], mean),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=6.2,
                    color=color,
                )
    axis.set_xticks(layers)


def _panel_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    features = payload["results"].get("features", [])
    conditions = {
        (int(row["layer"]), str(row["method"])): row
        for row in payload["results"].get("conditions", [])
    }
    rows: list[dict[str, Any]] = []
    layers = [int(value) for value in payload["run"]["resolved"]["layers"]]
    methods = [str(value) for value in payload["run"]["resolved"]["methods"]]
    for layer in layers:
        for method in METHODS:
            if method not in methods:
                continue
            values = [
                float(row["combined_score"])
                for row in features
                if int(row["layer"]) == layer
                and str(row["method"]) == method
                and np.isfinite(row.get("combined_score", float("nan")))
            ]
            condition = conditions.get((layer, method), {})
            selected = int(condition.get("selected_features", len(values)))
            rows.append(
                {
                    "layer": layer,
                    "method": method,
                    "scores": np.asarray(values, dtype=np.float64),
                    "defined": len(values),
                    "selected": selected,
                }
            )
    return rows


def _bootstrap_mean(scores: np.ndarray, seed: int) -> tuple[float, float, float]:
    mean = float(scores.mean())
    if len(scores) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(scores, size=(10_000, len(scores)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return mean, float(low), float(high)


def _seed(base: int, layer: int, method: str) -> int:
    digest = hashlib.sha256(f"{base}:{layer}:{method}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _shared_limits(scores: np.ndarray) -> tuple[float, float]:
    low = min(0.0, float(scores.min()))
    high = max(0.0, float(scores.max()))
    padding = max(0.05, 0.06 * (high - low))
    lower = max(-1.0, math.floor((low - padding) * 10) / 10)
    upper = min(1.0, math.ceil((high + padding) * 10) / 10)
    return lower, upper


def _load(path: Path) -> dict[str, Any]:
    run_path, results_path = path / "run.json", path / "results.json"
    if not run_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(f"evaluation output is incomplete: {path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if run.get("resolved", {}).get("format") != "icalens.autointerpretability-modern-evaluation":
        raise ValueError(f"not an autointerpretability evaluation output: {path}")
    if results.get("format") != "icalens.autointerpretability-modern-results":
        raise ValueError(f"unsupported autointerpretability results in {path}")
    return {"path": path, "run": run, "results": results}


def _merge_model_payloads(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine compatible layer runs into one panel per underlying language model."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        model = str(payload["run"]["resolved"]["preparation_resolved"]["model"]["repo_id"])
        grouped.setdefault(model, []).append(payload)

    merged: list[dict[str, Any]] = []
    for model, members in grouped.items():
        first = members[0]
        reference_evaluation = first["run"].get("evaluation", {})
        methods = list(first["run"]["resolved"]["methods"])
        conditions: list[dict[str, Any]] = []
        features: list[dict[str, Any]] = []
        occupied: set[tuple[int, str]] = set()
        layers: set[int] = set()
        paths: list[Path] = []
        for payload in members:
            evaluation = payload["run"].get("evaluation", {})
            if evaluation != reference_evaluation:
                raise ValueError(
                    f"cannot combine evaluator conditions for {model}; use separate commands"
                )
            if list(payload["run"]["resolved"]["methods"]) != methods:
                raise ValueError(f"cannot combine different method sets for {model}")
            paths.append(payload["path"])
            for condition in payload["results"].get("conditions", []):
                key = (int(condition["layer"]), str(condition["method"]))
                if key in occupied:
                    raise ValueError(
                        f"duplicate {model} layer {key[0]} {key[1]} results across inputs"
                    )
                occupied.add(key)
                layers.add(key[0])
                conditions.append(condition)
            features.extend(payload["results"].get("features", []))
        merged_run = {
            **first["run"],
            "resolved": {
                **first["run"]["resolved"],
                "layers": sorted(layers),
            },
        }
        merged.append(
            {
                "path": paths[0],
                "source_paths": paths,
                "run": merged_run,
                "results": {
                    **first["results"],
                    "conditions": conditions,
                    "features": features,
                },
            }
        )
    return merged


def _titles(value: str | None, payloads: Sequence[dict[str, Any]]) -> list[str]:
    if value is not None:
        titles = [item.strip() for item in value.split(",")]
        if len(titles) != len(payloads) or any(not title for title in titles):
            raise ValueError(
                f"--panel-titles must contain exactly {len(payloads)} comma-separated titles"
            )
        return titles
    titles = []
    for payload in payloads:
        model = payload["run"]["resolved"]["preparation_resolved"]["model"]["repo_id"]
        titles.append(MODEL_TITLES.get(str(model), str(model).split("/")[-1]))
    return titles


def _companion(
    payloads: Sequence[dict[str, Any]],
    titles: Sequence[str],
    panels: Sequence[list[dict[str, Any]]],
) -> str:
    single_layer = all(len({row["layer"] for row in panel}) == 1 for panel in panels)
    lines = [
        "Autointerpretability comparison. Scores are finite per-feature Pearson",
        "correlations over five held-out top-activating and five random fragments.",
        "Large markers show means; error bars are deterministic feature-level bootstrap",
        "95% intervals (10,000 resamples).",
        "",
        "panel\tlayer\tmethod\tdefined/selected\tmean\tbootstrap_95_low\tbootstrap_95_high",
    ]
    if single_layer:
        lines.insert(4, "Faint markers show individual features in this single-layer view.")
    for index, (payload, title, rows) in enumerate(zip(payloads, titles, panels, strict=True)):
        evaluation = payload["run"].get("evaluation", {})
        lines.append(
            f"# {title}: {evaluation.get('provider', 'unknown')} · "
            f"{evaluation.get('explainer_model', 'unknown')} → "
            f"{evaluation.get('simulator_model', 'unknown')}"
        )
        for row in rows:
            if row["scores"].size:
                mean, low, high = _bootstrap_mean(
                    row["scores"], _seed(index, row["layer"], row["method"])
                )
                summary = f"{mean:.6f}\t{low:.6f}\t{high:.6f}"
            else:
                summary = "nan\tnan\tnan"
            lines.append(
                f"{title}\t{row['layer']}\t{STYLES[row['method']][0]}\t"
                f"{row['defined']}/{row['selected']}\t{summary}"
            )
    return "\n".join(lines) + "\n"


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
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
