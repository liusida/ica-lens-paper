"""Create paper-ready held-out reconstruction figures."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="icalens experiment figure reconstruction", description=__doc__
    )
    parser.add_argument("experiment", nargs="+", help="Reconstruction result directories.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--format",
        default="png",
        help="Comma-separated output formats: png or pdf (default: png).",
    )
    parser.add_argument("--panel-titles", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiments = [Path(item).expanduser().resolve() for item in args.experiment]
    payloads = [_load(path) for path in experiments]
    formats = [item.strip() for item in args.format.split(",") if item.strip()]
    if not formats or any(item not in {"png", "pdf"} for item in formats):
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
    cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    style = {
        "ica": ("ICA top-k", "#3D5F99", "D"),
        "sae": ("SAE top-k", "#B45F4D", "o"),
        "sae_context_64": ("SAE top-k · positions <64", "#B45F4D", "o"),
        "pca": ("PCA top-k", "#5B8C6A", "^"),
        "random": ("Random top-k", "#777777", "s"),
    }
    outputs: list[Path] = []
    for metric, ylabel in (("nmse", "Normalized MSE"), ("cosine", "Cosine similarity")):
        overall = [
            (title, payload.get("rows", []))
            for payload, title in zip(payloads, titles, strict=True)
        ]
        outputs.extend(
            _render_grid(
                plt,
                overall,
                metric=metric,
                ylabel=ylabel,
                style=style,
                output=output,
                stem=f"reconstruction-{metric}",
                formats=formats,
                force=force,
                caption=(
                    "Held-out token-level top-k reconstruction, averaged equally across "
                    "selected layers and evaluation datasets."
                ),
                include_context_control=False,
            )
        )
        if len(payloads) == 1 and payloads[0].get("layer_payloads"):
            layer_panels, dataset_panels = _breakdown_panels(payloads[0])
            include_context_control = (
                payloads[0].get("experiment", {}).get("evaluation_context_length") is None
            )
            outputs.extend(
                _render_grid(
                    plt,
                    layer_panels,
                    metric=metric,
                    ylabel=ylabel,
                    style=style,
                    output=output,
                    stem=f"reconstruction-{metric}-by-layer",
                    formats=formats,
                    force=force,
                    caption="One subplot per layer; each curve is averaged across datasets.",
                    include_context_control=include_context_control,
                )
            )
            outputs.extend(
                _render_grid(
                    plt,
                    dataset_panels,
                    metric=metric,
                    ylabel=ylabel,
                    style=style,
                    output=output,
                    stem=f"reconstruction-{metric}-by-dataset",
                    formats=formats,
                    force=force,
                    caption="One subplot per dataset; each curve is averaged across layers.",
                    include_context_control=include_context_control,
                )
            )
    return outputs


def _render_grid(
    plt: Any,
    panels: Sequence[tuple[str, list[dict[str, Any]]]],
    *,
    metric: str,
    ylabel: str,
    style: dict[str, tuple[str, str, str]],
    output: Path,
    stem: str,
    formats: Sequence[str],
    force: bool,
    caption: str,
    include_context_control: bool,
) -> list[Path]:
    if not panels or any(not rows for _, rows in panels):
        raise ValueError("reconstruction experiment contains no completed rows")
    paths = [output / f"{stem}.{suffix}" for suffix in formats]
    caption_path = output / f"{stem}.txt"
    has_context_control = include_context_control and any(
        str(row.get("method", "")).startswith("sae_context_") for _, rows in panels for row in rows
    )
    for path in [*paths, caption_path]:
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force to replace it")
    columns = min(3, len(panels))
    row_count = math.ceil(len(panels) / columns)
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axes = plt.subplots(
            row_count,
            columns,
            figsize=(2.75 * columns, 2.45 * row_count),
            sharey=True,
            squeeze=False,
        )
        handles: dict[str, Any] = {}
        for axis, (title, rows) in zip(axes.flat, panels, strict=False):
            shared_budgets = _shared_budgets(rows)
            methods = ["random", "pca", "sae"]
            if include_context_control:
                methods.append("sae_context_64")
            methods.append("ica")
            for method in methods:
                points = _mean_curve(rows, method, metric, budgets=shared_budgets)
                if not points:
                    continue
                label, color, marker = style[method]
                endpoint = (
                    _native(rows, method, metric)
                    if method.startswith("sae")
                    else _full_linear_endpoint(rows, method, metric, shared_budgets)
                )
                handles[method] = axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    marker=marker,
                    linestyle="--" if method.startswith("sae_context_") else "-",
                    linewidth=1.5,
                    markersize=4,
                    label=label,
                )[0]
                if endpoint is not None:
                    segment = _linear_endpoint_segment(points, endpoint, method=method)
                    if segment is not None:
                        axis.plot(
                            [point[0] for point in segment],
                            [point[1] for point in segment],
                            color=color,
                            linewidth=1.5,
                        )
                    axis.scatter(*endpoint, marker="*", s=55, color=color, zorder=6)
            axis.set_xscale("log")
            ticks = _axis_ticks(rows, metric, shared_budgets)
            axis.set_xticks(ticks)
            axis.set_xticklabels(
                [str(value) for value in ticks], rotation=45, ha="right", fontsize=8
            )
            axis.set_title(title)
            axis.set_xlabel("mean active directions per token")
            axis.grid(axis="y", alpha=0.25)
        for axis in list(axes.flat)[len(panels) :]:
            axis.set_visible(False)
        for axis in axes[:, 0]:
            axis.set_ylabel(ylabel)
        ordered = [
            handles[name] for name in _legend_order(include_context_control) if name in handles
        ]
        figure.legend(
            ordered,
            [item.get_label() for item in ordered],
            loc="upper center",
            ncol=max(1, len(ordered)),
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        for path in paths:
            figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)
    control_note = (
        " For GPT-2, the dashed SAE curve is a training-context control evaluated only "
        "at token positions 0–63; solid curves use the full 1,024-position context."
        if has_context_control
        else ""
    )
    budget_sets = [set(_shared_budgets(rows)) for _, rows in panels]
    shared_caption_budgets = sorted(set.intersection(*budget_sets))
    budget_text = ", ".join(str(value) for value in shared_caption_budgets)
    caption_path.write_text(
        f"{caption} The plotted metric is {ylabel}. Linear methods use the requested "
        f"top-k budgets {budget_text}. Linear curves extend from the largest requested "
        "budget to their measured complete-basis endpoints. SAE points are placed at their "
        "measured mean number of nonzero features; repeated budgets above native "
        "SAE activity collapse to one point. Stars mark complete-basis reconstruction "
        "for ICA, PCA, and Random, and native sparse reconstruction for SAE."
        f"{control_note}\n",
        encoding="utf-8",
    )
    return [*paths, caption_path]


def _legend_order(include_context_control: bool) -> tuple[str, ...]:
    """Order legend entries independently from the back-to-front drawing order."""
    if include_context_control:
        return ("ica", "sae", "sae_context_64", "pca", "random")
    return ("ica", "sae", "pca", "random")


def _breakdown_panels(
    payload: dict[str, Any],
) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[tuple[str, list[dict[str, Any]]]]]:
    from .reconstruction import _aggregate_layer

    layer_payloads = payload["layer_payloads"]
    layers = [(f"Layer {item['layer']}", item["rows"]) for item in layer_payloads]
    dataset_count = len(layer_payloads[0]["datasets"])
    datasets: list[tuple[str, list[dict[str, Any]]]] = []
    for index in range(dataset_count):
        rows: list[dict[str, Any]] = []
        config = layer_payloads[0]["datasets"][index]["dataset"]
        for item in layer_payloads:
            rows.extend(_aggregate_layer(int(item["layer"]), [item["datasets"][index]]))
        datasets.append((_dataset_title(config), rows))
    return layers, datasets


def _dataset_title(config: dict[str, Any]) -> str:
    """Return an unambiguous, compact title for a reconstruction dataset."""
    if config.get("repo_id") == "wikimedia/wikipedia" and config.get("domain"):
        return str(config["domain"])
    return str(config["repo_id"])


def _shared_budgets(rows: list[dict[str, Any]]) -> list[int]:
    """Return numeric top-k budgets evaluated for every primary method."""
    method_keys: list[set[int]] = []
    for method in ("ica", "sae", "pca", "random"):
        keys = {
            int(str(row["k"]))
            for row in rows
            if row.get("method") == method and str(row.get("k", "")).isdigit()
        }
        if keys:
            method_keys.append(keys)
    if not method_keys:
        return []
    return sorted(set.intersection(*method_keys))


def _mean_curve(
    rows: list[dict[str, Any]],
    method: str,
    metric: str,
    *,
    budgets: Sequence[int] | None = None,
) -> list[tuple[float, float]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    allowed = set(budgets) if budgets is not None else None
    for row in rows:
        if row.get("method") != method or row.get("k") == "native":
            continue
        requested = int(str(row["k"]))
        if allowed is not None and requested not in allowed:
            continue
        grouped.setdefault(str(row["k"]), []).append(
            (float(row["effective_k"]), float(row[f"{metric}_mean"]))
        )
    requested_points = [
        (
            sum(x for x, _ in values) / len(values),
            sum(y for _, y in values) / len(values),
        )
        for key, values in sorted(grouped.items(), key=lambda item: int(item[0]))
    ]
    points: list[tuple[float, float]] = []
    for x, y in requested_points:
        if points and math.isclose(x, points[-1][0], rel_tol=0.03, abs_tol=1e-5):
            # Sparse encoders cannot retain more than their nonzero native codes.
            # Budgets beyond that point reconstruct the same vector and should not
            # appear as fictitious 100- or 300-direction measurements.
            points[-1] = ((points[-1][0] + x) / 2, (points[-1][1] + y) / 2)
        else:
            points.append((x, y))
    return points


def _linear_endpoint_segment(
    points: list[tuple[float, float]],
    endpoint: tuple[float, float] | None,
    *,
    method: str,
) -> list[tuple[float, float]] | None:
    """Connect to full basis without applying the regular curve marker there."""
    if endpoint is None or method.startswith("sae") or _same_plot_point(points[-1], endpoint):
        return None
    return [points[-1], endpoint]


def _axis_ticks(rows: list[dict[str, Any]], metric: str, budgets: Sequence[int]) -> list[int]:
    """Label shared budgets and the complete linear width."""
    ticks = set(int(value) for value in budgets)
    full = _full_linear_endpoint(rows, "ica", metric, budgets)
    if full is not None:
        ticks.add(round(full[0]))
    return sorted(ticks)


def _native(rows: list[dict[str, Any]], method: str, metric: str) -> tuple[float, float] | None:
    selected = [row for row in rows if row.get("method") == method and row.get("k") == "native"]
    if not selected:
        return None
    return (
        sum(float(row["effective_k"]) for row in selected) / len(selected),
        sum(float(row[f"{metric}_mean"]) for row in selected) / len(selected),
    )


def _full_linear_endpoint(
    rows: list[dict[str, Any]],
    method: str,
    metric: str,
    budgets: Sequence[int],
) -> tuple[float, float] | None:
    """Return a linear method's evaluated complete-basis endpoint."""
    requested = set(int(value) for value in budgets)
    selected = [
        row
        for row in rows
        if row.get("method") == method
        and str(row.get("k", "")).isdigit()
        and int(str(row["k"])) not in requested
    ]
    if not selected:
        return None
    return (
        sum(float(row["effective_k"]) for row in selected) / len(selected),
        sum(float(row[f"{metric}_mean"]) for row in selected) / len(selected),
    )


def _same_plot_point(point: tuple[float, float], endpoint: tuple[float, float]) -> bool:
    """Return whether two markers would be visually indistinguishable on the plot."""
    return math.isclose(point[0], endpoint[0], rel_tol=0.03, abs_tol=1e-6) and math.isclose(
        point[1], endpoint[1], rel_tol=0.005, abs_tol=0.005
    )


def _load(path: Path) -> dict[str, Any]:
    result = path / "results.json"
    if not result.is_file():
        candidates = list(Path.cwd().glob("results/**/results.json"))
        ranked = sorted(
            candidates,
            key=lambda item: difflib.SequenceMatcher(
                None, str(path), str(item.parent.resolve())
            ).ratio(),
            reverse=True,
        )[:3]
        suggestion = ""
        if ranked:
            choices = "\n".join(f"  {item.parent}" for item in ranked)
            suggestion = f"\nAvailable result directories include:\n{choices}"
        raise FileNotFoundError(
            f"no reconstruction results at {result}. The input must match the "
            f"--output directory used by the experiment.{suggestion}"
        )
    payload = json.loads(result.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid reconstruction results in {result}: expected an object")
    payload["layer_payloads"] = [
        json.loads(layer.read_text(encoding="utf-8"))
        for layer in sorted((path / "layers").glob("layer_*.json"))
    ]
    return payload


def _titles(value: str | None, payloads: Sequence[dict[str, Any]]) -> list[str]:
    if value is not None:
        titles = [item.strip() for item in value.split(",")]
        if len(titles) != len(payloads):
            raise ValueError("--panel-titles must contain one title per experiment")
        return titles
    return [str(payload["experiment"]["model_id"]).split("/")[-1] for payload in payloads]
