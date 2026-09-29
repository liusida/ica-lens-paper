#!/usr/bin/env python3
"""Construct directions A-D and plot their Layer-0 projection distributions."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from safetensors.torch import load_file
from scipy.stats import jarque_bera, kurtosis, norm, skew

from icalens import ICALens

COLORS = {"background": "#B8C1CC", "target": "#B45F4D", "concept": "#3D5F99",
          "random": "#777777"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=root / "work/selected")
    parser.add_argument("--output", type=Path, default=root / "work/render")
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--b-selection", choices=("isolated", "random", "concept"),
                        default="isolated")
    parser.add_argument("--b-concept-rank", type=int, default=1)
    parser.add_argument("--ica-lens", type=Path)
    parser.add_argument("--ica-component", type=int, default=65)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    figure_output = (
        args.figure_output.expanduser().resolve() if args.figure_output is not None else output
    )
    names = ["direction-a", "direction-b", "direction-c", "direction-d", "directions-row"]
    if args.ica_lens is not None:
        names.extend(("direction-e", "directions-row-raw"))
    paths = [figure_output / f"{name}.png" for name in names]
    results_path = output / "results.json"
    if (results_path.exists() or any(path.exists() for path in paths)) and not args.force:
        raise FileExistsError("outputs exist; pass --force to replace them")
    capture = args.capture.expanduser().resolve()
    samples = json.loads((capture / "samples.json").read_text(encoding="utf-8"))
    x = load_file(capture / "activations.safetensors")[f"layer_{args.layer:02d}"].double()
    background = torch.tensor([sample["role"] == "background" for sample in samples])
    concept = ~background
    center = x[background].mean(dim=0)
    z = x - center
    normalized = z / torch.linalg.vector_norm(z, dim=1, keepdim=True)

    background_indices = torch.nonzero(background, as_tuple=False).flatten()
    if args.b_selection == "concept":
        target_index, target_nearest_similarity, target_nearest_index = _pick_concept_target(
            normalized, concept, rank=args.b_concept_rank
        )
    elif args.b_selection == "random":
        target_index, target_nearest_similarity, target_nearest_index = _pick_random_target(
            normalized, background_indices, seed=args.seed
        )
    else:
        target_index, target_nearest_similarity, target_nearest_index = _find_isolated(
            normalized, background_indices, seed=args.seed
        )

    generator = torch.Generator().manual_seed(args.seed)
    random_order = torch.randperm(len(background_indices), generator=generator)
    group_size = int(concept.sum())
    random_indices = background_indices[random_order[:group_size]]
    direction_a = _unit(torch.randn(x.shape[1], generator=generator, dtype=x.dtype))
    plane_direction = torch.randn(x.shape[1], generator=generator, dtype=x.dtype)
    plane_direction = _unit(plane_direction - torch.dot(plane_direction, direction_a) * direction_a)
    plane_v_coordinates = torch.empty(0, dtype=x.dtype)
    direction_b = _unit(z[target_index])
    direction_c = _unit(z[concept].mean(dim=0))
    direction_d = _unit(z[random_indices].mean(dim=0))
    directions = {"a": direction_a, "b": direction_b, "c": direction_c, "d": direction_d}
    evaluation_masks = {
        "a": torch.ones(len(samples), dtype=torch.bool),
        "b": torch.ones(len(samples), dtype=torch.bool),
        "c": torch.ones(len(samples), dtype=torch.bool),
        "d": torch.ones(len(samples), dtype=torch.bool),
    }
    raw_projections = {name: z @ direction for name, direction in directions.items()}
    if args.ica_lens is not None:
        lens = ICALens.from_pretrained(args.ica_lens.expanduser().resolve())
        artifact = lens._get_layer(args.layer)  # Research diagnostic: use the reading vector.
        reading = torch.from_numpy(artifact.reading_matrix[args.ica_component]).double()
        reading = _unit(reading)
        component_scores = z @ reading
        if component_scores[concept].mean() < component_scores[background].mean():
            reading = -reading
            component_scores = -component_scores
        plane_direction = _unit(reading - torch.dot(reading, direction_a) * direction_a)
        plane_v_coordinates = torch.stack(
            (torch.dot(reading, direction_a), torch.dot(reading, plane_direction))
        )
        raw_projections["c1"] = component_scores
        evaluation_masks["c1"] = torch.ones(len(samples), dtype=torch.bool)
    projections = {
        name: _standardize(values, evaluation_masks[name])
        for name, values in raw_projections.items()
    }
    statistics = {
        name: _statistics(projections[name][evaluation_masks[name]].numpy())
        for name in projections
    }
    concept_rows = normalized[concept]
    random_rows = normalized[random_indices]
    coherence = {
        "concept_mean_pairwise_cosine": _mean_pairwise_cosine(concept_rows),
        "random_mean_pairwise_cosine": _mean_pairwise_cosine(random_rows),
        "concept_split_half_angle_degrees": _angle(
            concept_rows[:group_size // 2].mean(dim=0),
            concept_rows[group_size // 2:].mean(dim=0),
        ),
        "random_split_half_angle_degrees": _angle(
            random_rows[:group_size // 2].mean(dim=0),
            random_rows[group_size // 2:].mean(dim=0),
        ),
    }
    results = {
        "layer": args.layer,
        "directions": {
            "A": "seeded dense random unit direction",
            "B": {
                "concept": "centered activation direction of one related token",
                "random": "centered activation direction of one random token",
                "isolated": "centered activation direction of one isolated token",
            }[args.b_selection],
            "C": f"unit mean of {group_size} centered related-token activations",
            **({"D": "fitted ICA direction scores",
                "E": f"unit mean of {group_size} centered random-token activations"}
               if args.ica_lens is not None else
               {"D": f"unit mean of {group_size} centered random-token activations"}),
        },
        "statistics": {
            ({"c1": "D", "d": "E"}.get(name, name.upper())
             if args.ica_lens is not None else name.upper()): value
            for name, value in statistics.items()
        },
        "direction_b": {
            "target": samples[target_index],
            "nearest_token": samples[target_nearest_index],
            "nearest_cosine_similarity": target_nearest_similarity,
        },
        "direction_a_top_tokens": [
            {
                **samples[index],
                "projection": float(raw_projections["a"][index]),
            }
            for index in torch.argsort(raw_projections["a"], descending=True)[:10].tolist()
        ],
        "direction_ica_top_tokens": [
            {
                **samples[index],
                "projection": float(raw_projections["c1"][index]),
            }
            for index in torch.argsort(
                raw_projections["c1"], descending=True
            )[:10].tolist()
        ] if "c1" in raw_projections else [],
        "direction_c_tokens": [samples[i] for i in torch.nonzero(concept).flatten().tolist()],
        "direction_e_tokens": [samples[i] for i in random_indices.tolist()],
        "coherence": coherence,
        "note": "Every direction is evaluated over the full token set.",
    }
    output.mkdir(parents=True, exist_ok=True)
    figure_output.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(
        output / "overview-data.npz",
        projection_a=raw_projections["a"].numpy(),
        projection_plane=z.numpy() @ plane_direction.numpy(),
        plane_v_coordinates=plane_v_coordinates.numpy(),
        projection_b=raw_projections["b"].numpy(),
        projection_c=raw_projections["c"].numpy(),
        projection_ica=raw_projections.get("c1", torch.empty(0)).numpy(),
        projection_e=raw_projections["d"].numpy(),
        background=background.numpy(),
        concept=concept.numpy(),
        tokens=np.asarray([sample["token"] for sample in samples]),
        random_indices=random_indices.numpy(),
        target_index=np.asarray(target_index),
    )
    _plot_all(projections, evaluation_masks, background, concept, target_index,
              random_indices, statistics, figure_output, b_selection=args.b_selection,
              ica_component=args.ica_component if args.ica_lens is not None else None)
    if "c1" in raw_projections:
        _plot_raw_row(raw_projections, evaluation_masks, background, concept, target_index,
                      random_indices, statistics, figure_output, args.ica_component,
                      args.b_selection)
    print(results_path)
    for path in paths:
        print(path)


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector)


def _standardize(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (values - values[reference].mean()) / values[reference].std(correction=1)


def _statistics(values: np.ndarray) -> dict[str, float]:
    jb = jarque_bera(values)
    return {"excess_kurtosis": float(kurtosis(values)), "skewness": float(skew(values)),
            "jarque_bera": float(jb.statistic), "jarque_bera_p": float(jb.pvalue)}


def _mean_pairwise_cosine(rows: torch.Tensor) -> float:
    matrix = rows @ rows.T
    return float(matrix[torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)].mean())


def _find_isolated(
    normalized: torch.Tensor, background_indices: torch.Tensor, *, seed: int,
) -> tuple[int, float, int]:
    if len(background_indices) <= 5000:
        candidates = background_indices
    else:
        generator = torch.Generator().manual_seed(seed + 1)
        order = torch.randperm(len(background_indices), generator=generator)
        candidates = background_indices[order[:1024]]
    best_target = -1
    best_nearest = -1
    best_similarity = torch.inf
    for chunk in candidates.split(32):
        cosine = normalized[chunk] @ normalized.T
        cosine[torch.arange(len(chunk)), chunk] = -torch.inf
        similarities, indices = cosine.max(dim=1)
        position = int(similarities.argmin())
        if similarities[position] < best_similarity:
            best_similarity = similarities[position]
            best_target = int(chunk[position])
            best_nearest = int(indices[position])
    return best_target, float(best_similarity), best_nearest


def _pick_random_target(
    normalized: torch.Tensor, background_indices: torch.Tensor, *, seed: int,
) -> tuple[int, float, int]:
    generator = torch.Generator().manual_seed(seed + 1)
    target = int(background_indices[torch.randint(len(background_indices), (1,),
                                                  generator=generator)])
    cosine = normalized[target] @ normalized.T
    cosine[target] = -torch.inf
    nearest = int(cosine.argmax())
    return target, float(cosine[nearest]), nearest


def _pick_concept_target(
    normalized: torch.Tensor, concept: torch.Tensor, *, rank: int,
) -> tuple[int, float, int]:
    indices = torch.nonzero(concept, as_tuple=False).flatten()
    if not 1 <= rank <= len(indices):
        raise ValueError(f"--b-concept-rank must be between 1 and {len(indices)}")
    target = int(indices[rank - 1])
    cosine = normalized[target] @ normalized.T
    cosine[target] = -torch.inf
    nearest = int(cosine.argmax())
    return target, float(cosine[nearest]), nearest


def _angle(first: torch.Tensor, second: torch.Tensor) -> float:
    cosine = torch.dot(_unit(first), _unit(second)).clamp(-1, 1)
    return float(torch.rad2deg(torch.acos(cosine)))


def _plot_all(
    projections: dict[str, torch.Tensor], evaluation_masks: dict[str, torch.Tensor],
    background: torch.Tensor, concept: torch.Tensor, target: int,
    random_indices: torch.Tensor, statistics: dict[str, dict[str, float]], output: Path,
    b_selection: str, ica_component: int | None,
) -> None:
    maximum = max(
        float(projections[name][evaluation_masks[name]].abs().max()) for name in projections
    )
    limit = max(4.0, float(np.ceil(maximum)))
    bins = np.linspace(-limit, limit, 31)
    grid = np.linspace(-limit, limit, 500)
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
             "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
             "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        order = ("a", "b", "c", "c1", "d") if "c1" in projections else ("a", "b", "c", "d")
        fig, axes = plt.subplots(1, len(order), figsize=(2.55 * len(order), 2.6),
                                 sharex=True, sharey=True)
        b_title = {"random": "B · Random token",
                   "concept": "B · Direction of one related token",
                   "isolated": "B · Isolated token"}[b_selection]
        titles = {"a": "A · Random direction", "b": b_title,
                  "c": "C · Mean direction of related tokens",
                  "d": "D · Mean direction of random tokens"}
        if ica_component is not None:
            titles["c1"] = "D · ICA-discovered direction"
            titles["d"] = "E · Mean direction of random tokens"
        for ax, name in zip(axes, order, strict=True):
            values = projections[name].numpy()
            bg = (background & evaluation_masks[name]).numpy()
            ax.hist(values[bg], bins=bins, density=True, color=COLORS["background"], alpha=0.8)
            ax.plot(grid, norm.pdf(grid), color="0.45", linestyle="--", linewidth=1.2)
            if name == "b":
                ax.scatter(values[concept.numpy()], np.full(int(concept.sum()), 0.008),
                           color=COLORS["concept"], marker="|", s=55, linewidths=1.2)
                ax.scatter(values[target], 0.008, color=COLORS["target"], s=24, zorder=4)
            if name in {"c", "c1"}:
                ax.scatter(values[concept.numpy()], np.full(int(concept.sum()), 0.008),
                           color=COLORS["concept"], marker="|", s=55, linewidths=1.2)
            if name == "d":
                ax.scatter(values[random_indices], np.full(len(random_indices), 0.008),
                           color=COLORS["random"],
                           marker="|", s=55, linewidths=1.2)
            ax.set_title(titles[name], loc="left", fontweight="bold")
            sigma = values[evaluation_masks[name].numpy()].std(ddof=1)
            ax.text(0.97, 0.95,
                    f"Excess kurtosis = {statistics[name]['excess_kurtosis']:.2f}\nσ = {sigma:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.88", linewidth=0.5)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Density of background tokens")
        fig.supxlabel("Projection (standardized)", y=0.04, fontsize=8)
        fig.subplots_adjust(left=0.06, right=0.995, bottom=0.22, top=0.86, wspace=0.12)
        _save(fig, output / "directions-row")
        for index, name in enumerate(order):
            single, ax = plt.subplots(figsize=(3.35, 2.5))
            source = axes[index]
            values = projections[name].numpy()
            bg = (background & evaluation_masks[name]).numpy()
            ax.hist(values[bg], bins=bins, density=True, color=COLORS["background"], alpha=0.8)
            ax.plot(grid, norm.pdf(grid), color="0.45", linestyle="--", linewidth=1.2)
            if name == "b":
                ax.scatter(values[concept.numpy()], np.full(int(concept.sum()), 0.008),
                           color=COLORS["concept"], marker="|", s=55, linewidths=1.2)
                ax.scatter(values[target], 0.008, color=COLORS["target"], s=24, zorder=4)
            if name in {"c", "c1"}:
                ax.scatter(values[concept.numpy()], np.full(int(concept.sum()), 0.008),
                           color=COLORS["concept"], marker="|", s=55, linewidths=1.2)
            if name == "d":
                ax.scatter(values[random_indices], np.full(len(random_indices), 0.008),
                           color=COLORS["random"],
                           marker="|", s=55, linewidths=1.2)
            ax.set_title(titles[name], loc="left", fontweight="bold")
            ax.set_xlabel("Projection (standardized)")
            ax.set_ylabel("Density")
            ax.text(0.97, 0.95, source.texts[0].get_text(), transform=ax.transAxes,
                    ha="right", va="top", fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.88", linewidth=0.5)
            single.subplots_adjust(left=0.17, right=0.98, bottom=0.2, top=0.88)
            display_name = {"c1": "d", "d": "e"}.get(name, name) if "c1" in order else name
            _save(single, output / f"direction-{display_name}")


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def _plot_raw_row(
    projections: dict[str, torch.Tensor], evaluation_masks: dict[str, torch.Tensor],
    background: torch.Tensor, concept: torch.Tensor, target: int,
    random_indices: torch.Tensor, statistics: dict[str, dict[str, float]], output: Path,
    ica_component: int, b_selection: str,
) -> None:
    order = ("a", "b", "c", "c1", "d")
    values_by_name = {name: projections[name].numpy() for name in order}
    maximum = max(float(np.abs(values).max()) for values in values_by_name.values())
    limit = float(np.ceil(maximum))
    bins = np.linspace(-limit, limit, 31)
    grid = np.linspace(-limit, limit, 500)
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "Times",
             "DejaVu Serif"], "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7,
             "pdf.fonttype": 42, "ps.fonttype": 42}
    b_title = {"random": "B · Random token",
               "concept": "B · Direction of one related token",
               "isolated": "B · Isolated token"}[b_selection]
    titles = {"a": "A · Random direction", "b": b_title,
              "c": "C · Mean direction of related tokens",
              "c1": "D · ICA-discovered direction",
              "d": "E · Mean direction of random tokens"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 5, figsize=(12.75, 2.8), sharex=True, sharey=True)
        for ax, name in zip(axes, order, strict=True):
            values = values_by_name[name]
            bg = (background & evaluation_masks[name]).numpy()
            fit_values = values[evaluation_masks[name].numpy()]
            ax.hist(values[bg], bins=bins, density=True,
                    color=COLORS["background"], alpha=0.8)
            ax.plot(grid, norm.pdf(grid, loc=fit_values.mean(), scale=fit_values.std(ddof=1)),
                    color="0.45", linestyle="--", linewidth=1.2)
            if name == "b":
                ax.scatter(values[concept.numpy()], np.full(int(concept.sum()), 0.008),
                           color=COLORS["concept"], marker="|", s=55, linewidths=1.2)
                ax.scatter(values[target], 0.008, color=COLORS["target"], s=24, zorder=4)
            if name in {"c", "c1"}:
                ax.scatter(values[concept.numpy()], np.full(int(concept.sum()), 0.008),
                           color=COLORS["concept"], marker="|", s=55, linewidths=1.2)
            if name == "d":
                ax.scatter(values[random_indices], np.full(len(random_indices), 0.008),
                           color=COLORS["random"], marker="|", s=55, linewidths=1.2)
            ax.set_title(titles[name], loc="left", fontweight="bold")
            sigma = fit_values.std(ddof=1)
            ax.text(0.97, 0.95,
                    f"Excess kurtosis = {statistics[name]['excess_kurtosis']:.2f}\nσ = {sigma:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.88", linewidth=0.5)
            ax.set_axisbelow(True)
        legend_handles = [
            Line2D([], [], color=COLORS["target"], marker="o", linestyle="None",
                   markersize=4.5, label="Selected token in B"),
            Patch(facecolor=COLORS["concept"], edgecolor="none",
                  label="Related tokens in B, C, D"),
            Patch(facecolor=COLORS["random"], edgecolor="none", label="Random tokens in E"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.99), handlelength=1.2, columnspacing=1.5)
        axes[0].set_ylabel("Density of background tokens")
        fig.supxlabel("Raw projection", y=0.04, fontsize=8)
        fig.subplots_adjust(left=0.05, right=0.995, bottom=0.2, top=0.76, wspace=0.12)
        _save(fig, output / "directions-row-raw")


if __name__ == "__main__":
    main()
