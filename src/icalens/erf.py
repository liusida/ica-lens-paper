"""Effective-receptive-field measurements for an :class:`ICALens`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .lens import ICALens


class ERFAnalysis:
    """Effective-receptive-field methods bound to one ICA Lens."""

    def __init__(self, lens: ICALens) -> None:
        self._lens = lens
        self._suffix_datasets: dict[str, Any] = {}
        self._suffix_token_cache: dict[tuple[str, int, int], list[int]] = {}

    def suffix_sweep(
        self,
        *,
        layer: int,
        component: int,
        rank_thresholds: Sequence[int] = (1, 3, 5, 10, 15),
        occurrences: int = 20,
        exact_suffix_length: int = 10,
        max_batch_size: int = 64,
        batch_token_budget: int = 64,
        device: str | torch.device | None = "auto",
        model: torch.nn.Module | None = None,
        tokenizer: Any = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Measure suffix-sweep ERF for one stored component profile.

        The language model and tokenizer are loaded lazily and reused by later
        Lens analyses. The returned dictionary contains component summaries for
        every requested rank threshold and all occurrence-level sweep records.
        """
        from .analysis import _resolve_device, _resolve_model_and_tokenizer
        from .experiments.erf_suffix_sweep import (
            METHOD,
            SCHEMA_VERSION,
            _measure_layer,
            _validate_full_context_ranks,
        )

        layer = _positive_or_zero_integer("layer", layer)
        component = _positive_or_zero_integer("component", component)
        occurrences = _positive_integer("occurrences", occurrences)
        exact_suffix_length = _positive_integer("exact_suffix_length", exact_suffix_length)
        max_batch_size = _positive_integer("max_batch_size", max_batch_size)
        batch_token_budget = _positive_integer("batch_token_budget", batch_token_budget)
        thresholds = _rank_thresholds(rank_thresholds)

        lens = self._lens
        profile = lens.component_profile(layer=layer, component=component)
        direction = profile.get("tail_direction")
        if direction not in ("positive", "negative"):
            raise ValueError(f"layer {layer} component {component} has no selected tail direction")
        selected_occurrences = profile["examples"][direction]["occurrences"][:occurrences]
        if not selected_occurrences:
            raise ValueError(
                f"layer {layer} component {component} has no stored {direction}-tail occurrences"
            )
        prepared = {
            str(layer): {
                str(component): {
                    "component": component,
                    "tail_direction": direction,
                    "occurrences": selected_occurrences,
                }
            }
        }
        _validate_full_context_ranks({"lens": prepared})

        loaded_model, loaded_tokenizer = _resolve_model_and_tokenizer(
            lens,
            model,
            tokenizer,
            device,
            verbose=verbose,
        )
        automatic_device = device is None or str(device) == "auto"
        target_device = lens._analysis_device if model is None else None
        if model is not None and automatic_device:
            try:
                target_device = str(next(loaded_model.parameters()).device)
            except StopIteration as error:
                raise ValueError(
                    "model has no parameters from which to infer its device"
                ) from error
        if target_device is None:
            target_device = _resolve_device(device)
        results: dict[int, dict[str, Any]] = {}

        def checkpoint(index: int, result: dict[str, Any]) -> None:
            results[index] = result

        _measure_layer(
            lens=lens,
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            datasets=self._suffix_datasets,
            token_cache=self._suffix_token_cache,
            layer=layer,
            prepared_components={component: prepared[str(layer)][str(component)]},
            exact_suffix_length=exact_suffix_length,
            rank_thresholds=thresholds,
            max_batch_size=max_batch_size,
            batch_token_budget=batch_token_budget,
            device=target_device,
            checkpoint=checkpoint,
        )
        result = results.get(component)
        if result is None:
            raise RuntimeError("suffix-sweep ERF did not produce a component result")
        result.update(
            method=METHOD,
            schema_version=SCHEMA_VERSION,
            model_id=lens.model_id,
            model_revision=lens.model_revision,
        )
        return result


def _positive_or_zero_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rank_thresholds(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("rank_thresholds must be a sequence of positive integers")
    thresholds = tuple(values)
    if not thresholds:
        raise ValueError("rank_thresholds must not be empty")
    normalized = tuple(_positive_integer("rank threshold", value) for value in thresholds)
    if len(set(normalized)) != len(normalized):
        raise ValueError("rank_thresholds must not contain duplicates")
    return tuple(sorted(normalized))
