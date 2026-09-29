"""Optional Transformers integration for capturing and analyzing model activations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import torch

from ._capture import capture_resid_post, clamp_resid_post


class ComponentProfile(dict[str, Any]):
    """Stored metadata for one ICA component, with notebook rendering."""

    def __init__(self, values: Mapping[str, Any], *, layer: int) -> None:
        super().__init__(values)
        self.layer = int(layer)

    @property
    def component(self) -> int:
        """Component index represented by this profile."""
        return int(self["component"])

    def to_html(self, output_file: str | Path) -> Path:
        """Write the component profile panel to a standalone HTML file."""
        from .html import write_component_profile_html

        return write_component_profile_html(self, output_file, layer=self.layer)

    def _repr_html_(self) -> str:
        """Render the component profile panel in Jupyter and Colab."""
        from .html import component_profile_iframe

        return component_profile_iframe(self, layer=self.layer)

    def display(self, *, height: int = 520) -> None:
        """Display the component profile panel in a notebook."""
        try:
            from IPython.display import HTML, display
        except ImportError as error:
            raise ImportError(
                "ComponentProfile.display() requires IPython; use to_html() outside a notebook"
            ) from error
        from .html import component_profile_iframe

        display(  # type: ignore[no-untyped-call]
            HTML(component_profile_iframe(self, layer=self.layer, height=height))  # type: ignore[no-untyped-call]
        )


@dataclass(frozen=True)
class CaptureResult:
    """Tokens and aligned activations captured from one model input."""

    tokens: tuple[str, ...]
    token_texts: tuple[str, ...]
    token_labels: tuple[str, ...]
    token_tooltips: tuple[str, ...]
    token_groups: tuple[str, ...]
    token_ids: torch.Tensor
    positions: torch.Tensor
    activations: torch.Tensor


@dataclass(frozen=True)
class AnalysisResult(CaptureResult):
    """A capture together with signed ICA scores and component energy shares."""

    scores: torch.Tensor
    energy: torch.Tensor
    model: str
    layer: int
    input_text: str
    token_scope: str
    messages: tuple[dict[str, str], ...]
    selected_components: tuple[int, ...] = ()
    component_profiles: dict[int, dict[str, Any]] | None = None
    logit_effects: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_html(
        self,
        output_file: str | Path,
        *,
        metric: Literal["score", "energy"] = "score",
        top_k: int = 3,
        title: str = "ICA Lens Explorer",
    ) -> Path:
        """Write a self-contained interactive report and return its path."""
        from .html import write_analysis_html

        return write_analysis_html(self, output_file, metric=metric, top_k=top_k, title=title)

    def _repr_html_(self) -> str:
        """Render the default interactive explorer in Jupyter and Colab."""
        from .html import analysis_iframe

        return analysis_iframe(self)

    def display(
        self,
        *,
        metric: Literal["score", "energy"] = "score",
        top_k: int = 3,
        title: str = "ICA Lens Explorer",
        height: int = 720,
    ) -> None:
        """Display a configurable interactive explorer in a notebook."""
        try:
            from IPython.display import HTML, display
        except ImportError as error:
            raise ImportError(
                "AnalysisResult.display() requires IPython; use to_html() outside a notebook"
            ) from error
        from .html import analysis_iframe

        display(  # type: ignore[no-untyped-call]
            HTML(  # type: ignore[no-untyped-call]
                analysis_iframe(
                    self,
                    metric=metric,
                    top_k=top_k,
                    title=title,
                    height=height,
                )
            )
        )


def capture(
    lens: Any,
    inputs: str | list[dict[str, str]],
    *,
    layer: int,
    model: torch.nn.Module | None = None,
    tokenizer: Any = None,
    token_scope: Literal["assistant", "user", "content", "all"] = "all",
    document_framing: Literal["model", "none", "recorded"] = "recorded",
    context_length: int | None = None,
    device: str | torch.device | None = "auto",
    verbose: bool = False,
) -> CaptureResult:
    """Capture activations for raw text or a completed chat conversation."""
    if document_framing not in ("model", "none", "recorded"):
        raise ValueError("document_framing must be 'model', 'none', or 'recorded'")
    started = perf_counter()
    _analysis_log(verbose, f"Preparing {lens.model_id}...")
    model, tokenizer = _resolve_model_and_tokenizer(lens, model, tokenizer, device, verbose=verbose)
    if isinstance(inputs, str):
        framing = (
            _document_framing_for_layer(lens, layer)
            if document_framing == "recorded"
            else {"strategy": "none", "token": None, "token_id": None}
        )
        prefix_id = framing.get("token_id") if framing.get("strategy") != "none" else None
        content_limit = context_length
        if content_limit is not None and prefix_id is not None:
            content_limit -= 1
            if content_limit <= 0:
                raise ValueError("context_length must leave room for document framing")
        encoded = tokenizer(
            inputs,
            add_special_tokens=document_framing == "model",
            truncation=content_limit is not None,
            max_length=content_limit,
            return_special_tokens_mask=document_framing == "model",
            return_tensors="pt",
        )
        if prefix_id is not None:
            prefix = torch.tensor([[int(prefix_id)]], dtype=encoded["input_ids"].dtype)
            encoded["input_ids"] = torch.cat((prefix, encoded["input_ids"]), dim=1)
            if "attention_mask" in encoded:
                prefix_mask = torch.ones((1, 1), dtype=encoded["attention_mask"].dtype)
                encoded["attention_mask"] = torch.cat(
                    (prefix_mask, encoded["attention_mask"]), dim=1
                )
            _analysis_log(
                verbose,
                f"Applied recorded document framing {framing.get('token')!r}; "
                "the prefix is excluded from reported tokens.",
            )
        if document_framing == "model" and "special_tokens_mask" in encoded:
            positions = torch.nonzero(
                encoded.pop("special_tokens_mask")[0] == 0, as_tuple=False
            ).flatten()
        else:
            offset = 1 if prefix_id is not None else 0
            positions = torch.arange(offset, encoded["input_ids"].shape[1], dtype=torch.long)
        token_groups: tuple[str, ...] = ()
    else:
        encoded, positions, token_groups = _encode_chat(
            tokenizer, inputs, token_scope=token_scope, context_length=context_length
        )
    model_device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded.get("attention_mask")
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(model_device)
    selected = positions.to(model_device)
    _analysis_log(
        verbose,
        f"Capturing {len(selected)} token activations at {lens.activation_site} layer {layer}...",
    )
    capture_started = perf_counter()
    if lens.activation_site == "resid_post":
        activations = capture_resid_post(
            model,
            model_inputs=kwargs,
            layers=(layer,),
            positions=selected,
        )[layer].float()
    else:
        with torch.inference_mode():
            outputs = model(**kwargs, output_hidden_states=True, use_cache=False)
        if outputs.hidden_states is None:
            raise RuntimeError("model did not return hidden states")
        hidden_index = (
            layer + 1
            if lens.layer_indexing == "hidden_states_without_initial_embedding_state"
            else layer
        )
        if hidden_index < 0 or hidden_index >= len(outputs.hidden_states):
            raise ValueError(f"layer {layer} is unavailable in model hidden states")
        activations = outputs.hidden_states[hidden_index][0].index_select(0, selected).float()
    _analysis_log(
        verbose,
        f"Captured activations in {perf_counter() - capture_started:.2f}s.",
    )
    ids = input_ids[0].index_select(0, selected).detach().cpu()
    id_values = ids.tolist()
    tokens = tuple(tokenizer.convert_ids_to_tokens(id_values))
    token_texts, token_labels, token_tooltips = _token_presentations(
        tokenizer,
        input_ids[0].detach().cpu().tolist(),
        positions.tolist(),
        tokens,
    )
    result = CaptureResult(
        tokens=tokens,
        token_texts=token_texts,
        token_labels=token_labels,
        token_tooltips=token_tooltips,
        token_groups=token_groups,
        token_ids=ids,
        positions=positions,
        activations=activations,
    )
    _analysis_log(verbose, f"Capture complete in {perf_counter() - started:.2f}s.")
    return result


def _document_framing_for_layer(lens: Any, layer: int) -> dict[str, Any]:
    """Read the portable text-framing policy recorded during fitting."""
    try:
        fitting = lens.metadata["layers"][str(layer)]["fitting"]
        provenance = fitting.get("provenance") or {}
        framing = provenance.get("document_framing")
    except (KeyError, TypeError):
        framing = None
    if not isinstance(framing, dict):
        return {"strategy": "none", "token": None, "token_id": None}
    strategy = framing.get("strategy")
    if strategy not in {"none", "prepend-bos", "prepend-eos"}:
        raise ValueError(f"unsupported recorded document framing: {strategy!r}")
    return framing


def analyze(
    lens: Any,
    inputs: Any,
    *,
    layer: int,
    selected_components: int | tuple[int, ...] | list[int] | None = None,
    document_framing: Literal["model", "none", "recorded"] = "recorded",
    verbose: bool = False,
    **kwargs: Any,
) -> AnalysisResult:
    """Capture an input and calculate signed scores and per-token energy shares."""
    started = perf_counter()
    captured = capture(
        lens,
        inputs,
        layer=layer,
        document_framing=document_framing,
        verbose=verbose,
        **kwargs,
    )
    _analysis_log(verbose, "Computing ICA scores and component energy...")
    transform_started = perf_counter()
    scores = lens.transform(captured.activations, layer=layer)
    energy = lens.energy(scores)
    selected = _normalize_selected_components(selected_components, int(scores.shape[-1]))
    component_profiles = lens._component_profile_summaries(layer)
    _analysis_log(
        verbose,
        f"Computed {scores.shape[-1]} component scores for {scores.shape[0]} tokens "
        f"in {perf_counter() - transform_started:.2f}s.",
    )
    result = AnalysisResult(
        tokens=captured.tokens,
        token_texts=captured.token_texts,
        token_labels=captured.token_labels,
        token_tooltips=captured.token_tooltips,
        token_groups=captured.token_groups,
        token_ids=captured.token_ids,
        positions=captured.positions,
        activations=captured.activations,
        scores=scores,
        energy=energy,
        model=f"{lens.model_id}@{lens.model_revision}",
        layer=layer,
        input_text=(
            inputs
            if isinstance(inputs, str)
            else "\n".join(f"{message['role'].title()}: {message['content']}" for message in inputs)
        ),
        token_scope=(
            "all text tokens" if isinstance(inputs, str) else kwargs.get("token_scope", "all")
        ),
        messages=() if isinstance(inputs, str) else tuple(dict(message) for message in inputs),
        selected_components=selected,
        component_profiles=component_profiles,
    )
    _analysis_log(verbose, f"Analysis complete in {perf_counter() - started:.2f}s.")
    return result


def _normalize_selected_components(
    values: int | tuple[int, ...] | list[int] | None,
    component_count: int,
) -> tuple[int, ...]:
    if values is None:
        return ()
    selected = (values,) if isinstance(values, int) and not isinstance(values, bool) else values
    if not isinstance(selected, (tuple, list)):
        raise TypeError("selected_components must be an integer or a sequence of integers")
    if len(selected) > 1:
        raise ValueError("the explorer currently supports one selected component")
    if any(not isinstance(component, int) or isinstance(component, bool) for component in selected):
        raise TypeError("selected_components must contain only integers")
    if any(component < 0 or component >= component_count for component in selected):
        raise ValueError(f"selected component must be between 0 and {component_count - 1}")
    return tuple(selected)


def add_logit_effects(
    lens: Any,
    result: AnalysisResult,
    *,
    components_per_token: int = 3,
    multiplier: float = 1.1,
    effect_tokens_per_component: int = 10,
    batch_size: int = 32,
    vocabulary_batch_size: int = 16384,
) -> AnalysisResult:
    """Attach local Logit Lens effects for each token's top-scoring components."""
    if result.layer not in lens.available_layers:
        raise ValueError(f"analysis layer {result.layer} is unavailable in this lens")
    if components_per_token <= 0:
        raise ValueError("components_per_token must be positive")
    if not torch.isfinite(torch.tensor(float(multiplier))):
        raise ValueError("multiplier must be finite")
    if effect_tokens_per_component <= 0:
        raise ValueError("effect_tokens_per_component must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if vocabulary_batch_size <= 0:
        raise ValueError("vocabulary_batch_size must be positive")
    model = lens._analysis_model
    tokenizer = lens._analysis_tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError("analyze() must load and cache the model before add_logit_effects()")

    from .profiling import _final_norm, _vocabulary_entries

    final_norm = _final_norm(model)
    output_embeddings = model.get_output_embeddings()
    if final_norm is None or output_embeddings is None:
        raise ValueError("model does not expose a supported final norm and output embeddings")

    selected_count = min(components_per_token, int(result.scores.shape[-1]))
    components = result.scores.abs().topk(selected_count, dim=-1).indices
    token_indices = torch.arange(result.scores.shape[0], device=components.device)
    token_indices = token_indices[:, None].expand_as(components).reshape(-1)
    components = components.reshape(-1)
    edited_scores = result.scores[token_indices].clone()
    rows = torch.arange(edited_scores.shape[0], device=edited_scores.device)
    original_scores = edited_scores[rows, components]
    edited_scores[rows, components] = original_scores * float(multiplier)
    reconstructed = lens.inverse_transform(edited_scores, layer=result.layer)
    references = result.activations[token_indices]
    edited_hidden = lens.restore_norm(reconstructed, reference=references)
    original_hidden = references
    weight = output_embeddings.weight
    device = weight.device
    dtype = weight.dtype
    effects: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, int(edited_hidden.shape[0]), batch_size):
            stop = min(start + batch_size, int(edited_hidden.shape[0]))
            original_normed = final_norm(
                original_hidden[start:stop].to(device=device, dtype=dtype)[:, None, :]
            )
            edited_normed = final_norm(
                edited_hidden[start:stop].to(device=device, dtype=dtype)[:, None, :]
            )
            # Subtract in float32 before unembedding. Subtracting separately
            # computed bf16 logits creates quantized deltas such as repeated 0.5.
            delta_hidden = edited_normed.float() - original_normed.float()
            count = min(effect_tokens_per_component, int(weight.shape[0]))
            best_abs = torch.empty((stop - start, 0), device=device, dtype=torch.float32)
            signed_values = best_abs
            ids = torch.empty((stop - start, 0), device=device, dtype=torch.long)
            for vocabulary_start in range(0, int(weight.shape[0]), vocabulary_batch_size):
                vocabulary_stop = min(
                    vocabulary_start + vocabulary_batch_size, int(weight.shape[0])
                )
                chunk_logits = torch.nn.functional.linear(
                    delta_hidden[:, 0],
                    weight[vocabulary_start:vocabulary_stop].float(),
                    bias=None,
                )
                chunk_count = min(count, int(chunk_logits.shape[-1]))
                chunk_abs, chunk_ids = torch.topk(chunk_logits.abs(), k=chunk_count, dim=-1)
                chunk_signed = chunk_logits.gather(1, chunk_ids)
                chunk_ids = chunk_ids + vocabulary_start
                candidate_abs = torch.cat((best_abs, chunk_abs), dim=1)
                candidate_signed = torch.cat((signed_values, chunk_signed), dim=1)
                candidate_ids = torch.cat((ids, chunk_ids), dim=1)
                keep = min(count, int(candidate_abs.shape[-1]))
                best_abs, selected = torch.topk(candidate_abs, k=keep, dim=-1)
                signed_values = candidate_signed.gather(1, selected)
                ids = candidate_ids.gather(1, selected)
            for offset in range(stop - start):
                item = start + offset
                token_index = int(token_indices[item])
                component = int(components[item])
                effects.append(
                    {
                        "component": component,
                        "token_index": token_index,
                        "position": int(result.positions[token_index]),
                        "token_text": result.token_texts[token_index],
                        "multiplier": float(multiplier),
                        "original_score": float(original_scores[item]),
                        "edited_score": float(original_scores[item] * float(multiplier)),
                        "tokens": _vocabulary_entries(
                            tokenizer, ids[offset], signed_values[offset]
                        ),
                    }
                )
    return replace(result, logit_effects=tuple(effects))


def _intervention_values(
    intervention: tuple[int, float] | Mapping[int, float] | None, *, name: str
) -> dict[int, float]:
    if intervention is None:
        return {}
    values = (
        dict(intervention.items())
        if isinstance(intervention, Mapping)
        else {intervention[0]: intervention[1]}
    )
    if not values:
        raise ValueError(f"{name} mapping cannot be empty")
    for component, value in values.items():
        if isinstance(component, bool) or not isinstance(component, int) or component < 0:
            raise ValueError(f"{name} components must be non-negative integers")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not torch.isfinite(torch.tensor(float(value)))
        ):
            raise ValueError(f"{name} values must be finite numbers")
    return {int(component): float(value) for component, value in values.items()}


def generate(
    lens: Any,
    prompt: str | list[dict[str, str]],
    *,
    layer: int | None = None,
    clamp: tuple[int, float] | Mapping[int, float] | None = None,
    steer: tuple[int, float] | tuple[int, float, float] | Mapping[int, float] | None = None,
    initial_steer: tuple[int, float] | tuple[int, float, float] | Mapping[int, float] | None = None,
    steer_cap: tuple[int, float] | Mapping[int, float] | None = None,
    steering_scope: Literal["current-position", "all-positions"] = "current-position",
    document_framing: Literal["model", "none", "recorded"] = "model",
    max_new_tokens: int = 64,
    device: str | torch.device | None = "auto",
    model: torch.nn.Module | None = None,
    tokenizer: Any = None,
    debug: bool = False,
    **generation_kwargs: Any,
) -> str:
    """Generate text, optionally clamping and/or additively steering ICA scores.

    Both ``steer`` and ``initial_steer`` accept ``(component, offset, cap)``. Step zero
    uses ``initial_steer`` when supplied and otherwise falls back to ``steer``; later
    steps use ``steer``. A cap limits positive (negative) steering at an upper (lower)
    component score without reducing a score already beyond the limit.

    When both interventions are supplied, clamping is applied at every position first,
    followed by additive steering according to ``steering_scope``.

    Raw-text generation uses the tokenizer's normal special-token policy by default,
    matching direct model generation. Use ``"none"`` to suppress special tokens or
    ``"recorded"`` to prepend the BOS/EOS policy stored during fitting.

    Set ``debug=True`` to print the selected components' scores immediately before and
    after each intervention. For interventions affecting multiple positions, the output
    reports the last-position score plus the range and mean across affected positions.
    """
    if isinstance(prompt, list):
        prompt = _normalize_messages(prompt)
        if not prompt:
            raise ValueError("prompt messages cannot be empty")
    elif not isinstance(prompt, str):
        raise TypeError("prompt must be a string or a list of messages")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not isinstance(debug, bool):
        raise TypeError("debug must be a bool")
    shorthand_steer_cap: tuple[int, float] | None = None
    shorthand_initial_cap: tuple[int, float] | None = None
    if isinstance(steer, tuple) and len(steer) == 3:
        if steer_cap is not None:
            raise ValueError("three-item steer cannot be combined with steer_cap")
        component, offset, cap = steer
        steer = (component, offset)
        shorthand_steer_cap = (component, cap)
        steer_cap = shorthand_steer_cap
    if isinstance(initial_steer, tuple) and len(initial_steer) == 3:
        component, offset, cap = initial_steer
        initial_steer = (component, offset)
        shorthand_initial_cap = (component, cap)
    if initial_steer is not None and steer is None:
        raise ValueError("initial_steer requires steer")
    if steer_cap is not None and steer is None:
        raise ValueError("steer_cap requires steer")
    clamp_values = _intervention_values(clamp, name="clamp")
    steer_values = _intervention_values(steer, name="steer")
    has_intervention = bool(clamp_values or steer_values)
    if not has_intervention:
        if layer is not None:
            raise ValueError("layer is only used when clamp or steer is provided")
    else:
        if layer is None:
            raise ValueError("layer is required when clamp or steer is provided")
        if lens.activation_site != "resid_post":
            raise ValueError(
                "generation interventions currently require activation_site='resid_post'"
            )
    if steering_scope not in ("current-position", "all-positions"):
        raise ValueError("steering_scope must be 'current-position' or 'all-positions'")
    if document_framing not in ("model", "none", "recorded"):
        raise ValueError("document_framing must be 'model', 'none', or 'recorded'")

    steering_direction: torch.Tensor | None = None
    if steer is not None:
        assert layer is not None
        artifact = lens._get_layer(layer)
        if lens.row_normalize or artifact.preprocessing_center is not None:
            raise ValueError(
                "additive steering requires an unnormalized Lens without a preprocessing center"
            )
        if artifact.writing_matrix is None:
            raise ValueError(f"layer {layer} has no ICA writing matrix")
        for component in steer_values:
            if component >= artifact.writing_matrix.shape[1]:
                raise ValueError(
                    f"steer component {component} is unavailable; "
                    f"layer {layer} has {artifact.writing_matrix.shape[1]} components"
                )
        steering_direction = torch.zeros(artifact.writing_matrix.shape[0], dtype=torch.float32)
        for component, offset in steer_values.items():
            steering_direction += float(offset) * torch.from_numpy(
                artifact.writing_matrix[:, component]
            )
        initial_values = _intervention_values(initial_steer, name="initial_steer")
        cap_values = _intervention_values(steer_cap, name="steer_cap")
        initial_cap_values = _intervention_values(shorthand_initial_cap, name="initial_steer cap")
        unknown_initial = set(initial_values) - set(steer_values)
        unknown_caps = set(cap_values) - set(steer_values)
        if unknown_initial:
            raise ValueError("initial_steer components must also appear in steer")
        if unknown_caps:
            raise ValueError("steer_cap components must also appear in steer")
        unknown_initial_caps = set(initial_cap_values) - set(initial_values)
        if unknown_initial_caps:
            raise ValueError("initial_steer cap components must appear in initial_steer")
    else:
        initial_values = {}
        cap_values = {}
        initial_cap_values = {}

    model, tokenizer = _resolve_model_and_tokenizer(lens, model, tokenizer, device)
    is_text_prompt = isinstance(prompt, str)
    rendered_prompt = (
        prompt
        if is_text_prompt
        else tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=is_text_prompt and document_framing == "model",
        return_tensors="pt",
    )
    if is_text_prompt and document_framing == "recorded":
        framing_layer = layer
        if framing_layer is None:
            available_layers = tuple(lens.available_layers)
            framing_layer = available_layers[0] if available_layers else None
        framing = (
            _document_framing_for_layer(lens, framing_layer)
            if framing_layer is not None
            else {"strategy": "none", "token": None, "token_id": None}
        )
        prefix_id = framing.get("token_id") if framing.get("strategy") != "none" else None
        if prefix_id is not None:
            prefix = torch.tensor([[int(prefix_id)]], dtype=encoded["input_ids"].dtype)
            encoded["input_ids"] = torch.cat((prefix, encoded["input_ids"]), dim=1)
            if "attention_mask" in encoded:
                prefix_mask = torch.ones((1, 1), dtype=encoded["attention_mask"].dtype)
                encoded["attention_mask"] = torch.cat(
                    (prefix_mask, encoded["attention_mask"]), dim=1
                )
        elif not prompt:
            raise ValueError("an empty prompt requires a recorded BOS/EOS document-framing token")
    elif is_text_prompt and document_framing == "none" and not prompt:
        raise ValueError(
            "an empty prompt requires document_framing='recorded' and a recorded framing token"
        )
    elif is_text_prompt and document_framing == "model" and encoded["input_ids"].shape[1] == 0:
        raise ValueError(
            "the model tokenizer produced no tokens for an empty prompt; use "
            "document_framing='recorded' with a recorded framing token"
        )
    model_device = next(model.parameters()).device
    model_inputs = {
        name: value.to(model_device)
        for name, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    prompt_length = int(model_inputs["input_ids"].shape[1])
    kwargs = {"do_sample": False, **generation_kwargs}
    model_generate = cast(Any, model).generate

    if not has_intervention:
        with torch.inference_mode():
            generated = model_generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
    else:
        assert layer is not None

        def clamp_scores(hidden: torch.Tensor) -> torch.Tensor:
            original_dtype = hidden.dtype
            scores = lens.transform(hidden.float(), layer=layer)
            for component, value in clamp_values.items():
                if component >= scores.shape[-1]:
                    raise ValueError(
                        f"clamp component {component} is unavailable; "
                        f"layer {layer} has {scores.shape[-1]} components"
                    )
                scores[..., component] = float(value)
            reconstructed = cast(torch.Tensor, lens.inverse_transform(scores, layer=layer))
            restored = cast(
                torch.Tensor,
                lens.restore_norm(reconstructed, reference=hidden.float()),
            )
            return restored.to(original_dtype)

        debug_step = 0
        debug_records: list[dict[str, Any]] = []

        def edit(hidden: torch.Tensor) -> torch.Tensor:
            nonlocal debug_step
            edited = clamp_scores(hidden) if clamp_values else hidden
            steering_affected = slice(None)
            if steer_values:
                assert steering_direction is not None
                if steering_scope == "current-position":
                    if hidden.ndim != 3:
                        raise ValueError(
                            "current-position steering requires a [batch, sequence, hidden] tensor"
                        )
                    steering_affected = slice(-1, None)
                use_initial = debug_step == 0 and bool(initial_values)
                step_values = initial_values if use_initial else steer_values
                step_caps = initial_cap_values if use_initial else cap_values
                needs_dynamic_offset = bool(step_caps)
                if not needs_dynamic_offset and step_values == steer_values:
                    steering = steering_direction.to(device=hidden.device, dtype=hidden.dtype)
                    edited = edited.clone()
                    edited[:, steering_affected, :] += steering
                else:
                    before_scores_for_edit = lens.transform(edited.float(), layer=layer)
                    selected_scores = before_scores_for_edit[:, steering_affected, :]
                    score_offsets = torch.zeros_like(selected_scores)
                    for component, configured_offset in step_values.items():
                        offset = float(configured_offset)
                        if component in step_caps:
                            cap = float(step_caps[component])
                            if offset > 0:
                                applied = torch.clamp(
                                    cap - selected_scores[..., component], min=0.0, max=offset
                                )
                            elif offset < 0:
                                applied = torch.clamp(
                                    cap - selected_scores[..., component], min=offset, max=0.0
                                )
                            else:
                                applied = torch.zeros_like(selected_scores[..., component])
                            score_offsets[..., component] = applied
                        else:
                            score_offsets[..., component] = offset
                    writing = torch.from_numpy(artifact.writing_matrix).to(
                        device=hidden.device, dtype=torch.float32
                    )
                    residual_offsets = score_offsets @ writing.T
                    edited = edited.clone()
                    edited[:, steering_affected, :] += residual_offsets.to(hidden.dtype)
            if debug:
                before_scores = lens.transform(hidden.float(), layer=layer)
                after_scores = lens.transform(edited.float(), layer=layer)
                for component in dict.fromkeys((*clamp_values, *steer_values)):
                    affected = slice(None) if component in clamp_values else steering_affected
                    before = before_scores[:, affected, component].detach().cpu().reshape(-1)
                    after = after_scores[:, affected, component].detach().cpu().reshape(-1)
                    debug_records.append(
                        {
                            "step": debug_step,
                            "component": component,
                            "last_before": before[-1].item(),
                            "last_after": after[-1].item(),
                            "affected": before.numel(),
                            "before_min": before.min().item(),
                            "before_mean": before.mean().item(),
                            "before_max": before.max().item(),
                            "after_min": after.min().item(),
                            "after_mean": after.mean().item(),
                            "after_max": after.max().item(),
                        }
                    )
            debug_step += 1
            return edited

        with clamp_resid_post(model, layer=layer, edit=edit), torch.inference_mode():
            generated = model_generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    if not isinstance(sequences, torch.Tensor):
        raise TypeError("model.generate() must return token IDs or an object with .sequences")
    if debug and has_intervention:
        prompt_last_id = int(model_inputs["input_ids"][0, -1].item())
        for record in debug_records:
            step = int(record["step"])
            if step == 0:
                token_id = prompt_last_id
            else:
                generated_index = prompt_length + step - 1
                token_id = (
                    int(sequences[0, generated_index].item())
                    if generated_index < sequences.shape[1]
                    else -1
                )
            token_text = (
                str(tokenizer.decode([token_id], skip_special_tokens=False))
                if token_id >= 0
                else "<unavailable>"
            )
            print(
                f"[ICALens debug] step={step} token={token_text!r} "
                f"C{record['component']} "
                f"last={record['last_before']:+.4f}->{record['last_after']:+.4f} "
                f"affected={record['affected']} "
                f"before[min/mean/max]={record['before_min']:+.4f}/"
                f"{record['before_mean']:+.4f}/{record['before_max']:+.4f} "
                f"after[min/mean/max]={record['after_min']:+.4f}/"
                f"{record['after_mean']:+.4f}/{record['after_max']:+.4f}"
            )
    continuation = sequences[0, prompt_length:].detach().cpu()
    return str(tokenizer.decode(continuation, skip_special_tokens=True))


def _token_presentations(
    tokenizer: Any,
    all_ids: list[int],
    positions: list[int],
    tokens: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Create readable labels and contextual tooltips for tokenizer fragments."""
    texts: list[str] = []
    labels: list[str] = []
    tooltips: list[str] = []
    for position, token in zip(positions, tokens, strict=True):
        token_id = all_ids[position]
        decoded = str(tokenizer.decode([token_id]))
        texts.append(decoded)
        if "\ufffd" not in decoded:
            labels.append(decoded)
            tooltips.append(token)
            continue

        label = "<?>"
        labels.append(label)
        combined = _decode_smallest_valid_span(tokenizer, all_ids, position)
        tooltip = f"Token ID {token_id} (0x{token_id:04X}); incomplete UTF-8 fragment"
        if combined is not None:
            start, end, text = combined
            tooltip += f"; tokens {start}–{end - 1} decode together as {text!r}"
        tooltips.append(tooltip)
    return tuple(texts), tuple(labels), tuple(tooltips)


def _decode_smallest_valid_span(
    tokenizer: Any, all_ids: list[int], position: int, *, max_tokens: int = 8
) -> tuple[int, int, str] | None:
    """Find the shortest neighboring span containing a fragment that decodes cleanly."""
    for length in range(2, max_tokens + 1):
        first_start = max(0, position - length + 1)
        last_start = min(position, len(all_ids) - length)
        for start in range(first_start, last_start + 1):
            end = start + length
            decoded = str(tokenizer.decode(all_ids[start:end]))
            if decoded and "\ufffd" not in decoded:
                return start, end, decoded
    return None


def _resolve_model_and_tokenizer(
    lens: Any,
    model: torch.nn.Module | None,
    tokenizer: Any,
    device: str | torch.device | None,
    *,
    verbose: bool = False,
) -> tuple[torch.nn.Module, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "ICA Lens analysis dependencies are missing; reinstall or upgrade 'icalens'"
        ) from error
    if lens.model_revision is None:
        from huggingface_hub import HfApi

        resolved = HfApi().model_info(lens.model_id).sha
        if resolved is None:
            raise RuntimeError(f"could not resolve an exact revision for {lens.model_id}")
        lens.model_revision = str(resolved)
    if tokenizer is None:
        tokenizer = lens._analysis_tokenizer
        if tokenizer is None:
            _analysis_log(verbose, "Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                lens.model_id, revision=lens.model_revision, use_fast=True
            )
            lens._analysis_tokenizer = tokenizer
        else:
            _analysis_log(verbose, "Reusing cached tokenizer.")
    if model is None:
        target = _resolve_device(device)
        model = lens._analysis_model
        if model is None or lens._analysis_device != target:
            _analysis_log(verbose, f"Loading model weights on {target}...")
            load_started = perf_counter()
            if not target.startswith("cuda"):
                model = AutoModelForCausalLM.from_pretrained(
                    lens.model_id, revision=lens.model_revision
                ).to(target)  # type: ignore[arg-type]
            else:
                try:
                    from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
                except ImportError as error:
                    raise ImportError(
                        "ICA Lens CUDA model-loading dependencies are missing; "
                        "reinstall or upgrade 'icalens'"
                    ) from error
                model = load_model_to_cuda(
                    AutoModelForCausalLM,
                    lens.model_id,
                    revision=lens.model_revision,
                    device=target,
                    dtype=torch.bfloat16,
                    touch="auto",
                    low_cpu_mem_usage=True,
                )
            model.eval()
            lens._analysis_model = model
            lens._analysis_device = target
            _analysis_log(verbose, f"Loaded model in {perf_counter() - load_started:.2f}s.")
        else:
            _analysis_log(verbose, f"Reusing cached model on {target}.")
    return model, tokenizer


def _analysis_log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[ICALens] {message}", flush=True)


def _resolve_device(device: str | torch.device | None) -> str:
    """Resolve the public automatic-device setting to a concrete target."""
    if device is None or str(device) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _encode_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    token_scope: str,
    context_length: int | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, tuple[str, ...]]:
    if not tokenizer.is_fast or tokenizer.chat_template is None:
        raise ValueError("chat analysis requires a fast tokenizer with a chat template")
    normalized = _normalize_messages(messages)
    rendered = tokenizer.apply_chat_template(
        normalized, tokenize=False, add_generation_prompt=False
    )
    encoded_offsets = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=context_length is not None,
        max_length=context_length,
        return_offsets_mapping=True,
    )
    offsets = encoded_offsets["offset_mapping"]
    if token_scope == "all":
        selected = list(range(len(encoded_offsets["input_ids"])))
    else:
        spans = _content_spans(rendered, normalized, token_scope)
        selected = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and any(end > left and start < right for left, right in spans)
        ]
    if not selected:
        raise ValueError(f"conversation contains no {token_scope} tokens")
    group_titles = _chat_group_titles(
        rendered,
        normalized,
        offsets,
        tokenizer.convert_ids_to_tokens(encoded_offsets["input_ids"]),
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=context_length is not None,
        max_length=context_length,
        return_tensors="pt",
    )
    return (
        encoded,
        torch.tensor(selected, dtype=torch.long),
        tuple(group_titles[index] for index in selected),
    )


def _chat_group_titles(
    rendered: str,
    messages: list[dict[str, str]],
    offsets: list[tuple[int, int]],
    input_tokens: list[str],
) -> list[str]:
    """Assign every rendered chat token to a template or message group."""
    starts: list[int] = []
    visible_messages: list[dict[str, str]] = []
    cursor = 0
    for message in messages:
        content_start = rendered.find(message["content"], cursor)
        if content_start < 0:
            raise ValueError("chat template transformed message content; offsets are ambiguous")
        cursor = content_start + len(message["content"])
        content_token = next(
            (index for index, (start, end) in enumerate(offsets) if end > content_start),
            None,
        )
        if content_token is None:
            break
        opening_candidates = [
            index
            for index in range(content_token + 1)
            if ("start" in input_tokens[index].lower() or "begin" in input_tokens[index].lower())
            and (not starts or index > starts[-1])
        ]
        starts.append(opening_candidates[-1] if opening_candidates else content_token)
        visible_messages.append(message)

    titles = ["Prompt template"] * len(offsets)
    role_counts = {"system": 0, "user": 0, "assistant": 0}
    for message, start, end in zip(
        visible_messages, starts, [*starts[1:], len(offsets)], strict=True
    ):
        role = message["role"]
        role_counts[role] += 1
        title = role.title() if role == "system" else f"{role.title()} {role_counts[role]}"
        titles[start:end] = [title] * (end - start)
    return titles


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise TypeError("messages must be a non-empty list")
    result: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
        }:
            raise ValueError("messages must contain system/user/assistant roles")
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError("message content must be a non-empty string")
        result.append({"role": message["role"], "content": content})
    return result


def _content_spans(
    rendered: str, messages: list[dict[str, str]], token_scope: str
) -> list[tuple[int, int]]:
    if token_scope not in {"assistant", "user", "content"}:
        raise ValueError(f"unsupported token scope: {token_scope!r}")
    spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        start = rendered.find(message["content"], cursor)
        if start < 0:
            raise ValueError("chat template transformed message content; offsets are ambiguous")
        end = start + len(message["content"])
        if token_scope == "content" or message["role"] == token_scope:
            spans.append((start, end))
        cursor = end
    return spans
