"""The main ICA Lens API."""

from __future__ import annotations

import copy
import json
import re
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from numpy.typing import NDArray

from ._arrays import transform_array
from ._artifact import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    MINIMUM_PACKAGE_VERSION,
    LayerArtifact,
    layer_from_manifest,
    load_layer,
    load_profile,
    read_manifest,
    save_directory,
    save_profile_checkpoint,
)
from ._fastica import OBJECTIVE_PERCENTILES, fit_fastica, geometric_median
from .exceptions import ArtifactError, NotFittedError

if TYPE_CHECKING:
    from .analysis import ComponentProfile
    from .erf import ERFAnalysis


class ICALens:
    """A model-level collection of fitted layerwise ICA transformations."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        model_revision: str | None = None,
        model_type: Literal["base", "instruct"] = "base",
        base_model: str | None = None,
        base_model_revision: str | None = None,
        activation_site: str = "resid_post",
        layer_indexing: str = "hidden_states",
        row_normalize: bool | None = None,
        icalens_preprocessing: Literal["none", "l2", "geometric-median-l2"] | None = None,
        norm_eps: float = 1e-12,
    ) -> None:
        model_id = _resolve_compatibility_argument("model_id", model_id, "base_model", base_model)
        model_revision = _resolve_optional_compatibility_argument(
            "model_revision", model_revision, "base_model_revision", base_model_revision
        )
        model_type = _validate_model_type(model_type)
        if not isinstance(activation_site, str) or not activation_site.strip():
            raise ValueError("activation_site must be a non-empty string")
        if re.fullmatch(r"[A-Za-z0-9_.-]+", activation_site.strip()) is None:
            raise ValueError("activation_site may contain only letters, digits, '.', '_', and '-'")
        if not isinstance(layer_indexing, str) or not layer_indexing.strip():
            raise ValueError("layer_indexing must be a non-empty string")
        if not np.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be a finite positive number")
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_type = model_type
        self.activation_site = activation_site.strip()
        self.layer_indexing = layer_indexing.strip()
        if icalens_preprocessing is None:
            icalens_preprocessing = "l2" if row_normalize is not False else "none"
        if icalens_preprocessing not in {"none", "l2", "geometric-median-l2"}:
            raise ValueError("icalens_preprocessing must be 'none', 'l2', or 'geometric-median-l2'")
        if row_normalize is not None:
            expected = icalens_preprocessing != "none"
            if bool(row_normalize) != expected:
                raise ValueError("row_normalize conflicts with icalens_preprocessing")
        self.icalens_preprocessing = icalens_preprocessing
        self.row_normalize = icalens_preprocessing != "none"
        self.norm_eps = float(norm_eps)
        self._hidden_size: int | None = None
        self._layers: dict[int, LayerArtifact] = {}
        self._local_root: Path | None = None
        self._hub_source: dict[str, Any] | None = None
        self._r_lens_profiles: dict[str, Any] = {}
        self._analysis_model: torch.nn.Module | None = None
        self._analysis_tokenizer: Any = None
        self._analysis_device: str | None = None
        self._erf_analysis: ERFAnalysis | None = None

    @property
    def available_layers(self) -> tuple[int, ...]:
        """Sorted layer indices present in this lens."""
        return tuple(sorted(self._layers))

    @property
    def erf(self) -> ERFAnalysis:
        """Effective-receptive-field measurements bound to this Lens."""
        if self._erf_analysis is None:
            from .erf import ERFAnalysis

            self._erf_analysis = ERFAnalysis(self)
        return self._erf_analysis

    @property
    def base_model(self) -> str:
        """Deprecated compatibility alias for :attr:`model_id`."""
        return self.model_id

    @property
    def base_model_revision(self) -> str | None:
        """Deprecated compatibility alias for :attr:`model_revision`."""
        return self.model_revision

    @property
    def hidden_size(self) -> int | None:
        """Activation width, or ``None`` before the first fit."""
        return self._hidden_size

    @property
    def metadata(self) -> dict[str, Any]:
        """A detached copy of the portable artifact manifest."""
        return copy.deepcopy(self._manifest())

    def plot_fitting_curve(
        self,
        *,
        layer: int | None = None,
        layers: list[int] | tuple[int, ...] | Literal["all"] | None = None,
        columns: int | None = None,
    ) -> Any:
        """Plot saved FastICA objective distributions for fitted layers.

        Returns a Matplotlib figure. In Jupyter, leave the returned figure as the
        final expression in a cell to display it inline.
        """
        from ._plotting import plot_fitting_curves

        if (layer is None) == (layers is None):
            raise ValueError("pass exactly one of layer or layers")
        if columns is not None and (isinstance(columns, bool) or not isinstance(columns, int)):
            raise TypeError("columns must be a positive integer or None")
        if columns is not None and columns <= 0:
            raise ValueError("columns must be positive")
        if layer is not None:
            selected = [_validate_layer(layer)]
        elif layers == "all":
            selected = list(self.available_layers)
        elif isinstance(layers, (list, tuple)):
            selected = [_validate_layer(value) for value in layers]
        else:
            raise TypeError("layers must be a list, tuple, or 'all'")
        if not selected:
            raise ValueError("layers must not be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("layers must not contain duplicates")
        missing = [value for value in selected if value not in self._layers]
        if missing:
            raise NotFittedError(
                f"layers {missing} are unavailable; available layers: {self.available_layers}"
            )
        return plot_fitting_curves(
            layers=[(value, self._layers[value].fitting) for value in selected],
            model_id=self.model_id,
            columns=columns,
        )

    def fit(
        self,
        activations: Any,
        *,
        layer: int,
        n_components: int | None = None,
        algorithm: str = "parallel",
        fun: str = "logcosh",
        max_iter: int = 200,
        random_state: int | None = 0,
        progress: bool = False,
        device: str | torch.device | None = None,
        batch_size: int = 8192,
        objective_every: int = 1,
        provenance: dict[str, Any] | None = None,
    ) -> ICALens:
        """Fit or replace the ICA transformation for one layer."""
        layer = _validate_layer(layer)
        values = _as_fit_tensor(activations)
        hidden_size = int(values.shape[1])
        if self._hidden_size is not None and hidden_size != self._hidden_size:
            raise ValueError(
                f"activation hidden size {hidden_size} does not match lens hidden size "
                f"{self._hidden_size}"
            )
        components = hidden_size if n_components is None else int(n_components)
        maximum = min(int(values.shape[0]) - 1, hidden_size)
        if maximum <= 0:
            raise ValueError("at least two activation samples are required after centering")
        if components <= 0 or components > maximum:
            raise ValueError(
                f"n_components must be between 1 and {maximum}, got {components}; "
                "centering limits the data rank to at most n_samples - 1"
            )
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if objective_every <= 0:
            raise ValueError("objective_every must be positive")
        provenance = _validate_provenance(provenance)

        preprocessing_center = None
        if self.icalens_preprocessing == "geometric-median-l2":
            if progress:
                print("Estimating geometric-median center before L2 normalization...", flush=True)
            preprocessing_center = geometric_median(
                values,
                device=device,
                batch_size=batch_size,
                progress=progress,
            )

        result = fit_fastica(
            values,
            n_components=components,
            algorithm=algorithm,
            fun=fun,
            max_iter=max_iter,
            random_state=random_state,
            progress=progress,
            device=device,
            batch_size=batch_size,
            row_normalize=self.row_normalize,
            preprocessing_center=preprocessing_center,
            norm_eps=self.norm_eps,
            objective_every=objective_every,
        )

        center = _to_numpy(result.center)
        reading = _to_numpy(result.components)
        writing = _to_numpy(result.mixing)
        filename = f"artifacts/{self.activation_site}/layer_{layer:02d}.safetensors"
        self._layers[layer] = LayerArtifact(
            layer=layer,
            file=filename,
            n_components=components,
            fitting={
                "algorithm": "fastica",
                "implementation": "icalens.torch",
                "implementation_version": "2",
                "torch_version": torch.__version__,
                "ica_algorithm": algorithm,
                "fun": fun,
                "whiten": "unit-variance",
                "whiten_solver": "eigh",
                "source_scaling": "none",
                "max_iter": int(max_iter),
                "stopping_criterion": "fixed_iterations",
                "random_state": random_state,
                "n_iter": result.n_iter,
                "objective_every": int(objective_every),
                "objective_history": (
                    {
                        "contrast": fun,
                        "aggregation": "mean_over_tokens_then_percentiles_over_components",
                        "iterations": result.objective_iterations,
                        "percentiles": list(OBJECTIVE_PERCENTILES),
                        "values": result.objective_history,
                    }
                    if result.objective_history is not None
                    else None
                ),
                "gaussian_objective": result.gaussian_objective,
                "component_objectives": result.component_objectives,
                "component_strengths": result.component_strengths,
                "n_samples": int(values.shape[0]),
                "input_dtype": str(values.dtype).removeprefix("torch."),
                "fit_device": str(result.center.device),
                "batch_size": int(batch_size),
                "memory_strategy": "blockwise_multi_pass",
                "icalens_preprocessing": self.icalens_preprocessing,
                "stored_dtype": "float32",
                "component_id_convention": (
                    "descending absolute contrast deviation from Gaussian; "
                    "C0 is strongest; no sign canonicalization"
                ),
                "provenance": provenance,
            },
            center=center,
            reading_matrix=reading,
            writing_matrix=writing,
            preprocessing_center=(
                _to_numpy(preprocessing_center) if preprocessing_center is not None else None
            ),
        )
        self._hidden_size = hidden_size
        self._local_root = None
        self._hub_source = None
        return self

    def transform(self, activations: Any, *, layer: int) -> Any:
        """Map activations to signed ICA component scores."""
        artifact = self._get_layer(layer)
        self._validate_final_dimension(activations, self._hidden_size)
        assert artifact.center is not None and artifact.reading_matrix is not None
        return transform_array(
            activations,
            matrix=artifact.reading_matrix,
            offset=artifact.center,
            pre_offset=artifact.preprocessing_center,
            normalize=self.row_normalize,
            norm_eps=self.norm_eps,
        )

    @staticmethod
    def energy(scores: Any) -> Any:
        """Return each component's fraction of per-position squared score energy."""
        if isinstance(scores, torch.Tensor):
            if scores.ndim < 1 or not scores.is_floating_point():
                raise TypeError("scores must be a floating-point array")
            squared = scores.square()
            denominator = squared.sum(dim=-1, keepdim=True)
            return torch.where(denominator > 0, squared / denominator, torch.zeros_like(squared))
        array = np.asarray(scores)
        if array.ndim < 1 or not np.issubdtype(array.dtype, np.floating):
            raise TypeError("scores must be a floating-point array")
        squared_array = np.square(array)
        denominator_array = squared_array.sum(axis=-1, keepdims=True)
        return np.divide(
            squared_array,
            denominator_array,
            out=np.zeros_like(squared_array),
            where=denominator_array > 0,
        )

    @staticmethod
    def keep_topk(scores: Any, k: int) -> Any:
        """Keep the top-k absolute scores in each vector and zero the rest."""
        return _mask_topk_scores(scores, k=k, keep=True)

    @staticmethod
    def ablate_topk(scores: Any, k: int) -> Any:
        """Zero the top-k absolute scores in each vector and keep the rest."""
        return _mask_topk_scores(scores, k=k, keep=False)

    def restore_norm(self, values: Any, *, reference: Any) -> Any:
        """Restore per-vector norms from reference activations."""
        if isinstance(values, torch.Tensor) and isinstance(reference, torch.Tensor):
            _validate_matching_arrays(values, reference)
            value_norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
            reference_norms = torch.linalg.vector_norm(reference, dim=-1, keepdim=True).to(
                device=values.device, dtype=values.dtype
            )
            return values / value_norms.clamp_min(self.norm_eps) * reference_norms
        if isinstance(values, torch.Tensor) or isinstance(reference, torch.Tensor):
            raise TypeError("values and reference must both be PyTorch tensors or NumPy arrays")
        value_array = np.asarray(values)
        reference_array = np.asarray(reference)
        _validate_matching_arrays(value_array, reference_array)
        value_norms_array = np.linalg.norm(value_array, axis=-1, keepdims=True)
        reference_norms_array = np.linalg.norm(reference_array, axis=-1, keepdims=True).astype(
            value_array.dtype, copy=False
        )
        return value_array / np.maximum(value_norms_array, self.norm_eps) * reference_norms_array

    def capture(self, inputs: Any, *, layer: int, **kwargs: Any) -> Any:
        """Capture model activations aligned to text or conversation tokens."""
        from .analysis import capture

        return capture(self, inputs, layer=layer, **kwargs)

    def analyze(self, inputs: Any, *, layer: int, **kwargs: Any) -> Any:
        """Capture activations and return tokens, ICA scores, and energy shares."""
        from .analysis import analyze

        return analyze(self, inputs, layer=layer, **kwargs)

    def add_logit_effects(self, result: Any, **kwargs: Any) -> Any:
        """Attach token-local vocabulary effects from scaling one ICA score."""
        from .analysis import add_logit_effects

        return add_logit_effects(self, result, **kwargs)

    def profile_components(self, inputs: Any, *, layer: int, **kwargs: Any) -> dict[str, Any]:
        """Build sign, example-token, and logit-lens profiles without refitting."""
        from .profiling import profile_components

        return profile_components(self, inputs, layer=layer, **kwargs)

    def profile_components_from_activations(
        self,
        activations: torch.Tensor,
        records: list[dict[str, Any]],
        *,
        layer: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Profile components from token-aligned activations captured earlier."""
        from .profiling import profile_components_from_activations

        return profile_components_from_activations(
            self, activations, records, layer=layer, **kwargs
        )

    def add_r_lens_profile(self, *, layer: int, r_lens: Any, **kwargs: Any) -> dict[str, Any]:
        """Add R-lens token readouts to an existing component profile."""
        from .profiling import add_r_lens_profile

        return add_r_lens_profile(self, layer=layer, r_lens=r_lens, **kwargs)

    def refresh_profile_statistics_from_activations(
        self, activations: torch.Tensor, *, layer: int, **kwargs: Any
    ) -> dict[str, Any]:
        """Refresh profile moments and tail selection from captured activations."""
        from .profiling import refresh_profile_statistics_from_activations

        return refresh_profile_statistics_from_activations(self, activations, layer=layer, **kwargs)

    def refresh_profile_examples_from_activations(
        self,
        activations: torch.Tensor,
        records: list[dict[str, Any]],
        *,
        layer: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Refresh selected-tail examples from captured activations."""
        from .profiling import refresh_profile_examples_from_activations

        return refresh_profile_examples_from_activations(
            self, activations, records, layer=layer, **kwargs
        )

    def refresh_profile_example_ranks_from_activations(
        self,
        activations: torch.Tensor,
        occurrence_rows: list[tuple[int, str, int, int]],
        *,
        layer: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add absolute-score ranks to stored selected-tail occurrences."""
        from .profiling import refresh_profile_example_ranks_from_activations

        return refresh_profile_example_ranks_from_activations(
            self, activations, occurrence_rows, layer=layer, **kwargs
        )

    def component_profile(self, *, layer: int, component: int) -> ComponentProfile:
        """Return a dictionary-compatible, notebook-displayable component profile."""
        from .analysis import ComponentProfile

        artifact = self._get_layer(layer)
        profile = self._get_profile(artifact)
        if isinstance(component, bool) or not isinstance(component, (int, np.integer)):
            raise TypeError("component must be a non-negative integer")
        index = int(component)
        if index < 0 or index >= artifact.n_components:
            raise ValueError(
                f"component must be between 0 and {artifact.n_components - 1}, got {index}"
            )
        value = copy.deepcopy(profile["components"][index])
        fitting_statistics = _component_fitting_statistics(artifact, index)
        if fitting_statistics is not None:
            value["fitting_statistics"] = fitting_statistics
        return ComponentProfile(value, layer=layer)

    def top_occurrence(
        self,
        *,
        layer: int,
        component: int,
        id: int = 0,
        sign: Literal["positive", "negative"] | None = None,
        full: bool = False,
    ) -> dict[str, Any]:
        """Return one stored occurrence, optionally with its complete source text."""
        if isinstance(id, bool) or not isinstance(id, (int, np.integer)) or id < 0:
            raise ValueError("id must be a non-negative integer")
        if sign not in {None, "positive", "negative"}:
            raise ValueError("sign must be 'positive', 'negative', or None")
        artifact = self._get_layer(layer)
        layer_profile = self._get_profile(artifact)
        component_profile = self.component_profile(layer=layer, component=component)
        selected_sign = (
            sign
            or component_profile.get("tail_direction")
            or component_profile.get("dominant_sign")
        )
        if selected_sign not in {"positive", "negative"}:
            raise ValueError("component profile has no selected tail direction")
        occurrences = (
            component_profile.get("examples", {}).get(selected_sign, {}).get("occurrences", [])
        )
        index = int(id)
        if index >= len(occurrences):
            raise IndexError(
                f"occurrence id {index} is unavailable; {len(occurrences)} stored for the "
                f"{selected_sign} tail"
            )
        occurrence = copy.deepcopy(occurrences[index])
        if not isinstance(occurrence, dict):
            raise ArtifactError("stored occurrence must be an object")
        if not full:
            return occurrence

        provenance = layer_profile.get("example_provenance") or layer_profile.get("provenance")
        source = provenance.get("dataset") if isinstance(provenance, dict) else None
        if not isinstance(source, dict) or not source.get("repo_id"):
            raise ValueError("full source text is unavailable: profile has no dataset provenance")
        if "source_index" not in occurrence:
            raise ValueError("full source text is unavailable: occurrence has no source_index")
        from datasets import load_dataset  # type: ignore[import-untyped]

        dataset = load_dataset(
            source["repo_id"],
            split=source.get("split", "train"),
            revision=source.get("revision"),
        )
        assert isinstance(provenance, dict)
        text_field = provenance.get("text_field", "text")
        occurrence["full_text"] = str(dataset[int(occurrence["source_index"])][text_field])
        return occurrence

    def checkpoint_component_profile(self, path: str | Path, *, layer: int) -> Path:
        """Write one completed profile into an existing local lens artifact."""
        artifact = self._get_layer(layer)
        if artifact.profile is None:
            raise NotFittedError(f"layer {layer} has no in-memory component profile to checkpoint")
        destination = Path(path).expanduser().resolve()
        save_profile_checkpoint(destination, self._manifest(), artifact)
        return destination

    def generate(self, prompt: Any, **kwargs: Any) -> str:
        """Generate a continuation, optionally clamping or steering ICA coordinates."""
        from .analysis import generate

        return generate(self, prompt, **kwargs)

    def unload_model(self) -> None:
        """Release the model and tokenizer loaded lazily by capture, analyze, or generate."""
        self._analysis_model = None
        self._analysis_tokenizer = None
        self._analysis_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def inverse_transform(self, scores: Any, *, layer: int) -> Any:
        """Approximately reconstruct preprocessed activations from scores."""
        artifact = self._get_layer(layer)
        self._validate_final_dimension(scores, artifact.n_components)
        assert artifact.center is not None and artifact.writing_matrix is not None
        return transform_array(
            scores,
            matrix=artifact.writing_matrix,
            offset=np.zeros(artifact.n_components, dtype=np.float32),
            normalize=False,
            norm_eps=self.norm_eps,
        ) + _array_offset(scores, artifact.center)

    def save(self, path: str | Path) -> Path:
        """Write this lens to a local artifact directory and return its path."""
        if not self._layers:
            raise NotFittedError("cannot save an ICA Lens with no fitted layers")
        for layer in self.available_layers:
            artifact = self._get_layer(layer)
            if artifact.profile_file is not None:
                self._get_profile(artifact)
        destination = Path(path).expanduser().resolve()
        save_directory(destination, self._manifest(), self._layers)
        return destination

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool | None = None,
        token: str | bool | None = None,
        revision: str = "main",
        commit_message: str = "Upload ICA Lens artifacts",
    ) -> str:
        """Create or update a Hugging Face Model repository."""
        if not repo_id or "/" not in repo_id:
            raise ValueError("repo_id must have the form 'owner/name'")
        with tempfile.TemporaryDirectory(prefix="icalens-upload-") as temporary:
            local_dir = self.save(Path(temporary) / "artifact")
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
            result = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=local_dir,
                revision=revision,
                commit_message=commit_message,
            )
        return str(result)

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> ICALens:
        """Load a lens manifest from a local directory or Hugging Face Hub."""
        candidate = Path(repo_id_or_path).expanduser()
        hub_source: dict[str, Any] | None = None
        if candidate.is_dir():
            root = candidate.resolve()
            manifest_path = root / MANIFEST_FILENAME
        else:
            repo_id = str(repo_id_or_path)
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=MANIFEST_FILENAME,
                    repo_type="model",
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                    force_download=force_download,
                )
            except Exception as error:
                raise ArtifactError(
                    f"could not download {repo_id}/{MANIFEST_FILENAME}: {error}"
                ) from error
            manifest_path = Path(downloaded)
            root = None
            hub_source = {
                "repo_id": repo_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "token": token,
                "local_files_only": local_files_only,
                "force_download": force_download,
            }

        manifest = read_manifest(manifest_path)
        preprocessing = manifest["input_preprocessing"]
        if not isinstance(preprocessing, dict):
            raise ArtifactError("manifest input_preprocessing must be an object")
        normalization = preprocessing.get("row_normalization")
        if normalization not in ("l2", "none"):
            raise ArtifactError(f"unsupported row_normalization: {normalization!r}")
        format_version = int(manifest["format_version"])
        model = manifest["base_model"] if format_version == 1 else manifest["model"]
        preprocessing_mode = preprocessing.get("icalens_preprocessing")
        if preprocessing_mode is None:
            preprocessing_mode = "l2" if normalization == "l2" else "none"
        if preprocessing_mode not in {"none", "l2", "geometric-median-l2"}:
            raise ArtifactError(f"unsupported icalens_preprocessing: {preprocessing_mode!r}")
        preprocessing_mode = cast(Literal["none", "l2", "geometric-median-l2"], preprocessing_mode)
        lens = cls(
            model_id=str(model["repo_id"]),
            model_revision=str(model.get("revision") or "unknown"),
            model_type=_validate_model_type(str(model.get("type", "base"))),
            activation_site=str(manifest["activation_site"]),
            layer_indexing=str(manifest.get("layer_indexing", "hidden_states")),
            icalens_preprocessing=preprocessing_mode,
            norm_eps=float(preprocessing.get("norm_eps", 1e-12)),
        )
        try:
            lens._hidden_size = int(manifest["hidden_size"])
        except (TypeError, ValueError) as error:
            raise ArtifactError("manifest hidden_size must be a positive integer") from error
        if lens._hidden_size <= 0:
            raise ArtifactError("manifest hidden_size must be a positive integer")
        lens._layers = {
            int(key): layer_from_manifest(key, value) for key, value in manifest["layers"].items()
        }
        stored_r_lens_profiles = manifest.get("r_lens_profiles", {})
        lens._r_lens_profiles = (
            copy.deepcopy(stored_r_lens_profiles)
            if isinstance(stored_r_lens_profiles, dict)
            else {}
        )
        lens._local_root = root
        lens._hub_source = hub_source
        return lens

    def _get_layer(self, layer: int) -> LayerArtifact:
        layer = _validate_layer(layer)
        if layer not in self._layers:
            available = ", ".join(map(str, self.available_layers)) or "none"
            raise NotFittedError(f"layer {layer} is unavailable; available layers: {available}")
        artifact = self._layers[layer]
        if not artifact.loaded:
            if self._hidden_size is None:
                raise ArtifactError("artifact has no hidden size")
            if self._local_root is not None:
                tensor_path = self._local_root / artifact.file
            elif self._hub_source is not None:
                source = self._hub_source
                try:
                    tensor_path = Path(
                        hf_hub_download(
                            repo_id=source["repo_id"],
                            filename=artifact.file,
                            repo_type="model",
                            revision=source["revision"],
                            cache_dir=source["cache_dir"],
                            token=source["token"],
                            local_files_only=source["local_files_only"],
                            force_download=source["force_download"],
                        )
                    )
                except Exception as error:
                    raise ArtifactError(f"could not download layer {layer}: {error}") from error
            else:
                raise ArtifactError(f"no source is available for unloaded layer {layer}")
            load_layer(tensor_path, artifact, self._hidden_size)
        return artifact

    def _get_profile(self, artifact: LayerArtifact) -> dict[str, Any]:
        if artifact.profile is not None:
            return artifact.profile
        if artifact.profile_file is None:
            raise NotFittedError(
                f"layer {artifact.layer} has no component profile; call profile_components()"
            )
        if self._local_root is not None:
            profile_path = self._local_root / artifact.profile_file
        elif self._hub_source is not None:
            source = self._hub_source
            try:
                profile_path = Path(
                    hf_hub_download(
                        repo_id=source["repo_id"],
                        filename=artifact.profile_file,
                        repo_type="model",
                        revision=source["revision"],
                        cache_dir=source["cache_dir"],
                        token=source["token"],
                        local_files_only=source["local_files_only"],
                        force_download=source["force_download"],
                    )
                )
            except Exception as error:
                raise ArtifactError(
                    f"could not download component profile for layer {artifact.layer}: {error}"
                ) from error
        else:
            raise ArtifactError(f"no source is available for layer {artifact.layer} profile")
        artifact.profile = load_profile(profile_path)
        return artifact.profile

    def _component_profile_summaries(self, layer: int) -> dict[int, dict[str, Any]] | None:
        artifact = self._get_layer(layer)
        if artifact.profile_file is None:
            return None
        profile = self._get_profile(artifact)
        summaries: dict[int, dict[str, Any]] = {}
        for component in profile["components"]:
            component_index = int(component["component"])
            sign = str(component["dominant_sign"])
            summaries[component_index] = {
                "dominant_sign": sign,
                "tail_direction": component.get("tail_direction"),
                "sign_statistics": component["sign_statistics"],
                "score_statistics": component.get("score_statistics"),
                "fitting_statistics": _component_fitting_statistics(artifact, component_index),
                "occurrences": [
                    {
                        key: occurrence[key]
                        for key in (
                            "text",
                            "token",
                            "token_id",
                            "context",
                            "context_target_start",
                            "context_target_end",
                            "score",
                            "energy",
                            "absolute_score_rank",
                        )
                        if key in occurrence
                    }
                    for occurrence in component["examples"][sign]["occurrences"]
                ],
                "logit_tokens": [
                    {
                        key: token[key]
                        for key in ("text", "token", "token_id", "logit")
                        if key in token
                    }
                    for token in component["logit_lens"]["dominant"]["top_tokens"][:10]
                ],
                "r_lens_tokens": [
                    {
                        key: token[key]
                        for key in ("text", "token", "token_id", "logit")
                        if key in token
                    }
                    for token in component.get("r_lens", {})
                    .get("dominant", {})
                    .get("top_tokens", [])[:10]
                ],
            }
        return summaries

    def _manifest(self) -> dict[str, Any]:
        if self._hidden_size is None:
            hidden_size = 0
        else:
            hidden_size = self._hidden_size
        r_lens_layers: dict[str, Any] = copy.deepcopy(self._r_lens_profiles)
        for layer, artifact in sorted(self._layers.items()):
            if artifact.profile is None:
                continue
            provenance = artifact.profile.get("r_lens_provenance")
            if isinstance(provenance, dict):
                r_lens_layers[str(layer)] = copy.deepcopy(provenance)
        return {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "minimum_package_version": MINIMUM_PACKAGE_VERSION,
            "package_version": _package_version(),
            "model": {
                "repo_id": self.model_id,
                "revision": self.model_revision,
                "type": self.model_type,
            },
            "activation_site": self.activation_site,
            "layer_indexing": self.layer_indexing,
            "hidden_size": hidden_size,
            "input_preprocessing": {
                "icalens_preprocessing": self.icalens_preprocessing,
                "row_normalization": "l2" if self.row_normalize else "none",
                "pre_normalization_center": (
                    "geometric_median"
                    if self.icalens_preprocessing == "geometric-median-l2"
                    else "none"
                ),
                "norm_eps": self.norm_eps,
            },
            **({"r_lens_profiles": r_lens_layers} if r_lens_layers else {}),
            "layers": {
                str(layer): {
                    "file": artifact.file,
                    "n_components": artifact.n_components,
                    "fitting": artifact.fitting,
                    **(
                        {"component_profile": artifact.profile_file}
                        if artifact.profile_file is not None
                        else {}
                    ),
                }
                for layer, artifact in sorted(self._layers.items())
            },
        }

    @staticmethod
    def _validate_final_dimension(values: Any, expected: int | None) -> None:
        shape = getattr(values, "shape", None)
        if shape is None or len(shape) < 2:
            raise ValueError("input must have at least two dimensions")
        if expected is not None and int(shape[-1]) != expected:
            raise ValueError(f"input final dimension is {shape[-1]}, expected {expected}")


def _component_fitting_statistics(artifact: LayerArtifact, component: int) -> dict[str, Any] | None:
    fitting = artifact.fitting
    if fitting.get("fun") != "logcosh":
        return None
    objectives = fitting.get("component_objectives")
    strengths = fitting.get("component_strengths")
    baseline = fitting.get("gaussian_objective")
    if (
        not isinstance(objectives, list)
        or not isinstance(strengths, list)
        or component >= len(objectives)
        or component >= len(strengths)
        or baseline is None
    ):
        return None
    order = sorted(range(len(strengths)), key=lambda index: (-float(strengths[index]), index))
    return {
        "contrast": "logcosh",
        "objective": float(objectives[component]),
        "gaussian_objective": float(baseline),
        "absolute_deviation": float(strengths[component]),
        "absolute_deviation_rank": order.index(component) + 1,
    }


def _validate_layer(layer: int) -> int:
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
        raise TypeError("layer must be a non-negative integer")
    value = int(layer)
    if value < 0:
        raise ValueError("layer must be a non-negative integer")
    return value


def _resolve_compatibility_argument(
    name: str, value: str | None, legacy_name: str, legacy_value: str | None
) -> str:
    if value is not None and legacy_value is not None:
        raise ValueError(f"pass {name}, not both {name} and {legacy_name}")
    resolved = value if value is not None else legacy_value
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return resolved.strip()


def _validate_model_type(value: str) -> Literal["base", "instruct"]:
    if value == "base":
        return "base"
    if value == "instruct":
        return "instruct"
    raise ValueError("model_type must be 'base' or 'instruct'")


def _mask_topk_scores(scores: Any, *, k: int, keep: bool) -> Any:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be a positive integer")
    count = int(k)
    shape = getattr(scores, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError("scores must have at least one dimension")
    components = int(shape[-1])
    if count <= 0 or count > components:
        raise ValueError(f"k must be between 1 and {components}, got {count}")

    if isinstance(scores, torch.Tensor):
        if not scores.is_floating_point():
            raise TypeError("scores must be a floating-point array")
        torch_indices = scores.abs().topk(count, dim=-1).indices
        if keep:
            result = torch.zeros_like(scores)
            return result.scatter(-1, torch_indices, scores.gather(-1, torch_indices))
        result = scores.clone()
        return result.scatter(
            -1, torch_indices, torch.zeros_like(torch_indices, dtype=scores.dtype)
        )

    array = np.asarray(scores)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("scores must be a floating-point array")
    numpy_indices = np.argsort(np.abs(array), axis=-1, kind="stable")[..., -count:]
    if keep:
        result_array = np.zeros_like(array)
        np.put_along_axis(
            result_array,
            numpy_indices,
            np.take_along_axis(array, numpy_indices, axis=-1),
            -1,
        )
        return result_array
    result_array = array.copy()
    np.put_along_axis(result_array, numpy_indices, 0, axis=-1)
    return result_array


def _validate_matching_arrays(values: Any, reference: Any) -> None:
    if values.ndim < 1 or reference.ndim < 1:
        raise ValueError("values and reference must have at least one dimension")
    if values.shape != reference.shape:
        raise ValueError(
            f"values and reference must have the same shape, got {values.shape} and "
            f"{reference.shape}"
        )
    values_are_floating = (
        np.issubdtype(values.dtype, np.floating)
        if isinstance(values, np.ndarray)
        else values.is_floating_point()
    )
    reference_is_floating = (
        np.issubdtype(reference.dtype, np.floating)
        if isinstance(reference, np.ndarray)
        else reference.is_floating_point()
    )
    if not values_are_floating:
        raise TypeError("values must have a floating-point dtype")
    if not reference_is_floating:
        raise TypeError("reference must have a floating-point dtype")


def _resolve_optional_compatibility_argument(
    name: str, value: str | None, legacy_name: str, legacy_value: str | None
) -> str | None:
    if value is not None and legacy_value is not None:
        raise ValueError(f"pass {name}, not both {name} and {legacy_name}")
    resolved = value if value is not None else legacy_value
    if resolved is None:
        return None
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(f"{name} must be a non-empty string or None")
    return resolved.strip()


def _array_offset(reference: Any, offset: NDArray[np.float32]) -> Any:
    if type(reference).__module__.split(".", maxsplit=1)[0] == "torch":
        import torch

        return torch.as_tensor(offset, dtype=reference.dtype, device=reference.device)
    return offset


def _as_fit_tensor(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach()
    else:
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError("activations must have a floating-point dtype")
        tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.ndim < 2:
        raise ValueError("activations must have at least two dimensions")
    if not tensor.is_floating_point():
        raise TypeError("activations must have a floating-point dtype")
    tensor = tensor.reshape(-1, tensor.shape[-1])
    # Preserve compact input storage (for example captured bfloat16 activations).
    # FastICA validates and converts one bounded batch at a time.
    return tensor


def _to_numpy(value: torch.Tensor) -> NDArray[np.float32]:
    return np.ascontiguousarray(value.detach().to(device="cpu", dtype=torch.float32).numpy())


def _validate_provenance(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("provenance must be a dictionary or None")
    try:
        encoded = json.dumps(value, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("provenance must contain only finite JSON-compatible values") from error
    if not isinstance(decoded, dict):
        raise ValueError("provenance must encode a JSON object")
    return decoded


def _package_version() -> str:
    try:
        return version("icalens")
    except PackageNotFoundError:
        return "unknown"
