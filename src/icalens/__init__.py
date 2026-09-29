"""Public package interface for ICA Lens."""

from ._activation_dataset import ActivationDataset
from .analysis import AnalysisResult, CaptureResult
from .exceptions import ArtifactError, ICALensError, NotFittedError
from .lens import ICALens

__all__ = [
    "AnalysisResult",
    "ActivationDataset",
    "ArtifactError",
    "CaptureResult",
    "ICALens",
    "ICALensError",
    "NotFittedError",
]
__version__ = "0.4.0"
