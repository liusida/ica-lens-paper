"""User-facing exceptions raised by ICA Lens."""


class ICALensError(Exception):
    """Base exception for ICA Lens errors."""


class ArtifactError(ICALensError):
    """An artifact is missing, malformed, or incompatible."""


class NotFittedError(ICALensError):
    """The requested layer has not been fitted or loaded."""
