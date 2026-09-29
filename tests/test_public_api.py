from __future__ import annotations

import icalens


def test_public_api_is_small() -> None:
    assert icalens.__all__ == [
        "AnalysisResult",
        "ActivationDataset",
        "ArtifactError",
        "CaptureResult",
        "ICALens",
        "ICALensError",
        "NotFittedError",
    ]
    assert icalens.__version__ == "0.4.0"
