"""Module B5: Vision-Language Model Extraction Harness and Backends."""

from src.models.harness import (
    BaseExtractionBackend,
    ClaudeVLMBackend,
    DePlotBackend,
    ExtractionResult,
    GeminiVLMBackend,
    MockBackend,
    OpenVLMBackend,
    VLMTableHarness,
)

__all__ = [
    "DePlotHarness",
    "VLMTableHarness",
    "BaseExtractionBackend",
    "DePlotBackend",
    "MockBackend",
    "OpenVLMBackend",
    "ClaudeVLMBackend",
    "GeminiVLMBackend",
    "ExtractionResult",
]


def __getattr__(name: str):
    if name == "DePlotHarness":
        from src.models.deplot import DePlotHarness

        return DePlotHarness
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
