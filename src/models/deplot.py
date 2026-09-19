"""Module B5: DePlot Model Harness Stub.

Provides a lightweight stub for Google's DePlot visual chart-to-table model.
"""


class DePlotHarness:
    """Minimal inference harness stub for DePlot model."""

    def __init__(self, model_name: str = "google/deplot") -> None:
        self.model_name = model_name

    def extract_table(self, image_path: str) -> str:
        """Extracts table string from input chart image path."""
        return "TITLE | Mock Title\nCol1 | Col2\n1 | 2"
