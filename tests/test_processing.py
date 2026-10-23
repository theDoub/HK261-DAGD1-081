from pathlib import Path

from src.models.harness import BaseExtractionBackend, VLMTableHarness
from src.pipeline import process_all_chart, process_chart

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "deplot_output.txt"
RAW_DEPLOT = FIXTURE_PATH.read_text(encoding="utf-8")


class FixedDePlotBackend(BaseExtractionBackend):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, image_path=None):
        self.calls += 1
        return RAW_DEPLOT


def make_harness(backend):
    return VLMTableHarness(backend=backend, target_format="deplot")


def test_process_chart_returns_requested_format():
    backend = FixedDePlotBackend()
    harness = make_harness(backend)

    result = process_chart("chart.png", "linearized", harness)

    assert result.startswith("<START> TITLE | Revenue vs Expenses\n")
    assert backend.calls == 1


def test_process_all_chart_extracts_once_and_returns_all_formats():
    backend = FixedDePlotBackend()
    harness = make_harness(backend)

    result = process_all_chart("chart.png", harness)

    assert set(result) == {"linearized", "markdown", "csv"}
    assert result["linearized"].startswith("<START> TITLE | Revenue vs Expenses\n")
    assert "### Revenue vs Expenses" in result["markdown"]
    assert result["csv"].startswith("product,revenue,expenses")
    assert backend.calls == 1

def main():
    test_process_chart_returns_requested_format()
    print("Passed")
    test_process_all_chart_extracts_once_and_returns_all_formats()
    print("Passed")


if __name__ == "__main__":
    main()