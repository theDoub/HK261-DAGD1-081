"""Unit tests for the Chart-to-Table -> Table QA pipeline components."""

import json
from pathlib import Path
import pytest

from src.data_loader.loader import ChartDataLoader, ChartDatasetItem
from src.evaluation.evaluator import PipelineEvaluator
from src.format.schema import TableSchema
from src.metrics.table_metrics import (
    compute_qa_accuracy,
    compute_relaxed_accuracy,
    compute_rms_f1,
    compute_rnss,
)
from src.models.deplot import DePlotHarness
from src.models.vlm_client import VLMClient


@pytest.fixture
def sample_table_data():
    return {
        "title": "Quarterly Revenue",
        "columns": ["Region", "Q1", "Q2"],
        "rows": [
            ["North America", 100.0, 110.0],
            ["Europe", 80.0, 85.0],
        ],
        "metadata": {"unit": "Million USD"},
    }


def test_table_schema_creation(sample_table_data):
    """Verifies TableSchema initialization, shape validation, and dictionary round-trip."""
    table = TableSchema.from_dict(sample_table_data)
    assert table.title == "Quarterly Revenue"
    assert len(table.columns) == 3
    assert len(table.rows) == 2
    assert table.validate_shape() is True

    # Test serialization
    data_dict = table.to_dict()
    assert data_dict["title"] == "Quarterly Revenue"
    assert data_dict["columns"] == ["Region", "Q1", "Q2"]


def test_table_schema_markdown_and_dataframe(sample_table_data):
    """Verifies rendering to Markdown and conversion to pandas DataFrame."""
    table = TableSchema.from_dict(sample_table_data)
    md = table.to_markdown()
    assert "### Quarterly Revenue" in md
    assert "| Region | Q1 | Q2 |" in md

    df = table.to_dataframe()
    assert df.shape == (2, 3)
    assert list(df.columns) == ["Region", "Q1", "Q2"]


def test_table_schema_linearization():
    """Verifies linearized text serialization and deserialization."""
    table = TableSchema(
        title="Test Plot",
        columns=["Year", "Score"],
        rows=[[2021, 95.5], [2022, 98.0]],
    )
    linearized = table.to_linearized_text()
    assert "TITLE | Test Plot" in linearized
    assert "Year | Score" in linearized

    reconstructed = TableSchema.from_linearized_text(linearized)
    assert reconstructed.title == "Test Plot"
    assert reconstructed.columns == ["Year", "Score"]
    assert len(reconstructed.rows) == 2
    assert reconstructed.rows[0][1] == 95.5


def test_metrics():
    """Verifies RMS-F1, RNSS, and QA accuracy calculations."""
    t1 = TableSchema(columns=["A", "B"], rows=[["x", 100.0], ["y", 200.0]])
    t2 = TableSchema(columns=["A", "B"], rows=[["x", 102.0], ["y", 199.0]])

    # 102 vs 100 is within 5% tolerance, 199 vs 200 is within 5%
    metrics = compute_rms_f1(t1, t2, tolerance=0.05)
    assert metrics["f1"] == 1.0

    rnss = compute_rnss(t1, t2)
    assert rnss == 1.0

    # Test QA accuracy
    assert compute_qa_accuracy("North America", "north america") == 1.0
    assert compute_qa_accuracy("North America", "Europe") == 0.0

    # Test relaxed accuracy with 5% threshold
    assert compute_relaxed_accuracy("104.5", "100.0", tolerance=0.05) == 1.0
    assert compute_relaxed_accuracy("110.0", "100.0", tolerance=0.05) == 0.0


def test_data_loader_sanity():
    """Verifies loading mock sanity dataset."""
    loader = ChartDataLoader(data_dir="data")
    samples = loader.load_sanity_data()
    assert len(samples) > 0
    assert samples[0].table is not None
    assert samples[0].table.validate_shape() is True


def test_end_to_end_pipeline_evaluation():
    """Verifies the two-stage evaluation pipeline execution on a mock sample."""
    harness = DePlotHarness(dry_run=True)
    vlm = VLMClient(provider="vlm_stub")
    evaluator = PipelineEvaluator(deplot_harness=harness, vlm_client=vlm)

    loader = ChartDataLoader(data_dir="data")
    samples = loader.load_sanity_data()
    summary = evaluator.evaluate_dataset(samples)

    assert summary["total_samples"] == len(samples)
    assert "mean_exact_match" in summary
    assert "mean_relaxed_accuracy" in summary
