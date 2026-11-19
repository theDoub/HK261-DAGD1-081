"""Unit test suite for Gate 2 Evaluation (Task 7.3: Real RMS-F1 on ChartQA)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_gate2 import (
    build_harness,
    evaluate_gate2,
    get_backend_task_metadata,
)
from src.data_loader.loader import ChartDataLoader


def test_build_harness_supported_backends():
    """Verifies that build_harness instantiates supported backends correctly."""
    mock_harness = build_harness("mock")
    assert mock_harness.target_format == "markdown"

    deplot_harness = build_harness("deplot")
    assert deplot_harness.target_format == "deplot"

    with pytest.raises(ValueError, match="Unsupported backend"):
        build_harness("unknown_backend")


def test_gate2_evaluation_runs_end_to_end_on_real_chartqa(tmp_path: Path):
    """Verifies that evaluate_gate2 scans data/chartqa_test, computes RMS-F1, and outputs valid summary."""
    project_root = Path(__file__).resolve().parent.parent
    chartqa_dir = project_root / "data" / "chartqa_test"

    if not chartqa_dir.exists():
        pytest.skip(f"ChartQA test directory not found at {chartqa_dir}")

    output_report = tmp_path / "gate2_test_report.json"

    # Evaluate first 3 samples with mock backend for fast regression checking
    summary = evaluate_gate2(
        data_dir=chartqa_dir,
        backend_name="mock",
        num_samples=3,
        output_path=output_report,
    )

    assert summary["week"] == 7
    assert summary["task_id"] == "7.2"
    assert summary["gate"] == "Pipeline End-to-End Verification (Mock)"
    assert summary["total_samples_evaluated"] == 3
    assert "mean_rms_f1" in summary
    assert "mean_rnss" in summary
    assert "mean_value_recall_at_5" in summary
    assert "error_distribution" in summary
    assert len(summary["records"]) == 3

    assert output_report.exists()
    with open(output_report, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_samples_evaluated"] == 3
    assert len(data["records"]) == 3


@pytest.mark.parametrize(
    ("backend_name", "expected_week", "expected_task"),
    [
        ("qwen", 7, "7.3"),
        ("gemini", 8, "8.2"),
        ("claude", 8, "8.2"),
        ("deplot", 12, "12.1"),
        ("mock", 7, "7.2"),
    ],
)
def test_backend_task_metadata_matches_weekly_plan(
    backend_name: str,
    expected_week: int,
    expected_task: str,
):
    meta = get_backend_task_metadata(backend_name)
    assert meta["week"] == expected_week
    assert meta["task_id"] == expected_task
