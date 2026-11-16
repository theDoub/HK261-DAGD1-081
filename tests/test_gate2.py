"""Unit test suite for Gate 2 Evaluation (Task 7.3: Real RMS-F1 on ChartQA)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_gate2 import build_harness, evaluate_gate2
from src.models.harness import MockBackend, VLMTableHarness
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

    assert summary["gate"] == "Gate 2 (First Real RMS-F1 on ChartQA)"
    assert summary["total_samples_evaluated"] == 3
    assert "mean_rms_f1" in summary
    assert "mean_rnss" in summary
    assert "mean_value_recall_at_5" in summary
    assert "error_distribution" in summary
    assert len(summary["records"]) == 3

    # Check generated JSON file
    assert output_report.exists()
    with open(output_report, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_samples_evaluated"] == 3
    assert len(data["records"]) == 3


def test_gate2_prompt_override_and_sample_order_are_recorded(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "chartqa"
    images_dir = data_dir / "images"
    tables_dir = data_dir / "tables"
    images_dir.mkdir(parents=True)
    tables_dir.mkdir()
    for sample_id in ("first", "second"):
        (images_dir / f"{sample_id}.png").write_bytes(b"mock image")
        (tables_dir / f"{sample_id}.csv").write_text("Quarter,Sales\nQ1,120\n", encoding="utf-8")

    prompts = []

    class PromptRecordingBackend(MockBackend):
        def generate(self, prompt, image_path=None):
            prompts.append(prompt)
            return super().generate(prompt, image_path)

    monkeypatch.setattr(
        "scripts.evaluate_gate2.build_harness",
        lambda *args, **kwargs: VLMTableHarness(backend=PromptRecordingBackend()),
    )
    report_path = tmp_path / "evaluation.json"
    prompt = "Use the baseline prompt exactly as supplied."

    summary = evaluate_gate2(
        data_dir=data_dir,
        backend_name="mock",
        output_path=report_path,
        prompt_override=prompt,
        sample_ids=["second", "first"],
    )

    assert prompts == [prompt, prompt]
    assert summary["conditions"]["prompt"] == prompt
    assert summary["conditions"]["prompt_source"] == "override"
    assert summary["conditions"]["sample_ids"] == ["second", "first"]
    assert [record["sample_id"] for record in summary["records"]] == ["second", "first"]
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["conditions"] == summary["conditions"]


def test_gate2_rejects_missing_requested_sample_ids(tmp_path: Path):
    data_dir = tmp_path / "chartqa"
    (data_dir / "images").mkdir(parents=True)
    (data_dir / "tables").mkdir()

    with pytest.raises(ValueError, match="missing from the dataset"):
        evaluate_gate2(data_dir=data_dir, backend_name="mock", sample_ids=["missing"])
