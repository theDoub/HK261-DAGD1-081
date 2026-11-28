"""Unit test suite for Gate 2 Evaluation (Task 7.3: Real RMS-F1 on ChartQA)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from scripts.evaluate_gate2 import (
    build_harness,
    evaluate_gate2,
    get_backend_task_metadata,
)
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
    images_dir = data_dir / "images"
    tables_dir = data_dir / "tables"
    images_dir.mkdir(parents=True)
    tables_dir.mkdir()
    (images_dir / "available.png").write_bytes(b"mock image")
    (tables_dir / "available.csv").write_text("Entity,Value\nA,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing from the dataset"):
        evaluate_gate2(data_dir=data_dir, backend_name="mock", sample_ids=["missing"])


@pytest.mark.parametrize(("override", "expected"), [(None, 512), (37, 37)])
def test_qwen_token_limit_uses_config_and_allows_override(
    tmp_path: Path,
    monkeypatch,
    override: Optional[int],
    expected: int,
):
    data_dir = tmp_path / "chartqa"
    images_dir = data_dir / "images"
    tables_dir = data_dir / "tables"
    images_dir.mkdir(parents=True)
    tables_dir.mkdir()
    (images_dir / "sample.png").write_bytes(b"mock image")
    (tables_dir / "sample.csv").write_text("Entity,Value\nA,1\n", encoding="utf-8")

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        'qwen:\n  model_name_or_path: "test-model"\n  max_new_tokens: 512\n  prompt: "test prompt"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    harness_args = {}

    def fake_build_harness(*args, **kwargs):
        harness_args.update(kwargs)
        return VLMTableHarness(backend=MockBackend(predefined_table_md="| Entity | Value |\n|---|---|\n| A | 1 |"))

    monkeypatch.setattr("scripts.evaluate_gate2.build_harness", fake_build_harness)
    summary = evaluate_gate2(
        data_dir=data_dir,
        backend_name="qwen",
        max_new_tokens=override,
    )

    assert harness_args["max_new_tokens"] == expected
    assert summary["conditions"]["max_new_tokens"] == expected
