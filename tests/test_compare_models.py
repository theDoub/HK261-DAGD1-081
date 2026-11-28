from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_models import (
    build_comparison_rows,
    format_markdown_table,
    main,
    write_comparison_outputs,
)


def _write_report(path: Path, backend: str, week: int, task_id: str, f1: float) -> Path:
    payload = {
        "backend": backend,
        "week": week,
        "task_id": task_id,
        "total_samples_evaluated": 300,
        "parse_success_rate": 98.5,
        "mean_precision": f1,
        "mean_recall": f1,
        "mean_rms_f1": f1,
        "mean_rnss": f1 - 0.05,
        "mean_value_recall_at_5": f1,
        "total_elapsed_seconds": 120.5,
        "error_distribution": {"none": 250, "numeric_error": 50},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_comparison_rows_sorts_by_rms_f1_descending(tmp_path: Path):
    qwen_path = _write_report(tmp_path / "qwen_report.json", "qwen", 7, "7.3", 0.72)
    gemini_path = _write_report(tmp_path / "gemini_report.json", "gemini", 8, "8.2", 0.85)

    rows = build_comparison_rows([qwen_path, gemini_path])

    assert len(rows) == 2
    assert rows[0]["backend"] == "gemini"
    assert rows[0]["week"] == 8
    assert rows[0]["task_id"] == "8.2"
    assert rows[1]["backend"] == "qwen"
    assert rows[1]["week"] == 7
    assert rows[1]["task_id"] == "7.3"


def test_write_comparison_outputs_creates_md_json_csv(tmp_path: Path):
    r1 = _write_report(tmp_path / "qwen_report.json", "qwen", 7, "7.3", 0.72)
    r2 = _write_report(tmp_path / "gemini_report.json", "gemini", 8, "8.2", 0.85)
    rows = build_comparison_rows([r1, r2])

    md_path = tmp_path / "out" / "compare.md"
    json_path = tmp_path / "out" / "compare.json"
    csv_path = tmp_path / "out" / "compare.csv"

    write_comparison_outputs(rows, output_md=md_path, output_json=json_path, output_csv=csv_path)

    md_text = md_path.read_text(encoding="utf-8")
    assert "| gemini | W8 / 8.2 |" in md_text
    assert "| qwen | W7 / 7.3 |" in md_text
    assert "0.8500" in format_markdown_table(rows)
    assert json.loads(json_path.read_text(encoding="utf-8"))["task"] == "8.3"
    assert "gemini,8,8.2,300" in csv_path.read_text(encoding="utf-8")


def test_compare_models_cli_autodiscovers_reports(tmp_path: Path):
    reports_dir = tmp_path / "outputs"
    reports_dir.mkdir()
    _write_report(reports_dir / "qwen_report.json", "qwen", 7, "7.3", 0.70)
    _write_report(reports_dir / "deplot_report.json", "deplot", 12, "12.1", 0.65)

    exit_code = main(
        [
            "--reports-dir",
            str(reports_dir),
            "--output-md",
            str(reports_dir / "table.md"),
            "--output-json",
            str(reports_dir / "table.json"),
            "--output-csv",
            str(reports_dir / "table.csv"),
        ]
    )

    assert exit_code == 0
    assert (reports_dir / "table.md").exists()
    assert (reports_dir / "table.json").exists()
    assert (reports_dir / "table.csv").exists()
