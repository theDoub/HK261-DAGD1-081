from __future__ import annotations

import json
from pathlib import Path

from scripts.inspect_errors import (
    build_error_inspection,
    format_inspection_markdown,
    main,
)


def _setup_sample_data(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "chartqa"
    tables_dir = data_dir / "tables"
    tables_dir.mkdir(parents=True)

    (tables_dir / "s1.csv").write_text("Country,GDP\nVietnam,400\n", encoding="utf-8")
    (tables_dir / "s2.csv").write_text("Year,Sales\n2020,10\n2021,20\n", encoding="utf-8")
    (tables_dir / "s3.csv").write_text("Item,Count\nA,5\n", encoding="utf-8")

    report = {
        "backend": "qwen",
        "records": [
            {
                "sample_id": "s1",
                "parse_ok": True,
                "rms_f1": 0.0,
                "rnss": 1.0,
                "value_recall_at_5": 1.0,
                "error_category": "header_mismatch",
                "raw_text": "| Entity | Value |\n|---|---|\n| Vietnam | 400 |",
                "error_message": None,
            },
            {
                "sample_id": "s2",
                "parse_ok": False,
                "rms_f1": 0.0,
                "rnss": 0.0,
                "value_recall_at_5": 0.0,
                "error_category": "missing_data",
                "raw_text": "<START> Year | Sales",
                "error_message": "Parsed table contained no rows or columns",
            },
            {
                "sample_id": "s3",
                "parse_ok": True,
                "rms_f1": 1.0,
                "rnss": 1.0,
                "value_recall_at_5": 1.0,
                "error_category": "none",
                "raw_text": "| Item | Count |\n|---|---|\n| A | 5 |",
                "error_message": None,
            },
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, data_dir


def test_build_error_inspection_computes_gold_vs_pred_diff(tmp_path: Path):
    report_path, data_dir = _setup_sample_data(tmp_path)

    cases = build_error_inspection(report_path, data_dir, category="header_mismatch", limit=10)

    assert len(cases) == 1
    c = cases[0]
    assert c["sample_id"] == "s1"
    assert c["gold_columns"] == ["country", "gdp"]
    assert c["pred_columns"] == ["entity", "value"]
    assert c["missing_columns"] == ["country", "gdp"]
    assert c["extra_columns"] == ["entity", "value"]
    assert c["gold_shape"] == [1, 2]
    assert c["pred_shape"] == [1, 2]
    assert "Vietnam,400" in c["gold_csv"]


def test_inspect_errors_cli_filters_all_and_writes_outputs(tmp_path: Path):
    report_path, data_dir = _setup_sample_data(tmp_path)
    out_md = tmp_path / "errors.md"
    out_json = tmp_path / "errors.json"

    exit_code = main(
        [
            "--report",
            str(report_path),
            "--data-dir",
            str(data_dir),
            "--category",
            "all",
            "--limit",
            "5",
            "--output-md",
            str(out_md),
            "--output-json",
            str(out_json),
        ]
    )

    assert exit_code == 0
    md_text = out_md.read_text(encoding="utf-8")
    assert "## 1. `s1` (header_mismatch)" in md_text
    assert "## 2. `s2` (missing_data)" in md_text
    assert "s3" not in md_text
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert "Missing in Pred" in format_inspection_markdown(saved)
