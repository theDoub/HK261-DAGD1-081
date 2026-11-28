from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_detailed_reports import (
    build_task12_2_report,
    build_task8_3_report,
    build_task9_2_report,
    generate_all_reports,
    load_chart_types,
)


def _sample_record(
    sample_id: str,
    rms_f1: float,
    rnss: float,
    vr5: float,
    error_category: str = "none",
    parse_ok: bool = True,
) -> dict:
    return {
        "sample_id": sample_id,
        "parse_ok": parse_ok,
        "precision": rms_f1,
        "recall": rms_f1,
        "rms_f1": rms_f1,
        "rnss": rnss,
        "value_recall_at_5": vr5,
        "error_category": error_category,
        "extraction_time_sec": 2.5,
    }


def test_load_chart_types_strips_png_suffix(tmp_path: Path) -> None:
    csv_path = tmp_path / "chart_types.csv"
    csv_path.write_text(
        '"image_name","chart_type","note"\n'
        '"001.png","h_bar",""\n'
        '"002.png","line",""\n'
        '"003.png","pie",""\n',
        encoding="utf-8",
    )
    mapping = load_chart_types(csv_path)
    assert mapping == {"001": "h_bar", "002": "line", "003": "pie"}


def test_task8_3_paired_and_chart_type_breakdown() -> None:
    qwen = {
        "records": [
            _sample_record("s1", 0.9, 0.95, 0.9, "none"),
            _sample_record("s2", 0.2, 0.80, 0.7, "header_mismatch"),
        ]
    }
    gemini = {
        "records": [
            _sample_record("s1", 0.5, 0.90, 0.8, "header_mismatch"),
            _sample_record("s2", 0.4, 0.80, 0.7, "numeric_error"),
        ]
    }
    chart_types = {"s1": "h_bar", "s2": "line"}
    report = build_task8_3_report(qwen, gemini, chart_types)
    assert report["qwen"]["mean_rms_f1"] == 0.55
    assert report["gemini"]["mean_rms_f1"] == 0.45
    assert report["paired_head_to_head"]["rms_f1"]["qwen_wins"] == 1
    assert report["paired_head_to_head"]["rms_f1"]["gemini_wins"] == 1
    assert report["by_chart_type"]["h_bar"]["delta_rms_f1_qwen_minus_gemini"] == 0.4


def test_task9_2_and_task12_2_reports_and_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "chart_types.csv"
    csv_path.write_text(
        '"image_name","chart_type","note"\n'
        '"s1.png","h_bar",""\n'
        '"s2.png","line",""\n'
        '"s3.png","pie",""\n',
        encoding="utf-8",
    )
    qwen_path = tmp_path / "qwen.json"
    gemini_path = tmp_path / "gemini.json"
    deplot_path = tmp_path / "deplot.json"

    qwen_path.write_text(
        json.dumps(
            {
                "records": [
                    _sample_record("s1", 0.8, 0.9, 0.8, "none"),
                    _sample_record("s2", 0.3, 0.8, 0.7, "header_mismatch"),
                    _sample_record("s3", 0.4, 0.85, 0.75, "structural_error"),
                ]
            }
        ),
        encoding="utf-8",
    )
    gemini_path.write_text(
        json.dumps(
            {
                "records": [
                    _sample_record("s1", 0.6, 0.85, 0.75, "header_mismatch"),
                    _sample_record("s2", 0.4, 0.82, 0.70, "numeric_error"),
                    _sample_record("s3", 0.2, 0.70, 0.60, "header_mismatch"),
                ]
            }
        ),
        encoding="utf-8",
    )
    deplot_path.write_text(
        json.dumps(
            {
                "records": [
                    _sample_record("s1", 0.95, 0.98, 0.95, "none"),
                    _sample_record("s2", 0.80, 0.92, 0.85, "none"),
                    _sample_record("s3", 0.75, 0.90, 0.80, "structural_error"),
                ]
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "outputs"
    result = generate_all_reports(
        qwen_path=qwen_path,
        gemini_path=gemini_path,
        deplot_path=deplot_path,
        chart_types_path=csv_path,
        output_dir=out_dir,
    )
    assert "task8_3_qwen_vs_gemini.md" in result["written_files"]
    assert "task9_2_chart_type_breakdown.md" in result["written_files"]
    assert "task12_2_deplot_vs_vlms.md" in result["written_files"]
    assert (out_dir / "task8_3_qwen_vs_gemini.md").exists()
    assert (out_dir / "task9_2_chart_type_breakdown.md").exists()
    assert (out_dir / "task12_2_deplot_vs_vlms.md").exists()
    assert (out_dir / "detailed_reports_summary.md").exists()

    t12 = build_task12_2_report(
        json.loads(deplot_path.read_text(encoding="utf-8")),
        {"qwen": json.loads(qwen_path.read_text(encoding="utf-8"))},
        load_chart_types(csv_path),
    )
    assert t12["comparisons"]["qwen"]["paired_head_to_head"]["rms_f1"]["deplot_wins"] == 3
    t9 = build_task9_2_report(
        {"deplot": json.loads(deplot_path.read_text(encoding="utf-8"))},
        load_chart_types(csv_path),
    )
    assert t9["models"]["deplot"]["pie"]["mean_rms_f1"] == 0.75
