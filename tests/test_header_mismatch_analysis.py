import json
from pathlib import Path

from scripts.analyze_header_mismatches import analyze_mismatches


def _write_table(data_dir: Path, sample_id: str, csv_text: str) -> None:
    (data_dir / "tables").mkdir(parents=True, exist_ok=True)
    (data_dir / "tables" / f"{sample_id}.csv").write_text(csv_text, encoding="utf-8")


def test_analyzer_reports_transpose_label_and_unit_cases_for_all_mismatches(tmp_path: Path):
    data_dir = tmp_path / "dataset"
    _write_table(data_dir, "transpose", "entity,2005,2006\nFrance,10,11\nGermany,12,13\n")
    _write_table(data_dir, "unit", "Year,Dissatisfied (%),Satisfied (%)\n2005,40,60\n")
    _write_table(data_dir, "label", "entity,disapprove,approve\nGroup A,30,70\n")

    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sample_id": "transpose",
                        "error_category": "header_mismatch",
                        "raw_text": "| Year | France | Germany |\n|---|---|---|\n| 2005 | 10 | 12 |\n| 2006 | 11 | 13 |",
                    },
                    {
                        "sample_id": "unit",
                        "error_category": "header_mismatch",
                        "raw_text": "| Year | Dissatisfied | Satisfied |\n|---|---|---|\n| 2005 | 40 | 60 |",
                    },
                    {
                        "sample_id": "label",
                        "error_category": "header_mismatch",
                        "raw_text": "| category | disapprove | approve |\n|---|---|---|\n| Group A | 30 | 70 |",
                    },
                    {
                        "sample_id": "no-gold",
                        "error_category": "header_mismatch",
                        "raw_text": "| Category | Value |\n|---|---|\n| A | 1 |",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_mismatches(report_path, data_dir)

    assert analysis["total_mismatches"] == 4
    assert analysis["analyzed_mismatches"] == 3
    assert analysis["unavailable_sample_ids"] == ["no-gold"]
    by_id = {result["sample_id"]: result for result in analysis["results"]}
    assert by_id["transpose"]["transpose_case"] is True
    assert by_id["unit"]["unit_or_title_case"] is True
    assert by_id["label"]["header_label_differences"] == ["category"]
