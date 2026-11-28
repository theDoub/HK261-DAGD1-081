from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import categorize_error
from src.format.parser import parse_csv_table, parse_table_text
from src.metrics.table_metrics import (
    compute_rms_f1,
    compute_rnss,
    compute_value_recall_at_5,
)


def re_evaluate_report(
    report_path: Path,
    tables_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    backend = str(report.get("backend") or "").lower()
    fmt = "deplot" if backend == "deplot" else "auto"

    new_records: list[dict[str, Any]] = []
    parse_ok_count = 0
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    rnsss: list[float] = []
    vr5s: list[float] = []
    error_dist: dict[str, int] = {}

    for rec in report.get("records", []):
        sample_id = str(rec["sample_id"])
        gt_path = tables_dir / f"{sample_id}.csv"
        gt_table = parse_csv_table(gt_path.read_text(encoding="utf-8"))

        raw_text = rec.get("raw_text") or ""
        pred_table = None
        parse_ok = False
        err_msg = rec.get("error_message")

        if raw_text.strip():
            try:
                pred_table = parse_table_text(raw_text, fmt=fmt)
                parse_ok = len(pred_table.columns) > 0 and len(pred_table.rows) > 0
            except Exception as exc:
                pred_table = None
                parse_ok = False
                err_msg = str(exc)

        if parse_ok and pred_table is not None:
            parse_ok_count += 1
            rms = compute_rms_f1(pred_table, gt_table)
            rnss = compute_rnss(pred_table, gt_table)
            vr5 = compute_value_recall_at_5(pred_table, gt_table)
        else:
            rms = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            rnss = 0.0
            vr5 = 0.0

        err_cat = categorize_error(pred_table if parse_ok else None, gt_table, rms["f1"])
        error_dist[err_cat] = error_dist.get(err_cat, 0) + 1

        precisions.append(rms["precision"])
        recalls.append(rms["recall"])
        f1s.append(rms["f1"])
        rnsss.append(rnss)
        vr5s.append(vr5)

        updated_rec = dict(rec)
        updated_rec["parse_ok"] = parse_ok
        updated_rec["precision"] = round(rms["precision"], 4)
        updated_rec["recall"] = round(rms["recall"], 4)
        updated_rec["rms_f1"] = round(rms["f1"], 4)
        updated_rec["rnss"] = round(rnss, 4)
        updated_rec["value_recall_at_5"] = round(vr5, 4)
        updated_rec["error_category"] = err_cat
        updated_rec["error_message"] = err_msg if not parse_ok else None
        new_records.append(updated_rec)

    total = len(new_records)
    updated_report = dict(report)
    updated_report["parse_success_count"] = parse_ok_count
    updated_report["parse_success_rate"] = round(100.0 * parse_ok_count / total, 2) if total else 0.0
    updated_report["mean_precision"] = round(sum(precisions) / total, 4) if total else 0.0
    updated_report["mean_recall"] = round(sum(recalls) / total, 4) if total else 0.0
    updated_report["mean_rms_f1"] = round(sum(f1s) / total, 4) if total else 0.0
    updated_report["mean_rnss"] = round(sum(rnsss) / total, 4) if total else 0.0
    updated_report["mean_value_recall_at_5"] = round(sum(vr5s) / total, 4) if total else 0.0
    updated_report["error_distribution"] = error_dist
    updated_report["records"] = new_records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(updated_report, indent=2), encoding="utf-8")
    return updated_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved Gate 2 JSON reports using updated parser and RMS-F1 metrics."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, default=Path("data/chartqa_test/tables"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    updated = re_evaluate_report(args.report, args.tables_dir, args.output)
    print(
        f"[{updated.get('backend', 'model')}] "
        f"parse_ok={updated['parse_success_rate']}% "
        f"RMS-F1={updated['mean_rms_f1']} "
        f"RNSS={updated['mean_rnss']} "
        f"VR@5%={updated['mean_value_recall_at_5']} "
        f"errors={updated['error_distribution']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
