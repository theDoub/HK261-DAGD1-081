from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.format.parser import parse_csv_table, parse_table_text


def inspect_case(
    record: Dict[str, Any],
    tables_dir: Path,
    target_format: str = "markdown",
) -> Dict[str, Any]:
    sample_id = str(record.get("sample_id", ""))
    gold_csv_path = tables_dir / f"{sample_id}.csv"
    gold_csv = gold_csv_path.read_text(encoding="utf-8").strip() if gold_csv_path.exists() else ""
    gold_table = parse_csv_table(gold_csv) if gold_csv else None

    raw_text = str(record.get("raw_text") or "")
    pred_table = None
    if raw_text.strip():
        try:
            pred_table = parse_table_text(raw_text, fmt=target_format)
        except Exception:
            pred_table = None

    gold_cols = list(gold_table.columns) if gold_table else []
    pred_cols = list(pred_table.columns) if pred_table else []
    gold_col_set = set(gold_cols)
    pred_col_set = set(pred_cols)

    return {
        "sample_id": sample_id,
        "error_category": record.get("error_category", "unknown"),
        "parse_ok": bool(record.get("parse_ok", False)),
        "rms_f1": float(record.get("rms_f1", 0.0)),
        "rnss": float(record.get("rnss", 0.0)),
        "value_recall_at_5": float(record.get("value_recall_at_5", 0.0)),
        "error_message": record.get("error_message"),
        "gold_shape": [len(gold_table.rows), len(gold_cols)] if gold_table else [0, 0],
        "pred_shape": [len(pred_table.rows), len(pred_cols)] if pred_table else [0, 0],
        "gold_columns": gold_cols,
        "pred_columns": pred_cols,
        "missing_columns": [c for c in gold_cols if c not in pred_col_set],
        "extra_columns": [c for c in pred_cols if c not in gold_col_set],
        "gold_csv": gold_csv,
        "raw_text": raw_text,
    }


def build_error_inspection(
    report_path: Path,
    data_dir: Path,
    category: str = "all",
    limit: Optional[int] = 10,
) -> List[Dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("records", [])
    backend = str(report.get("backend", "")).lower()
    target_format = "deplot" if backend == "deplot" else "markdown"
    tables_dir = data_dir / "tables"

    if category == "all":
        filtered = [r for r in records if r.get("error_category") not in (None, "none")]
    else:
        filtered = [r for r in records if r.get("error_category") == category]

    filtered.sort(key=lambda r: float(r.get("rms_f1", 0.0)))
    if limit is not None and limit > 0:
        filtered = filtered[:limit]

    return [inspect_case(r, tables_dir, target_format=target_format) for r in filtered]


def format_inspection_markdown(cases: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = ["# Error Inspection Report (Gold vs Raw)", ""]
    if not cases:
        lines.append("No matching error cases found.")
        return "\n".join(lines)

    for idx, c in enumerate(cases, start=1):
        lines.extend(
            [
                f"## {idx}. `{c['sample_id']}` ({c['error_category']})",
                f"- **parse_ok:** `{c['parse_ok']}` | **RMS-F1:** `{c['rms_f1']}` | **RNSS:** `{c['rnss']}` | **VR@5%:** `{c['value_recall_at_5']}`",
                f"- **error_message:** `{c['error_message']}`",
                f"- **Shape (rows x cols):** Gold `{c['gold_shape'][0]}x{c['gold_shape'][1]}` vs Pred `{c['pred_shape'][0]}x{c['pred_shape'][1]}`",
                f"- **Gold columns:** `{c['gold_columns']}`",
                f"- **Pred columns:** `{c['pred_columns']}`",
                f"- **Missing in Pred:** `{c['missing_columns']}`",
                f"- **Extra/Wrong in Pred:** `{c['extra_columns']}`",
                "",
                "**GOLD CSV:**",
                "```csv",
                c["gold_csv"] or "<MISSING GOLD CSV>",
                "```",
                "",
                "**RAW MODEL OUTPUT:**",
                "```text",
                c["raw_text"] or "<EMPTY>",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def print_inspection_report(cases: Sequence[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print(f" ERROR INSPECTION: GOLD VS RAW ({len(cases)} CASES)")
    print("=" * 80)
    if not cases:
        print(" No matching error cases found.")
        print("=" * 80 + "\n")
        return

    for idx, c in enumerate(cases, start=1):
        print(f"\n[{idx}] Sample: {c['sample_id']} | Category: {c['error_category']} | parse_ok={c['parse_ok']}")
        print(f"    Metrics       : RMS-F1={c['rms_f1']} | RNSS={c['rnss']} | VR@5%={c['value_recall_at_5']}")
        if c.get("error_message"):
            print(f"    Error Message : {c['error_message']}")
        print(f"    Shape (RxC)   : Gold={c['gold_shape'][0]}x{c['gold_shape'][1]} | Pred={c['pred_shape'][0]}x{c['pred_shape'][1]}")
        print(f"    Gold Columns  : {c['gold_columns']}")
        print(f"    Pred Columns  : {c['pred_columns']}")
        print(f"    Missing Cols  : {c['missing_columns']}")
        print(f"    Extra Cols    : {c['extra_columns']}")
        print("    --- GOLD CSV ---")
        print(c["gold_csv"] or "<MISSING GOLD CSV>")
        print("    --- RAW OUTPUT ---")
        print(c["raw_text"] or "<EMPTY>")
        print("-" * 80)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect error cases side-by-side (Gold CSV vs Raw model output)."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/gate2_qwen_report.json"),
        help="Path to evaluation report JSON.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/chartqa_test"),
        help="Path to ChartQA dataset directory containing tables/.",
    )
    parser.add_argument(
        "--category",
        default="all",
        choices=["all", "missing_data", "header_mismatch", "structural_error", "numeric_error"],
        help="Filter by error category.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of cases to inspect (0 for all).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional Markdown output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    args = parser.parse_args(argv)
    limit = args.limit if args.limit and args.limit > 0 else None

    cases = build_error_inspection(
        report_path=args.report,
        data_dir=args.data_dir,
        category=args.category,
        limit=limit,
    )
    print_inspection_report(cases)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(format_inspection_markdown(cases), encoding="utf-8")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
