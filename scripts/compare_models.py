from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_report_summary(report_path: Path) -> Dict[str, Any]:
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    backend = data.get("backend") or report_path.stem
    return {
        "report_file": str(report_path),
        "backend": str(backend),
        "week": data.get("week"),
        "task_id": data.get("task_id"),
        "samples": int(data.get("total_samples_evaluated", len(data.get("records", [])))),
        "parse_success_rate": float(data.get("parse_success_rate", 0.0)),
        "mean_precision": float(data.get("mean_precision", 0.0)),
        "mean_recall": float(data.get("mean_recall", 0.0)),
        "mean_rms_f1": float(data.get("mean_rms_f1", 0.0)),
        "mean_rnss": float(data.get("mean_rnss", 0.0)),
        "mean_value_recall_at_5": float(data.get("mean_value_recall_at_5", 0.0)),
        "total_elapsed_seconds": float(data.get("total_elapsed_seconds", 0.0)),
        "error_distribution": data.get("error_distribution", {}),
    }


def build_comparison_rows(report_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows = [load_report_summary(p) for p in report_paths if p.exists()]
    rows.sort(key=lambda r: r["mean_rms_f1"], reverse=True)
    return rows


def format_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Backend | Week / Task | Samples | Parse OK (%) | Precision | Recall | RMS-F1 | RNSS | Value-Recall@5% | Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        week_task = f"W{r['week']} / {r['task_id']}" if r.get("week") and r.get("task_id") else "-"
        lines.append(
            f"| {r['backend']} | {week_task} | {r['samples']} | "
            f"{r['parse_success_rate']:.2f}% | {r['mean_precision']:.4f} | "
            f"{r['mean_recall']:.4f} | {r['mean_rms_f1']:.4f} | "
            f"{r['mean_rnss']:.4f} | {r['mean_value_recall_at_5']:.4f} | "
            f"{r['total_elapsed_seconds']:.2f} |"
        )
    return "\n".join(lines)


def write_comparison_outputs(
    rows: Sequence[Dict[str, Any]],
    output_md: Optional[Path] = None,
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> None:
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        md_content = "# Multi-Model Chart-to-Table Comparison (Task 8.3)\n\n" + format_markdown_table(rows) + "\n"
        output_md.write_text(md_content, encoding="utf-8")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"task": "8.3", "models": list(rows)}, f, indent=2)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "backend",
            "week",
            "task_id",
            "samples",
            "parse_success_rate",
            "mean_precision",
            "mean_recall",
            "mean_rms_f1",
            "mean_rnss",
            "mean_value_recall_at_5",
            "total_elapsed_seconds",
        ]
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in fieldnames})


def print_comparison_table(rows: Sequence[Dict[str, Any]]) -> None:
    print("\n" + "=" * 98)
    print(" WEEK 8 - TASK 8.3: MULTI-MODEL COMPARISON TABLE")
    print("=" * 98)
    if not rows:
        print(" No valid report files found.")
        print("=" * 98 + "\n")
        return

    header = (
        f" {'Backend':<10} {'Task':<9} {'Samples':>7} {'Parse%':>8} "
        f"{'Prec':>8} {'Recall':>8} {'RMS-F1':>8} {'RNSS':>8} {'VR@5%':>8} {'Time(s)':>9}"
    )
    print(header)
    print("-" * 98)
    for r in rows:
        task_str = f"W{r['week']}/{r['task_id']}" if r.get("week") and r.get("task_id") else "-"
        print(
            f" {r['backend']:<10} {task_str:<9} {r['samples']:>7} {r['parse_success_rate']:>7.2f}% "
            f"{r['mean_precision']:>8.4f} {r['mean_recall']:>8.4f} {r['mean_rms_f1']:>8.4f} "
            f"{r['mean_rnss']:>8.4f} {r['mean_value_recall_at_5']:>8.4f} {r['total_elapsed_seconds']:>9.2f}"
        )
    print("=" * 98 + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare multiple Chart-to-Table model evaluation reports (Week 8 Task 8.3)."
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        type=Path,
        default=None,
        help="List of evaluation report JSON files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to scan for *_report.json if --reports is omitted.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/model_comparison_table.md"),
        help="Output Markdown comparison table path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/model_comparison_table.json"),
        help="Output JSON comparison report path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/model_comparison_table.csv"),
        help="Output CSV comparison table path.",
    )

    args = parser.parse_args(argv)

    if args.reports:
        report_paths = args.reports
    else:
        report_paths = sorted(args.reports_dir.glob("*_report.json")) if args.reports_dir.exists() else []

    rows = build_comparison_rows(report_paths)
    print_comparison_table(rows)
    if rows:
        write_comparison_outputs(
            rows,
            output_md=args.output_md,
            output_json=args.output_json,
            output_csv=args.output_csv,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
