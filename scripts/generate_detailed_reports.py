from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ERROR_BUCKETS = (
    "none",
    "header_mismatch",
    "structural_error",
    "numeric_error",
    "missing_data",
)


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_chart_types(csv_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = (row.get("image_name") or "").strip()
            chart_type = (row.get("chart_type") or "").strip()
            if not image_name or not chart_type:
                continue
            sample_id = Path(image_name).stem
            mapping[sample_id] = chart_type
    return mapping


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _compute_group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {
            "count": 0,
            "parse_ok_count": 0,
            "parse_ok_rate": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_rms_f1": 0.0,
            "mean_rnss": 0.0,
            "mean_value_recall_at_5": 0.0,
            "mean_time_sec": 0.0,
            "error_distribution": {bucket: 0 for bucket in ERROR_BUCKETS},
        }
    parse_ok_count = sum(1 for r in records if r.get("parse_ok"))
    error_dist = {bucket: 0 for bucket in ERROR_BUCKETS}
    for r in records:
        cat = str(r.get("error_category") or "missing_data")
        error_dist[cat] = error_dist.get(cat, 0) + 1
    return {
        "count": count,
        "parse_ok_count": parse_ok_count,
        "parse_ok_rate": round(100.0 * parse_ok_count / count, 2),
        "mean_precision": round(_mean([float(r.get("precision", 0.0)) for r in records]), 4),
        "mean_recall": round(_mean([float(r.get("recall", 0.0)) for r in records]), 4),
        "mean_rms_f1": round(_mean([float(r.get("rms_f1", 0.0)) for r in records]), 4),
        "mean_rnss": round(_mean([float(r.get("rnss", 0.0)) for r in records]), 4),
        "mean_value_recall_at_5": round(
            _mean([float(r.get("value_recall_at_5", 0.0)) for r in records]), 4
        ),
        "mean_time_sec": round(
            _mean([float(r.get("extraction_time_sec", 0.0)) for r in records]), 2
        ),
        "error_distribution": error_dist,
    }


def _paired_comparison(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    label_a: str,
    label_b: str,
    tie_eps: float = 1e-6,
) -> dict[str, Any]:
    map_a = {str(r["sample_id"]): r for r in records_a if "sample_id" in r}
    map_b = {str(r["sample_id"]): r for r in records_b if "sample_id" in r}
    common_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
    a_wins_f1 = 0
    b_wins_f1 = 0
    ties_f1 = 0
    a_wins_rnss = 0
    b_wins_rnss = 0
    ties_rnss = 0
    deltas_f1: list[float] = []
    deltas_rnss: list[float] = []
    for sid in common_ids:
        fa = float(map_a[sid].get("rms_f1", 0.0))
        fb = float(map_b[sid].get("rms_f1", 0.0))
        diff_f1 = fa - fb
        deltas_f1.append(diff_f1)
        if abs(diff_f1) <= tie_eps:
            ties_f1 += 1
        elif diff_f1 > 0:
            a_wins_f1 += 1
        else:
            b_wins_f1 += 1
        ra = float(map_a[sid].get("rnss", 0.0))
        rb = float(map_b[sid].get("rnss", 0.0))
        diff_rnss = ra - rb
        deltas_rnss.append(diff_rnss)
        if abs(diff_rnss) <= tie_eps:
            ties_rnss += 1
        elif diff_rnss > 0:
            a_wins_rnss += 1
        else:
            b_wins_rnss += 1
    return {
        "common_samples": len(common_ids),
        "label_a": label_a,
        "label_b": label_b,
        "rms_f1": {
            f"{label_a}_wins": a_wins_f1,
            f"{label_b}_wins": b_wins_f1,
            "ties": ties_f1,
            "mean_delta_a_minus_b": round(_mean(deltas_f1), 4),
        },
        "rnss": {
            f"{label_a}_wins": a_wins_rnss,
            f"{label_b}_wins": b_wins_rnss,
            "ties": ties_rnss,
            "mean_delta_a_minus_b": round(_mean(deltas_rnss), 4),
        },
    }


def build_task8_3_report(
    qwen_report: dict[str, Any],
    gemini_report: dict[str, Any],
    chart_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    qwen_records = list(qwen_report.get("records", []))
    gemini_records = list(gemini_report.get("records", []))
    qwen_metrics = _compute_group_metrics(qwen_records)
    gemini_metrics = _compute_group_metrics(gemini_records)
    paired = _paired_comparison(qwen_records, gemini_records, "qwen", "gemini")
    by_chart_type: dict[str, Any] = {}
    if chart_types:
        types = sorted(
            {
                chart_types[str(r["sample_id"])]
                for r in qwen_records + gemini_records
                if str(r.get("sample_id")) in chart_types
            }
        )
        for ctype in types:
            q_sub = [
                r
                for r in qwen_records
                if chart_types.get(str(r.get("sample_id"))) == ctype
            ]
            g_sub = [
                r
                for r in gemini_records
                if chart_types.get(str(r.get("sample_id"))) == ctype
            ]
            q_m = _compute_group_metrics(q_sub)
            g_m = _compute_group_metrics(g_sub)
            by_chart_type[ctype] = {
                "qwen": q_m,
                "gemini": g_m,
                "delta_rms_f1_qwen_minus_gemini": round(
                    q_m["mean_rms_f1"] - g_m["mean_rms_f1"], 4
                ),
            }
    return {
        "task": "8.3",
        "title": "Task 8.3: Qwen2.5-VL vs Gemini Detailed Comparison",
        "qwen": qwen_metrics,
        "gemini": gemini_metrics,
        "paired_head_to_head": paired,
        "by_chart_type": by_chart_type,
    }


def format_task8_3_markdown(data: dict[str, Any]) -> str:
    q = data["qwen"]
    g = data["gemini"]
    p = data["paired_head_to_head"]
    lines = [
        "# Task 8.3: Qwen2.5-VL vs Gemini Detailed Comparison",
        "",
        "## 1. Overall Metrics Comparison",
        "",
        "| Metric | Qwen2.5-VL-7B (4-bit) | Gemini | Delta (Qwen - Gemini) |",
        "|---|---:|---:|---:|",
        f"| Samples | {q['count']} | {g['count']} | {q['count'] - g['count']:+d} |",
        f"| Parse Success Rate | {q['parse_ok_rate']:.2f}% ({q['parse_ok_count']}/{q['count']}) | {g['parse_ok_rate']:.2f}% ({g['parse_ok_count']}/{g['count']}) | {q['parse_ok_rate'] - g['parse_ok_rate']:+.2f}% |",
        f"| Mean Precision | {q['mean_precision']:.4f} | {g['mean_precision']:.4f} | {q['mean_precision'] - g['mean_precision']:+.4f} |",
        f"| Mean Recall | {q['mean_recall']:.4f} | {g['mean_recall']:.4f} | {q['mean_recall'] - g['mean_recall']:+.4f} |",
        f"| Mean RMS-F1 | {q['mean_rms_f1']:.4f} | {g['mean_rms_f1']:.4f} | {q['mean_rms_f1'] - g['mean_rms_f1']:+.4f} |",
        f"| Mean RNSS | {q['mean_rnss']:.4f} | {g['mean_rnss']:.4f} | {q['mean_rnss'] - g['mean_rnss']:+.4f} |",
        f"| Mean Value-Recall@5% | {q['mean_value_recall_at_5']:.4f} | {g['mean_value_recall_at_5']:.4f} | {q['mean_value_recall_at_5'] - g['mean_value_recall_at_5']:+.4f} |",
        f"| Mean Latency / Sample (s) | {q['mean_time_sec']:.2f} | {g['mean_time_sec']:.2f} | {q['mean_time_sec'] - g['mean_time_sec']:+.2f} |",
        "",
        "## 2. Error Category Distribution",
        "",
        "| Error Category | Qwen Count (%) | Gemini Count (%) | Diff (Qwen - Gemini) |",
        "|---|---:|---:|---:|",
    ]
    for bucket in ERROR_BUCKETS:
        qc = q["error_distribution"].get(bucket, 0)
        gc = g["error_distribution"].get(bucket, 0)
        qp = (100.0 * qc / q["count"]) if q["count"] else 0.0
        gp = (100.0 * gc / g["count"]) if g["count"] else 0.0
        lines.append(
            f"| `{bucket}` | {qc} ({qp:.1f}%) | {gc} ({gp:.1f}%) | {qc - gc:+d} |"
        )
    f1_p = p["rms_f1"]
    rnss_p = p["rnss"]
    lines.extend(
        [
            "",
            f"## 3. Paired Head-to-Head ({p['common_samples']} common samples)",
            "",
            "| Metric | Qwen Wins | Gemini Wins | Ties | Mean Delta (Qwen - Gemini) |",
            "|---|---:|---:|---:|---:|",
            f"| RMS-F1 | {f1_p['qwen_wins']} | {f1_p['gemini_wins']} | {f1_p['ties']} | {f1_p['mean_delta_a_minus_b']:+.4f} |",
            f"| RNSS | {rnss_p['qwen_wins']} | {rnss_p['gemini_wins']} | {rnss_p['ties']} | {rnss_p['mean_delta_a_minus_b']:+.4f} |",
        ]
    )
    if data.get("by_chart_type"):
        lines.extend(
            [
                "",
                "## 4. Breakdown by Chart Type (Qwen vs Gemini)",
                "",
                "| Chart Type | Samples | Qwen RMS-F1 | Gemini RMS-F1 | Delta (Qwen - Gemini) | Qwen RNSS | Gemini RNSS |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for ctype, entry in data["by_chart_type"].items():
            qm = entry["qwen"]
            gm = entry["gemini"]
            lines.append(
                f"| `{ctype}` | {qm['count']} | {qm['mean_rms_f1']:.4f} | {gm['mean_rms_f1']:.4f} | {entry['delta_rms_f1_qwen_minus_gemini']:+.4f} | {qm['mean_rnss']:.4f} | {gm['mean_rnss']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def build_task9_2_report(
    models: dict[str, dict[str, Any]],
    chart_types: dict[str, str],
) -> dict[str, Any]:
    all_types = sorted(
        {
            chart_types.get(str(r.get("sample_id")), "unknown")
            for report in models.values()
            for r in report.get("records", [])
        }
    )
    breakdown: dict[str, dict[str, Any]] = {}
    for model_name, report in models.items():
        records = list(report.get("records", []))
        per_type: dict[str, Any] = {}
        for ctype in all_types:
            sub = [
                r
                for r in records
                if chart_types.get(str(r.get("sample_id")), "unknown") == ctype
            ]
            per_type[ctype] = _compute_group_metrics(sub)
        breakdown[model_name] = per_type
    return {
        "task": "9.2",
        "title": "Task 9.2: Chart-Type Breakdown Across Models",
        "chart_types": all_types,
        "models": breakdown,
    }


def format_task9_2_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Task 9.2: Full Chart-Type Breakdown",
        "",
        "| Model | Chart Type | Samples | Parse% | Precision | Recall | Mean RMS-F1 | Mean RNSS | Mean VR@5% | `none` | `header_mismatch` | `structural_error` | `numeric_error` |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, per_type in data["models"].items():
        for ctype in data["chart_types"]:
            m = per_type.get(ctype)
            if not m or m["count"] == 0:
                continue
            ed = m["error_distribution"]
            lines.append(
                f"| `{model_name}` | `{ctype}` | {m['count']} | {m['parse_ok_rate']:.2f}% | "
                f"{m['mean_precision']:.4f} | {m['mean_recall']:.4f} | {m['mean_rms_f1']:.4f} | "
                f"{m['mean_rnss']:.4f} | {m['mean_value_recall_at_5']:.4f} | "
                f"{ed.get('none', 0)} | {ed.get('header_mismatch', 0)} | "
                f"{ed.get('structural_error', 0)} | {ed.get('numeric_error', 0)} |"
            )
    return "\n".join(lines) + "\n"


def build_task12_2_report(
    deplot_report: dict[str, Any],
    other_models: dict[str, dict[str, Any]],
    chart_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    deplot_records = list(deplot_report.get("records", []))
    deplot_metrics = _compute_group_metrics(deplot_records)
    comparisons: dict[str, Any] = {}
    for model_name, report in other_models.items():
        other_records = list(report.get("records", []))
        other_metrics = _compute_group_metrics(other_records)
        paired = _paired_comparison(
            other_records, deplot_records, model_name, "deplot"
        )
        per_type_delta: dict[str, Any] = {}
        if chart_types:
            types = sorted(
                {
                    chart_types[str(r["sample_id"])]
                    for r in deplot_records + other_records
                    if str(r.get("sample_id")) in chart_types
                }
            )
            for ctype in types:
                d_sub = [
                    r
                    for r in deplot_records
                    if chart_types.get(str(r.get("sample_id"))) == ctype
                ]
                o_sub = [
                    r
                    for r in other_records
                    if chart_types.get(str(r.get("sample_id"))) == ctype
                ]
                dm = _compute_group_metrics(d_sub)
                om = _compute_group_metrics(o_sub)
                per_type_delta[ctype] = {
                    "samples": dm["count"],
                    "deplot_rms_f1": dm["mean_rms_f1"],
                    f"{model_name}_rms_f1": om["mean_rms_f1"],
                    "delta_vlm_minus_deplot": round(
                        om["mean_rms_f1"] - dm["mean_rms_f1"], 4
                    ),
                    "deplot_rnss": dm["mean_rnss"],
                    f"{model_name}_rnss": om["mean_rnss"],
                }
        comparisons[model_name] = {
            "metrics": other_metrics,
            "delta_vs_deplot": {
                "rms_f1": round(
                    other_metrics["mean_rms_f1"] - deplot_metrics["mean_rms_f1"], 4
                ),
                "rnss": round(
                    other_metrics["mean_rnss"] - deplot_metrics["mean_rnss"], 4
                ),
                "value_recall_at_5": round(
                    other_metrics["mean_value_recall_at_5"]
                    - deplot_metrics["mean_value_recall_at_5"],
                    4,
                ),
                "precision": round(
                    other_metrics["mean_precision"] - deplot_metrics["mean_precision"],
                    4,
                ),
                "recall": round(
                    other_metrics["mean_recall"] - deplot_metrics["mean_recall"], 4
                ),
            },
            "paired_head_to_head": paired,
            "per_chart_type": per_type_delta,
        }
    return {
        "task": "12.2",
        "title": "Task 12.2: DePlot vs Other VLMs (Qwen & Gemini)",
        "deplot": deplot_metrics,
        "comparisons": comparisons,
    }


def format_task12_2_markdown(data: dict[str, Any]) -> str:
    d = data["deplot"]
    lines = [
        "# Task 12.2: DePlot vs Other VLMs Detailed Comparison",
        "",
        "## 1. Overall Baseline & Delta vs DePlot",
        "",
        "| Model | Parse% | Precision | Recall | Mean RMS-F1 | Delta RMS-F1 (vs DePlot) | Mean RNSS | Mean VR@5% | `none` | `header_mismatch` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| `deplot` | {d['parse_ok_rate']:.2f}% | {d['mean_precision']:.4f} | {d['mean_recall']:.4f} | {d['mean_rms_f1']:.4f} | +0.0000 | {d['mean_rnss']:.4f} | {d['mean_value_recall_at_5']:.4f} | {d['error_distribution'].get('none', 0)} | {d['error_distribution'].get('header_mismatch', 0)} |",
    ]
    for model_name, comp in data["comparisons"].items():
        m = comp["metrics"]
        delta = comp["delta_vs_deplot"]
        lines.append(
            f"| `{model_name}` | {m['parse_ok_rate']:.2f}% | {m['mean_precision']:.4f} | {m['mean_recall']:.4f} | "
            f"{m['mean_rms_f1']:.4f} | {delta['rms_f1']:+.4f} | {m['mean_rnss']:.4f} | {m['mean_value_recall_at_5']:.4f} | "
            f"{m['error_distribution'].get('none', 0)} | {m['error_distribution'].get('header_mismatch', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 2. Paired Head-to-Head vs DePlot",
            "",
            "| VLM | Common Samples | VLM Wins (RMS-F1) | DePlot Wins (RMS-F1) | Ties | Mean Delta (VLM - DePlot) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, comp in data["comparisons"].items():
        p = comp["paired_head_to_head"]
        f1 = p["rms_f1"]
        lines.append(
            f"| `{model_name}` | {p['common_samples']} | {f1[f'{model_name}_wins']} | {f1['deplot_wins']} | {f1['ties']} | {f1['mean_delta_a_minus_b']:+.4f} |"
        )
    has_chart_types = any(
        comp.get("per_chart_type") for comp in data["comparisons"].values()
    )
    if has_chart_types:
        lines.extend(
            [
                "",
                "## 3. Per-Chart-Type Delta vs DePlot",
                "",
                "| VLM | Chart Type | Samples | DePlot RMS-F1 | VLM RMS-F1 | Delta (VLM - DePlot) |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for model_name, comp in data["comparisons"].items():
            for ctype, row in comp.get("per_chart_type", {}).items():
                lines.append(
                    f"| `{model_name}` | `{ctype}` | {row['samples']} | {row['deplot_rms_f1']:.4f} | {row[f'{model_name}_rms_f1']:.4f} | {row['delta_vlm_minus_deplot']:+.4f} |"
                )
    return "\n".join(lines) + "\n"


def render_charts(
    task9_2_data: dict[str, Any] | None,
    task12_2_data: dict[str, Any] | None,
    output_dir: Path,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    created: list[Path] = []
    if task9_2_data and task9_2_data.get("chart_types"):
        chart_types = [c for c in task9_2_data["chart_types"] if c != "unknown"]
        if chart_types:
            model_names = list(task9_2_data["models"].keys())
            x = list(range(len(chart_types)))
            width = 0.8 / max(1, len(model_names))
            fig, ax = plt.subplots(figsize=(9, 5))
            for idx, mname in enumerate(model_names):
                scores = [
                    task9_2_data["models"][mname]
                    .get(ctype, {})
                    .get("mean_rms_f1", 0.0)
                    for ctype in chart_types
                ]
                offsets = [pos + (idx - (len(model_names) - 1) / 2) * width for pos in x]
                bars = ax.bar(offsets, scores, width=width, label=mname)
                for bar, val in zip(bars, scores):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.015,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(chart_types)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Mean RMS-F1")
            ax.set_title("Task 9.2: Mean RMS-F1 by Chart Type")
            ax.legend()
            fig.tight_layout()
            out_path = output_dir / "task9_2_chart_type_breakdown.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            created.append(out_path)

    if task12_2_data:
        labels = ["deplot"] + list(task12_2_data["comparisons"].keys())
        rms_f1_vals = [task12_2_data["deplot"]["mean_rms_f1"]] + [
            task12_2_data["comparisons"][m]["metrics"]["mean_rms_f1"]
            for m in task12_2_data["comparisons"]
        ]
        rnss_vals = [task12_2_data["deplot"]["mean_rnss"]] + [
            task12_2_data["comparisons"][m]["metrics"]["mean_rnss"]
            for m in task12_2_data["comparisons"]
        ]
        vr5_vals = [task12_2_data["deplot"]["mean_value_recall_at_5"]] + [
            task12_2_data["comparisons"][m]["metrics"]["mean_value_recall_at_5"]
            for m in task12_2_data["comparisons"]
        ]
        x = list(range(len(labels)))
        width = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        for idx, (metric_name, vals) in enumerate(
            [("RMS-F1", rms_f1_vals), ("RNSS", rnss_vals), ("VR@5%", vr5_vals)]
        ):
            offsets = [pos + (idx - 1) * width for pos in x]
            bars = ax.bar(offsets, vals, width=width, label=metric_name)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.015,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score")
        ax.set_title("Task 12.2: DePlot vs Other VLMs (Overall Metrics)")
        ax.legend()
        fig.tight_layout()
        out_path = output_dir / "task12_2_deplot_vs_vlms.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        created.append(out_path)

    return created


def generate_all_reports(
    qwen_path: Path | None,
    gemini_path: Path | None,
    deplot_path: Path | None,
    chart_types_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_types = load_chart_types(chart_types_path) if chart_types_path else {}

    models: dict[str, dict[str, Any]] = {}
    if qwen_path:
        models["qwen"] = load_report(qwen_path)
    if gemini_path:
        models["gemini"] = load_report(gemini_path)
    if deplot_path:
        models["deplot"] = load_report(deplot_path)

    written_files: list[str] = []
    combined_sections: list[str] = []

    task8_3_data = None
    if "qwen" in models and "gemini" in models:
        task8_3_data = build_task8_3_report(
            models["qwen"], models["gemini"], chart_types or None
        )
        md8 = format_task8_3_markdown(task8_3_data)
        (output_dir / "task8_3_qwen_vs_gemini.md").write_text(md8, encoding="utf-8")
        (output_dir / "task8_3_qwen_vs_gemini.json").write_text(
            json.dumps(task8_3_data, indent=2), encoding="utf-8"
        )
        written_files.extend(
            ["task8_3_qwen_vs_gemini.md", "task8_3_qwen_vs_gemini.json"]
        )
        combined_sections.append(md8)

    task9_2_data = None
    if models and chart_types:
        task9_2_data = build_task9_2_report(models, chart_types)
        md9 = format_task9_2_markdown(task9_2_data)
        (output_dir / "task9_2_chart_type_breakdown.md").write_text(
            md9, encoding="utf-8"
        )
        (output_dir / "task9_2_chart_type_breakdown.json").write_text(
            json.dumps(task9_2_data, indent=2), encoding="utf-8"
        )
        written_files.extend(
            ["task9_2_chart_type_breakdown.md", "task9_2_chart_type_breakdown.json"]
        )
        combined_sections.append(md9)

    task12_2_data = None
    if "deplot" in models and any(k != "deplot" for k in models):
        others = {k: v for k, v in models.items() if k != "deplot"}
        task12_2_data = build_task12_2_report(
            models["deplot"], others, chart_types or None
        )
        md12 = format_task12_2_markdown(task12_2_data)
        (output_dir / "task12_2_deplot_vs_vlms.md").write_text(
            md12, encoding="utf-8"
        )
        (output_dir / "task12_2_deplot_vs_vlms.json").write_text(
            json.dumps(task12_2_data, indent=2), encoding="utf-8"
        )
        written_files.extend(
            ["task12_2_deplot_vs_vlms.md", "task12_2_deplot_vs_vlms.json"]
        )
        combined_sections.append(md12)

    charts = render_charts(task9_2_data, task12_2_data, output_dir)
    written_files.extend([c.name for c in charts])

    if combined_sections:
        (output_dir / "detailed_reports_summary.md").write_text(
            "\n---\n\n".join(combined_sections), encoding="utf-8"
        )
        written_files.append("detailed_reports_summary.md")

    return {
        "written_files": written_files,
        "task8_3": task8_3_data,
        "task9_2": task9_2_data,
        "task12_2": task12_2_data,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate detailed Task 8.3, 9.2, and 12.2 reports from evaluation JSONs and chart-type CSV."
    )
    parser.add_argument("--qwen", type=Path, default=None)
    parser.add_argument("--gemini", type=Path, default=None)
    parser.add_argument("--deplot", type=Path, default=None)
    parser.add_argument("--chart-types", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    result = generate_all_reports(
        qwen_path=args.qwen,
        gemini_path=args.gemini,
        deplot_path=args.deplot,
        chart_types_path=args.chart_types,
        output_dir=args.output_dir,
    )
    print("Generated files:")
    for name in result["written_files"]:
        print(f"  - {args.output_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
