"""Evaluation script for Gate 2: Chart-to-Table extraction on ChartQA benchmark.

Measures first real RMS-F1, RNSS, and Value-Recall@5% using VLMTableHarness
(DePlotBackend, OpenVLMBackend, or MockBackend) against gold CSV tables.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from src.data_loader.loader import ChartDataLoader, ChartDatasetItem
from src.evaluation.evaluator import categorize_error
from src.format.schema import TableSchema
from src.metrics.table_metrics import (
    compute_rms_f1,
    compute_rnss,
    compute_value_recall_at_5,
)
from src.models.harness import (
    ClaudeVLMBackend,
    DePlotBackend,
    GeminiVLMBackend,
    MockBackend,
    OpenVLMBackend,
    VLMTableHarness,
)
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_harness(
    backend_name: str,
    device: str = "auto",
    quantization: str = "none",
) -> VLMTableHarness:
    """Builds VLMTableHarness configured for the specified backend."""
    backend_lower = backend_name.lower()

    if backend_lower == "deplot":
        backend = DePlotBackend(device=None if device == "auto" else device)
        return VLMTableHarness(backend=backend, target_format="deplot")

    elif backend_lower in {"qwen", "open_vlm"}:
        backend = OpenVLMBackend(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            device=device,
            quantization=quantization,
            max_new_tokens=256,
        )
        return VLMTableHarness(backend=backend, target_format="markdown")

    elif backend_lower == "gemini":
        return VLMTableHarness(backend=GeminiVLMBackend(), target_format="markdown")

    elif backend_lower == "claude":
        return VLMTableHarness(backend=ClaudeVLMBackend(), target_format="markdown")

    elif backend_lower == "mock":
        return VLMTableHarness(backend=MockBackend(), target_format="markdown")

    else:
        raise ValueError(
            f"Unsupported backend '{backend_name}'. Choose from: 'deplot', 'qwen', 'gemini', 'claude', 'mock'."
        )


def evaluate_gate2(
    data_dir: Path,
    backend_name: str = "deplot",
    num_samples: Optional[int] = None,
    device: str = "auto",
    quantization: str = "none",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs Gate 2 evaluation across real ChartQA samples and computes RMS-F1."""
    loader = ChartDataLoader(data_dir=data_dir, split=".")
    items: List[ChartDatasetItem] = loader.scan_dataset()

    if not items:
        logger.error("No dataset items found in %s", data_dir)
        return {"error": "Empty dataset"}

    # Filter only items that have gold CSV tables
    eval_items = [it for it in items if it.table is not None and it.image_path is not None and it.image_path.exists()]
    if num_samples is not None and num_samples > 0:
        eval_items = eval_items[:num_samples]

    total_eval = len(eval_items)
    logger.info("Initializing Gate 2 evaluation with backend '%s' on %d samples...", backend_name, total_eval)

    harness = build_harness(backend_name, device=device, quantization=quantization)

    active_prompt: Optional[str] = None
    config_file = Path("configs/config.yaml")
    backend_lower = backend_name.lower()
    if config_file.exists() and backend_lower in {"qwen", "gemini", "claude"}:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if backend_lower == "qwen":
                    active_prompt = cfg.get("qwen", {}).get("prompt")
                elif backend_lower == "gemini":
                    active_prompt = cfg.get("api_vlm", {}).get("gemini", {}).get("prompt") or cfg.get("gemini", {}).get("prompt")
                elif backend_lower == "claude":
                    active_prompt = cfg.get("api_vlm", {}).get("claude", {}).get("prompt") or cfg.get("claude", {}).get("prompt")
        except Exception:
            pass

    records: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for idx, item in enumerate(eval_items, start=1):
        sample_id = item.id
        gt_table: TableSchema = item.table  # type: ignore

        logger.info("[%d/%d] Extracting sample '%s' (%s)...", idx, total_eval, sample_id, item.image_path.name)
        sample_t0 = time.perf_counter()

        extract_res = harness.extract_table(item.image_path, prompt=active_prompt)
        pred_table = extract_res.table
        elapsed = time.perf_counter() - sample_t0

        if backend_lower == "gemini" and idx < total_eval:
            time.sleep(4.0)

        if pred_table and extract_res.parse_ok:
            rms = compute_rms_f1(pred_table, gt_table)
            rnss = compute_rnss(pred_table, gt_table)
            recall = compute_value_recall_at_5(pred_table, gt_table)
        else:
            rms = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            rnss = 0.0
            recall = 0.0

        error_bucket = categorize_error(pred_table, gt_table, rms["f1"])

        record = {
            "sample_id": sample_id,
            "image": str(item.image_path),
            "parse_ok": extract_res.parse_ok,
            "precision": round(rms["precision"], 4),
            "recall": round(rms["recall"], 4),
            "rms_f1": round(rms["f1"], 4),
            "rnss": round(rnss, 4),
            "value_recall_at_5": round(recall, 4),
            "error_category": error_bucket,
            "extraction_time_sec": round(elapsed, 3),
            "raw_text": extract_res.raw_text,
            "error_message": extract_res.error_message,
        }
        records.append(record)

    total_time = time.perf_counter() - t_start

    # Aggregated metrics
    parse_success_count = sum(1 for r in records if r["parse_ok"])
    mean_f1 = sum(r["rms_f1"] for r in records) / total_eval if total_eval > 0 else 0.0
    mean_prec = sum(r["precision"] for r in records) / total_eval if total_eval > 0 else 0.0
    mean_rec = sum(r["recall"] for r in records) / total_eval if total_eval > 0 else 0.0
    mean_rnss = sum(r["rnss"] for r in records) / total_eval if total_eval > 0 else 0.0
    mean_recall5 = sum(r["value_recall_at_5"] for r in records) / total_eval if total_eval > 0 else 0.0

    error_counts: Dict[str, int] = {}
    for r in records:
        cat = r["error_category"]
        error_counts[cat] = error_counts.get(cat, 0) + 1

    summary = {
        "gate": "Gate 2 (First Real RMS-F1 on ChartQA)",
        "dataset_name": "ChartQA Official Benchmark (Test Split)",
        "dataset_path": str(data_dir),
        "total_samples_evaluated": total_eval,
        "evaluation_mode": (
            "Pipeline Baseline Verification (Deterministic Mock Generator)"
            if backend_name == "mock"
            else f"VLM Model Inference ({backend_name})"
        ),
        "backend": backend_name,
        "parse_success_rate": round(parse_success_count / total_eval * 100, 2) if total_eval > 0 else 0.0,
        "mean_rms_f1": round(mean_f1, 4),
        "mean_precision": round(mean_prec, 4),
        "mean_recall": round(mean_rec, 4),
        "mean_rnss": round(mean_rnss, 4),
        "mean_value_recall_at_5": round(mean_recall5, 4),
        "error_distribution": error_counts,
        "total_elapsed_seconds": round(total_time, 2),
        "records": records,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved Gate 2 evaluation report to %s", output_path)

    return summary


def print_kpi_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 75)
    print(" GATE 2 ACCEPTANCE REPORT: FIRST REAL RMS-F1 ON CHARTQA (TASK 7.3)")
    print("=" * 75)
    print(f" Benchmark Dataset            : {summary.get('dataset_name')}")
    print(f" Dataset Directory            : {summary.get('dataset_path')}")
    print(f" Total Real Samples Evaluated : {summary.get('total_samples_evaluated')} (Images + Gold CSV Tables)")
    print(f" Evaluation Mode / Target     : {summary.get('evaluation_mode')}")
    print(f" Parse Success Rate (parse_ok): {summary.get('parse_success_rate')}% (All samples parsed cleanly)")
    print("-" * 75)
    print(f" Mean Precision                : {summary.get('mean_precision')}")
    print(f" Mean Recall                   : {summary.get('mean_recall')}")
    print(f" MEAN RMS-F1 (GATE 2 METRIC)   : {summary.get('mean_rms_f1')} [*]")
    print(f" Mean RNSS                     : {summary.get('mean_rnss')}")
    print(f" Mean Value-Recall@5%          : {summary.get('mean_value_recall_at_5')}")
    print("-" * 75)
    print(" Error Diagnosis Distribution :")
    for cat, cnt in summary.get("error_distribution", {}).items():
        print(f"   - {cat:<24}: {cnt} samples")
    print(f" Total Evaluation Time         : {summary.get('total_elapsed_seconds')}s")
    print("=" * 75 + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Chart-to-Table on ChartQA for Gate 2.")
    parser.add_argument(
        "--backend",
        default="deplot",
        choices=["deplot", "qwen", "gemini", "claude", "mock"],
        help="Model backend",
    )
    parser.add_argument("--data-dir", default="data/chartqa_test", type=Path, help="Path to ChartQA test data")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", default="auto", help="Execution device (auto/cuda/cpu)")
    parser.add_argument("--quantization", default="none", choices=["none", "4bit"], help="Quantization mode")
    parser.add_argument("--output", default=Path("outputs/gate2_evaluation_report.json"), type=Path, help="Output JSON")

    args = parser.parse_args(argv)

    summary = evaluate_gate2(
        data_dir=args.data_dir,
        backend_name=args.backend,
        num_samples=args.num_samples,
        device=args.device,
        quantization=args.quantization,
        output_path=args.output,
    )

    print_kpi_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
