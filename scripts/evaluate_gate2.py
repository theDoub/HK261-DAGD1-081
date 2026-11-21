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

BACKEND_TASK_METADATA: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "week": 7,
        "task_id": "7.3",
        "stage_name": "Gate 2 (First Real RMS-F1 on ChartQA)",
        "report_title": "WEEK 7 - GATE 2 REPORT: FIRST REAL RMS-F1 ON CHARTQA (TASK 7.3)",
        "metric_label": "MEAN RMS-F1 (GATE 2 METRIC)",
    },
    "open_vlm": {
        "week": 7,
        "task_id": "7.3",
        "stage_name": "Gate 2 (First Real RMS-F1 on ChartQA)",
        "report_title": "WEEK 7 - GATE 2 REPORT: FIRST REAL RMS-F1 ON CHARTQA (TASK 7.3)",
        "metric_label": "MEAN RMS-F1 (GATE 2 METRIC)",
    },
    "gemini": {
        "week": 8,
        "task_id": "8.2",
        "stage_name": "Multi-Model API Benchmark (Gemini)",
        "report_title": "WEEK 8 BENCHMARK REPORT: SECOND MODEL EVALUATION - GEMINI (TASK 8.2)",
        "metric_label": "MEAN RMS-F1 (TASK 8.2)",
    },
    "claude": {
        "week": 8,
        "task_id": "8.2",
        "stage_name": "Multi-Model API Benchmark (Claude)",
        "report_title": "WEEK 8 BENCHMARK REPORT: SECOND MODEL EVALUATION - CLAUDE (TASK 8.2)",
        "metric_label": "MEAN RMS-F1 (TASK 8.2)",
    },
    "deplot": {
        "week": 12,
        "task_id": "12.1",
        "stage_name": "Baseline Comparison (DePlot)",
        "report_title": "WEEK 12 BASELINE REPORT: DEPLOT EVALUATION ON CHARTQA (TASK 12.1)",
        "metric_label": "MEAN RMS-F1 (TASK 12.1)",
    },
    "mock": {
        "week": 7,
        "task_id": "7.2",
        "stage_name": "Pipeline End-to-End Verification (Mock)",
        "report_title": "WEEK 7 PIPELINE VERIFICATION REPORT: MOCK BACKEND (TASK 7.2)",
        "metric_label": "MEAN RMS-F1 (MOCK CHECK)",
    },
}


def get_backend_task_metadata(backend_name: str) -> Dict[str, Any]:
    key = backend_name.lower()
    if key in BACKEND_TASK_METADATA:
        return dict(BACKEND_TASK_METADATA[key])
    return {
        "week": None,
        "task_id": "unknown",
        "stage_name": f"Evaluation ({backend_name})",
        "report_title": f"CHART-TO-TABLE EVALUATION REPORT ({backend_name.upper()})",
        "metric_label": "MEAN RMS-F1",
    }


def build_harness(
    backend_name: str,
    device: str = "auto",
    quantization: str = "none",
    model_name: Optional[str] = None,
    max_new_tokens: int = 256,
) -> VLMTableHarness:
    """Builds VLMTableHarness configured for the specified backend."""
    backend_lower = backend_name.lower()

    if backend_lower == "deplot":
        backend = DePlotBackend(device=None if device == "auto" else device)
        return VLMTableHarness(backend=backend, target_format="deplot")

    elif backend_lower in {"qwen", "open_vlm"}:
        if quantization == "4bit":
            if device == "auto":
                import torch

                if not torch.cuda.is_available():
                    raise ValueError(
                        "Qwen 4-bit quantization requires CUDA. "
                        "Use --device cpu --quantization none for CPU inference."
                    )
            elif not device.startswith("cuda"):
                raise ValueError(
                    "Qwen 4-bit quantization requires CUDA. "
                    "Use --device cpu --quantization none for CPU inference."
                )
        backend = OpenVLMBackend(
            model_name=model_name or "Qwen/Qwen2.5-VL-7B-Instruct",
            device=device,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
        )
        return VLMTableHarness(backend=backend, target_format="markdown")

    elif backend_lower == "gemini":
        model_name = None
        cfg_file = Path("configs/config.yaml")
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    model_name = cfg.get("api_vlm", {}).get("gemini", {}).get("model_name")
            except Exception:
                pass
        return VLMTableHarness(backend=GeminiVLMBackend(model_name=model_name), target_format="markdown")

    elif backend_lower == "claude":
        model_name = None
        cfg_file = Path("configs/config.yaml")
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    model_name = cfg.get("api_vlm", {}).get("claude", {}).get("model_name")
            except Exception:
                pass
        return VLMTableHarness(backend=ClaudeVLMBackend(model_name=model_name), target_format="markdown")

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
    prompt_override: Optional[str] = None,
    sample_ids: Optional[Sequence[str]] = None,
    max_new_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs Gate 2 evaluation across real ChartQA samples and computes RMS-F1."""
    loader = ChartDataLoader(data_dir=data_dir, split=".")
    items: List[ChartDatasetItem] = loader.scan_dataset()

    if not items:
        logger.error("No dataset items found in %s", data_dir)
        return {"error": "Empty dataset"}

    # Filter only items that have gold CSV tables
    eval_items = [it for it in items if it.table is not None and it.image_path is not None and it.image_path.exists()]
    if sample_ids is not None:
        requested_ids = list(sample_ids)
        if not requested_ids:
            raise ValueError("sample_ids must contain at least one sample ID")
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("sample_ids must not contain duplicates")
        items_by_id = {item.id: item for item in eval_items}
        missing_ids = [sample_id for sample_id in requested_ids if sample_id not in items_by_id]
        if missing_ids:
            raise ValueError(f"Requested sample IDs are missing from the dataset: {missing_ids}")
        eval_items = [items_by_id[sample_id] for sample_id in requested_ids]
    elif num_samples is not None and num_samples > 0:
        eval_items = eval_items[:num_samples]

    total_eval = len(eval_items)
    logger.info("Initializing Gate 2 evaluation with backend '%s' on %d samples...", backend_name, total_eval)

    active_prompt: Optional[str] = None
    config_file = Path("configs/config.yaml")
    backend_lower = backend_name.lower()
    cfg: Dict[str, Any] = {}
    if config_file.exists() and backend_lower in {"qwen", "open_vlm", "gemini", "claude"}:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    qwen_config = cfg.get("qwen", {})
    api_config = cfg.get("api_vlm", {})
    model_name: Optional[str] = None
    if backend_lower in {"qwen", "open_vlm"}:
        model_name = qwen_config.get("model_name_or_path")
        active_prompt = qwen_config.get("prompt")
    elif backend_lower == "gemini":
        model_name = api_config.get("gemini", {}).get("model_name")
        active_prompt = api_config.get("gemini", {}).get("prompt") or cfg.get("gemini", {}).get("prompt")
    elif backend_lower == "claude":
        model_name = api_config.get("claude", {}).get("model_name")
        active_prompt = api_config.get("claude", {}).get("prompt") or cfg.get("claude", {}).get("prompt")

    if max_new_tokens is None:
        max_new_tokens = 256
        if backend_lower in {"qwen", "open_vlm"}:
            configured_max_new_tokens = qwen_config.get("max_new_tokens", 256)
            if isinstance(configured_max_new_tokens, bool) or not isinstance(configured_max_new_tokens, int):
                raise ValueError("qwen.max_new_tokens in configs/config.yaml must be a positive integer")
            max_new_tokens = configured_max_new_tokens
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

    if prompt_override is not None:
        if not prompt_override.strip():
            raise ValueError("prompt_override must not be empty")
        active_prompt = prompt_override

    harness = build_harness(
        backend_name,
        device=device,
        quantization=quantization,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )

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

    task_meta = get_backend_task_metadata(backend_name)

    summary = {
        "week": task_meta["week"],
        "task_id": task_meta["task_id"],
        "stage_name": task_meta["stage_name"],
        "report_title": task_meta["report_title"],
        "metric_label": task_meta["metric_label"],
        "gate": task_meta["stage_name"],
        "dataset_name": "ChartQA Official Benchmark (Test Split)",
        "dataset_path": str(data_dir),
        "total_samples_evaluated": total_eval,
        "evaluation_mode": (
            "Pipeline Baseline Verification (Deterministic Mock Generator)"
            if backend_name == "mock"
            else f"VLM Model Inference ({backend_name})"
        ),
        "backend": backend_name,
        "conditions": {
            "model_name_or_path": model_name,
            "device_requested": device,
            "device_used": getattr(harness.backend, "device", device),
            "quantization": quantization if backend_lower in {"qwen", "open_vlm"} else None,
            "max_new_tokens": max_new_tokens if backend_lower in {"qwen", "open_vlm"} else None,
            "prompt": active_prompt,
            "prompt_source": (
                "override"
                if prompt_override is not None
                else "config"
                if active_prompt is not None
                else "backend_default"
            ),
            "sample_ids": [item.id for item in eval_items],
        },
        "parse_success_count": parse_success_count,
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
        logger.info("Saved evaluation report to %s", output_path)

    return summary


def print_kpi_summary(summary: Dict[str, Any]) -> None:
    title = summary.get("report_title") or "CHART-TO-TABLE BENCHMARK REPORT"
    parse_rate = summary.get("parse_success_rate", 0.0)
    total_eval = summary.get("total_samples_evaluated", 0)
    parse_count = summary.get("parse_success_count")
    if parse_rate == 100.0 and total_eval > 0:
        parse_note = "All samples parsed cleanly"
    elif parse_count is not None:
        parse_note = f"{parse_count}/{total_eval} samples parsed"
    else:
        parse_note = "Partial parse success"
    metric_label = summary.get("metric_label") or "MEAN RMS-F1"

    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)
    print(f" Weekly Plan Milestone        : Week {summary.get('week')} | Task {summary.get('task_id')}")
    print(f" Benchmark Dataset            : {summary.get('dataset_name')}")
    print(f" Dataset Directory            : {summary.get('dataset_path')}")
    print(f" Total Real Samples Evaluated : {total_eval} (Images + Gold CSV Tables)")
    print(f" Evaluation Mode / Target     : {summary.get('evaluation_mode')}")
    print(f" Parse Success Rate (parse_ok): {parse_rate}% ({parse_note})")
    print("-" * 75)
    print(f" Mean Precision                : {summary.get('mean_precision')}")
    print(f" Mean Recall                   : {summary.get('mean_recall')}")
    print(f" {metric_label:<30}: {summary.get('mean_rms_f1')} [*]")
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
    parser.add_argument(
        "--max-new-tokens",
        default=None,
        type=int,
        help="Maximum generated tokens (defaults to qwen.max_new_tokens in configs/config.yaml)",
    )
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 file containing the exact prompt to use")
    parser.add_argument(
        "--sample-ids-file",
        type=Path,
        help="UTF-8 newline-delimited sample IDs; preserves the listed order for paired evaluations",
    )
    parser.add_argument("--output", default=Path("outputs/gate2_evaluation_report.json"), type=Path, help="Output JSON")

    args = parser.parse_args(argv)
    prompt_override = None
    if args.prompt_file is not None:
        prompt_override = args.prompt_file.read_text(encoding="utf-8")
    sample_ids = None
    if args.sample_ids_file is not None:
        sample_ids = [
            line.strip()
            for line in args.sample_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    summary = evaluate_gate2(
        data_dir=args.data_dir,
        backend_name=args.backend,
        num_samples=args.num_samples,
        device=args.device,
        quantization=args.quantization,
        output_path=args.output,
        prompt_override=prompt_override,
        sample_ids=sample_ids,
        max_new_tokens=args.max_new_tokens,
    )

    print_kpi_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
