"""Module B6: Pipeline Evaluation and Error Analysis.

Orchestrates table extraction across datasets, evaluates predictions against gold tables
using Module B2 metrics, computes aggregations broken down by chart type and complexity,
and categorizes extraction errors for error analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import yaml

from src.data_loader.loader import ChartDatasetItem
from src.format.schema import TableSchema
from src.metrics.table_metrics import (
    compute_rms_f1,
    compute_rnss,
    compute_value_recall_at_5,
)
from src.models.harness import ExtractionResult, VLMTableHarness

logger = logging.getLogger(__name__)


def categorize_error(
    pred_table: Optional[TableSchema],
    gt_table: TableSchema,
    f1_score: float,
) -> str:
    """Classifies extraction failures into actionable diagnosis buckets.

    Buckets:
        - 'none': Successful extraction (F1 >= 0.8).
        - 'missing_data': Model produced no table or completely empty rows.
        - 'header_mismatch': Low column name overlap with ground-truth.
        - 'structural_error': Significant mismatch in row/column count or grid shape.
        - 'numeric_error': Structure is aligned but numerical values are inaccurate.

    Args:
        pred_table: Extracted table structure or None.
        gt_table: Ground-truth table structure.
        f1_score: Computed RMS-F1 score.

    Returns:
        str: Error category string.
    """
    if f1_score >= 0.8:
        return "none"

    if pred_table is None or len(pred_table.rows) == 0 or len(pred_table.columns) == 0:
        return "missing_data"

    pred_cols = set(c.strip().lower() for c in pred_table.columns)
    gt_cols = set(c.strip().lower() for c in gt_table.columns)
    col_overlap = len(pred_cols & gt_cols) / max(len(gt_cols), 1)

    if col_overlap < 0.5:
        return "header_mismatch"

    if len(pred_table.rows) != len(gt_table.rows) or len(pred_table.columns) != len(gt_table.columns):
        return "structural_error"

    return "numeric_error"


class PipelineEvaluator:
    """End-to-end evaluation harness supporting metric aggregation and error diagnosis."""

    def __init__(
        self,
        harness: Optional[VLMTableHarness] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes the evaluator with an extraction harness and optional config."""
        self.harness = harness or VLMTableHarness(backend="mock")
        self.config_path = Path(config_path) if config_path else Path("configs/config.yaml")

    def run_sanity_check(self) -> Dict[str, Any]:
        """Verifies environment and pipeline configuration loading."""
        if not self.config_path.exists():
            logger.warning("Config file not found at %s", self.config_path)
            return {"status": "warning", "message": f"Missing config: {self.config_path}"}

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return {"status": "ok", "config": config}

    def evaluate_sample(
        self,
        gt_table: Optional[TableSchema],
        image_path: Optional[Union[str, Path]] = None,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates extraction on a single chart sample against gold table.

        Args:
            gt_table: Ground-truth TableSchema, if available.
            image_path: Optional path to chart image.
            sample_id: Dataset identifier to preserve in the evaluation record.

        Returns:
            Dict[str, Any]: Single sample evaluation record.
        """
        # Step 1: Extract via VLM Harness (Module B5)
        extract_res: ExtractionResult = self.harness.extract_table(image_path)
        pred_table = extract_res.table

        # Step 2: Compute Metrics (Module B2)
        if gt_table is not None and pred_table and extract_res.parse_ok:
            rms = compute_rms_f1(pred_table, gt_table)
            rnss = compute_rnss(pred_table, gt_table)
            val_recall = compute_value_recall_at_5(pred_table, gt_table)
        else:
            rms = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            rnss = 0.0
            val_recall = 0.0

        # Step 3: Categorize Error
        error_bucket = (
            categorize_error(pred_table, gt_table, rms["f1"])
            if gt_table is not None
            else "missing_data"
        )

        meta = gt_table.metadata if gt_table is not None else {}
        return {
            "sample_id": sample_id or meta.get("id", "sample"),
            "image_path": str(image_path) if image_path is not None else None,
            "backend": extract_res.backend_name,
            "raw_text": extract_res.raw_text,
            "chart_type": meta.get("chart_type", "unknown"),
            "complexity": meta.get("complexity", "unknown"),
            "ground_truth_available": gt_table is not None,
            "parse_ok": extract_res.parse_ok,
            "precision": rms["precision"],
            "recall": rms["recall"],
            "rms_f1": rms["f1"],
            "rnss": rnss,
            "value_recall_at_5": val_recall,
            "error_category": error_bucket,
            "error_message": extract_res.error_message,
        }

    def evaluate_dataset(
        self,
        dataset: Sequence[Union[TableSchema, ChartDatasetItem]],
        predictions_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Runs full evaluation across a collection of tables, computing breakdowns by type and complexity.

        Args:
            dataset: Ground-truth tables or chart dataset items containing image paths.
            predictions_path: Optional JSONL file to write one extraction record per item.

        Returns:
            Dict[str, Any]: Aggregated summary and error analysis.
        """
        records: List[Dict[str, Any]] = []
        total = len(dataset)

        output_path = Path(predictions_path) if predictions_path is not None else None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        output_file = (
            output_path.open("w", encoding="utf-8", newline="\n")
            if output_path is not None
            else None
        )
        try:
            for sample in dataset:
                if isinstance(sample, ChartDatasetItem):
                    gt_table = sample.table
                    image_path = sample.image_path
                    sample_id = sample.id
                else:
                    gt_table = sample
                    image_path = None
                    sample_id = None

                record = self.evaluate_sample(gt_table, image_path, sample_id)
                records.append(record)
                if output_file is not None:
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
        finally:
            if output_file is not None:
                output_file.close()

        if total == 0:
            return {"total_samples": 0, "mean_rms_f1": 0.0}

        mean_f1 = sum(r["rms_f1"] for r in records) / total
        mean_rnss = sum(r["rnss"] for r in records) / total
        mean_val_recall = sum(r["value_recall_at_5"] for r in records) / total

        # Breakdown by chart_type
        by_chart_type: Dict[str, List[float]] = {}
        by_complexity: Dict[str, List[float]] = {}
        error_distribution: Dict[str, int] = {
            "missing_data": 0,
            "header_mismatch": 0,
            "structural_error": 0,
            "numeric_error": 0,
            "none": 0,
        }

        for r in records:
            ctype = r["chart_type"]
            comp = r["complexity"]
            err = r["error_category"]

            by_chart_type.setdefault(ctype, []).append(r["rms_f1"])
            by_complexity.setdefault(comp, []).append(r["rms_f1"])
            error_distribution[err] = error_distribution.get(err, 0) + 1

        chart_type_summary = {
            k: float(sum(v) / len(v)) for k, v in by_chart_type.items()
        }
        complexity_summary = {
            k: float(sum(v) / len(v)) for k, v in by_complexity.items()
        }

        return {
            "total_samples": total,
            "mean_rms_f1": float(mean_f1),
            "mean_rnss": float(mean_rnss),
            "mean_value_recall_at_5": float(mean_val_recall),
            "breakdown_by_chart_type": chart_type_summary,
            "breakdown_by_complexity": complexity_summary,
            "error_analysis": error_distribution,
            "records": records,
        }
