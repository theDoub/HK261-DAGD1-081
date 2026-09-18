"""Module B6: Two-Stage Pipeline Evaluation Engine.

Orchestrates the multimodal Chart-to-Table -> Table QA evaluation flow:
Stage 1: Input Chart -> DePlot Model -> Predicted TableSchema.
Stage 2: Predicted TableSchema + Question -> VLM/LLM Client -> Predicted Answer.
Evaluation: Computes RMS-F1, RNSS (Stage 1) and Exact/Relaxed Accuracy (Stage 2).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.data_loader.loader import ChartDatasetItem
from src.format.schema import TableSchema
from src.metrics.table_metrics import (
    compute_qa_accuracy,
    compute_relaxed_accuracy,
    compute_rms_f1,
    compute_rnss,
)
from src.models.deplot import DePlotHarness
from src.models.vlm_client import VLMClient

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Evaluates the end-to-end two-stage ChartQA pipeline on a dataset."""

    def __init__(
        self,
        deplot_harness: Optional[DePlotHarness] = None,
        vlm_client: Optional[VLMClient] = None,
    ) -> None:
        """Initializes the evaluator with model stubs.

        Args:
            deplot_harness: Harness for visual table extraction (Stage 1).
            vlm_client: Client for table QA reasoning (Stage 2).
        """
        self.deplot_harness = deplot_harness or DePlotHarness(dry_run=True)
        self.vlm_client = vlm_client or VLMClient(provider="vlm_stub")

    def evaluate_sample(self, item: ChartDatasetItem) -> Dict[str, Any]:
        """Runs the two-stage pipeline on a single sample and calculates metrics.

        Args:
            item: Input dataset item containing image path, question, answer, and optional table.

        Returns:
            Dict[str, Any]: Evaluation record containing predictions and metric scores.
        """
        # Stage 1: Chart -> Table
        pred_table: TableSchema = self.deplot_harness.extract_table(
            image_input=item.image_path or "mock_path.png"
        )

        # Stage 2: Table + Question -> Answer
        pred_answer: str = self.vlm_client.answer_question(
            table=item.table or pred_table,
            question=item.question,
        )

        result: Dict[str, Any] = {
            "id": item.id,
            "question": item.question,
            "ground_truth_answer": item.answer,
            "predicted_answer": pred_answer,
            "pred_table": pred_table.to_dict(),
        }

        # Calculate Table Extraction Metrics if Ground Truth Table exists
        if item.table:
            rms = compute_rms_f1(pred_table, item.table)
            rnss = compute_rnss(pred_table, item.table)
            result["rms_f1"] = rms["f1"]
            result["rnss"] = rnss

        # Calculate QA Metrics if Ground Truth Answer exists
        if item.answer is not None:
            result["qa_exact_match"] = compute_qa_accuracy(pred_answer, item.answer)
            result["qa_relaxed_accuracy"] = compute_relaxed_accuracy(pred_answer, item.answer)

        return result

    def evaluate_dataset(self, dataset: List[ChartDatasetItem]) -> Dict[str, Any]:
        """Evaluates a full collection of dataset items and aggregates scores.

        Args:
            dataset: List of ChartDatasetItem instances.

        Returns:
            Dict[str, Any]: Aggregated evaluation summary.
        """
        records: List[Dict[str, Any]] = []
        total_samples = len(dataset)

        for item in dataset:
            record = self.evaluate_sample(item)
            records.append(record)

        # Aggregate metrics
        avg_em = (
            sum(r.get("qa_exact_match", 0.0) for r in records) / total_samples
            if total_samples > 0
            else 0.0
        )
        avg_relaxed = (
            sum(r.get("qa_relaxed_accuracy", 0.0) for r in records) / total_samples
            if total_samples > 0
            else 0.0
        )

        summary = {
            "total_samples": total_samples,
            "mean_exact_match": avg_em,
            "mean_relaxed_accuracy": avg_relaxed,
            "sample_results": records,
        }

        logger.info(
            "Evaluation complete. Total: %d, Mean EM: %.4f, Mean Relaxed Acc: %.4f",
            total_samples,
            avg_em,
            avg_relaxed,
        )
        return summary
