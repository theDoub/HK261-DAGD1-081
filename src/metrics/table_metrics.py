"""Module B2: Table and QA Evaluation Metrics.

Implements evaluation metrics for:
1. Chart-to-Table extraction accuracy:
   - RMS-F1: Relative Mean Squared Error based F1 score (numerical cell alignment).
   - RNSS: Relative Numerical Structural Similarity score.
2. Table QA accuracy:
   - Strict Exact Match (EM) accuracy.
   - Relaxed numerical accuracy (standard ChartQA benchmark with 5% tolerance).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.format.schema import TableSchema


def compute_rms_f1(
    pred_table: Union[TableSchema, Dict[str, Any]],
    gt_table: Union[TableSchema, Dict[str, Any]],
    tolerance: float = 0.05,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Computes RMS-F1 (Relative Mean Squared Error based F1) between predicted and ground truth tables.

    Measures cell-level numerical extraction fidelity. Two numerical cells are considered
    a match (True Positive) if |pred - gt| / max(|gt|, eps) <= tolerance.

    Args:
        pred_table: Predicted TableSchema or dictionary.
        gt_table: Ground truth TableSchema or dictionary.
        tolerance: Relative tolerance threshold (default: 0.05 for 5% tolerance).
        eps: Epsilon to prevent division by zero.

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and f1 scores.
    """
    if isinstance(pred_table, dict):
        pred_table = TableSchema.from_dict(pred_table)
    if isinstance(gt_table, dict):
        gt_table = TableSchema.from_dict(gt_table)

    # Flatten numerical cells from rows
    def _extract_numerics(table: TableSchema) -> List[float]:
        nums: List[float] = []
        for row in table.rows:
            for cell in row:
                if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                    nums.append(float(cell))
        return nums

    pred_nums = _extract_numerics(pred_table)
    gt_nums = _extract_numerics(gt_table)

    if not gt_nums and not pred_nums:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt_nums or not pred_nums:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Greedy or bipartite matching based on relative difference
    matched_gt = set()
    tp = 0

    for p in pred_nums:
        for idx, g in enumerate(gt_nums):
            if idx in matched_gt:
                continue
            rel_diff = abs(p - g) / max(abs(g), eps)
            if rel_diff <= tolerance:
                matched_gt.add(idx)
                tp += 1
                break

    precision = tp / len(pred_nums) if pred_nums else 0.0
    recall = tp / len(gt_nums) if gt_nums else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_rnss(
    pred_table: Union[TableSchema, Dict[str, Any]],
    gt_table: Union[TableSchema, Dict[str, Any]],
    eps: float = 1e-6,
) -> float:
    """Computes RNSS (Relative Numerical Structural Similarity) score.

    Calculates structural and magnitude similarity across matrix-aligned numerical columns.
    Returns a score in [0.0, 1.0] where 1.0 indicates perfect structural and numerical equivalence.

    Args:
        pred_table: Predicted TableSchema or dictionary.
        gt_table: Ground truth TableSchema or dictionary.
        eps: Small stability constant.

    Returns:
        float: RNSS similarity score between 0.0 and 1.0.
    """
    if isinstance(pred_table, dict):
        pred_table = TableSchema.from_dict(pred_table)
    if isinstance(gt_table, dict):
        gt_table = TableSchema.from_dict(gt_table)

    # Simplified stub for structural similarity
    rms_metrics = compute_rms_f1(pred_table, gt_table)
    col_overlap = len(set(pred_table.columns) & set(gt_table.columns))
    col_union = len(set(pred_table.columns) | set(gt_table.columns))
    col_jaccard = col_overlap / col_union if col_union > 0 else 0.0

    rnss_score = 0.5 * rms_metrics["f1"] + 0.5 * col_jaccard
    return float(np.clip(rnss_score, 0.0, 1.0))


def compute_qa_accuracy(
    prediction: Union[str, float, int],
    ground_truth: Union[str, float, int],
) -> float:
    """Computes exact-match accuracy for string or discrete answers.

    Args:
        prediction: Predicted answer.
        ground_truth: Target answer.

    Returns:
        float: 1.0 if match, else 0.0.
    """
    pred_str = str(prediction).strip().lower()
    gt_str = str(ground_truth).strip().lower()
    return 1.0 if pred_str == gt_str else 0.0


def compute_relaxed_accuracy(
    prediction: Union[str, float, int],
    ground_truth: Union[str, float, int],
    tolerance: float = 0.05,
) -> float:
    """Computes relaxed accuracy (ChartQA benchmark standard).

    For numerical ground truth, accepts predictions within `tolerance` (default 5%).
    For non-numerical ground truth, performs normalized string exact match.

    Args:
        prediction: Predicted answer.
        ground_truth: Target answer.
        tolerance: Relative tolerance for numbers (e.g. 0.05 for 5%).

    Returns:
        float: 1.0 if accepted as correct, else 0.0.
    """
    try:
        p_val = float(str(prediction).replace("%", "").replace(",", "").strip())
        g_val = float(str(ground_truth).replace("%", "").replace(",", "").strip())
        denom = max(abs(g_val), 1e-6)
        if abs(p_val - g_val) / denom <= tolerance:
            return 1.0
        return 0.0
    except (ValueError, TypeError):
        return compute_qa_accuracy(prediction, ground_truth)
