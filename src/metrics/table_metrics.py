"""Module B2: Table Metrics.

Provides metric calculations for Chart-to-Table extraction evaluation:
- RMS-F1: Relative Mean Squared Error based F1 with Hungarian bipartite matching.
- RNSS: Relative Numerical Structural Similarity on numerical cells.
- Value-Recall@5%: Proportion of ground-truth numeric values captured within 5% relative error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.format.schema import TableSchema, TableTriplet


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Computes normalized Levenshtein distance between two strings in [0.0, 1.0].

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        float: Normalized edit distance.
    """
    if s1 == s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 1.0

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    max_len = max(len1, len2)
    return dp[len1][len2] / max_len


def normalized_levenshtein_tau(s1: str, s2: str, tau: float = 0.5) -> float:
    """Computes Normalized Levenshtein distance with cutoff threshold tau (NV 3.1).

    If the normalized edit distance exceeds tau, returns 1.0 (maximum penalty).
    Otherwise returns the computed normalized distance in [0.0, tau].

    Args:
        s1: First string (e.g. predicted column header).
        s2: Second string (e.g. ground-truth column header).
        tau: Tolerance threshold (default: 0.5).

    Returns:
        float: Thresholded edit distance in [0.0, 1.0].
    """
    nl = normalized_levenshtein(s1, s2)
    return 1.0 if nl > tau else nl


def relative_distance(pv: Any, gv: Any, eps: float = 1e-6) -> float:
    """Computes relative numerical distance D(p, t) = min(1.0, |p - t| / |t|) (NV 3.2).

    For numeric pairs, returns relative distance capped at 1.0.
    For non-numeric pairs, returns 0.0 if string values match (case-insensitive), else 1.0.

    Args:
        pv: Predicted cell value.
        gv: Ground-truth cell value.
        eps: Numerical stability constant to avoid division by zero.

    Returns:
        float: Relative distance in [0.0, 1.0].
    """
    if isinstance(pv, (int, float)) and isinstance(gv, (int, float)):
        denom = max(abs(float(gv)), eps)
        return float(min(1.0, abs(float(pv) - float(gv)) / denom))
    return 0.0 if str(pv).strip().lower() == str(gv).strip().lower() else 1.0


def compute_pair_similarity(
    p: TableTriplet,
    g: TableTriplet,
    tau: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Computes similarity between two table cell triplets (NV 3.2).

    Formula:
        sim(p, t) = (1 - NL_tau(col_p, col_t)) * (1 - D(val_p, val_t)) in [0.0, 1.0].

    Args:
        p: Predicted TableTriplet.
        g: Ground-truth TableTriplet.
        tau: Header Levenshtein threshold.
        eps: Numerical stability constant.

    Returns:
        float: Pairwise similarity score in [0.0, 1.0].
    """
    nl_tau = normalized_levenshtein_tau(str(p.col), str(g.col), tau=tau)
    d_val = relative_distance(p.value, g.value, eps=eps)
    return float((1.0 - nl_tau) * (1.0 - d_val))


def _extract_triplets(table: Union[TableSchema, Dict[str, Any], Any]) -> List[TableTriplet]:
    """Helper to convert any table input into a list of TableTriplets."""
    if isinstance(table, TableSchema):
        return table.triplets
    if isinstance(table, dict):
        schema = TableSchema.from_dict(table)
        return schema.triplets
    return []


def compute_rms_f1(
    pred_table: Any,
    gt_table: Any,
    tau: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Computes RMS-F1 between predicted and ground-truth table triplets.

    Mathematical Definition:
        1. Header matching: Normalized Levenshtein edit distance NL_tau(h_p, h_t).
        2. Value matching: Relative numerical distance D(v_p, v_t).
        3. Pair similarity: sim(p, t) = (1 - NL_tau) * (1 - D) in [0.0, 1.0].
        4. Bipartite matching: Optimal assignment via Hungarian algorithm.
        5. Score aggregation: Precision P, Recall R, Harmonic Mean F1.

    Args:
        pred_table: Predicted TableSchema, dict, or list of triplets.
        gt_table: Ground-truth TableSchema, dict, or list of triplets.
        tau: Header distance threshold (default: 0.5).
        eps: Stability constant for division.

    Returns:
        Dict[str, float]: Dictionary with 'precision', 'recall', and 'f1'.
    """
    preds = _extract_triplets(pred_table)
    gts = _extract_triplets(gt_table)

    n_pred, m_gt = len(preds), len(gts)
    if n_pred == 0 and m_gt == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if n_pred == 0 or m_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Construct similarity matrix [n_pred, m_gt]
    cost_matrix = np.ones((n_pred, m_gt), dtype=np.float64)
    sim_matrix = np.zeros((n_pred, m_gt), dtype=np.float64)

    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            sim = compute_pair_similarity(p, g, tau=tau, eps=eps)
            sim_matrix[i, j] = sim
            cost_matrix[i, j] = 1.0 - sim

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_sim_sum = float(np.sum(sim_matrix[row_ind, col_ind]))

    precision = matched_sim_sum / n_pred
    recall = matched_sim_sum / m_gt
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_rnss(pred_table: Any, gt_table: Any, eps: float = 1e-6) -> float:
    """Computes Relative Numerical Structural Similarity (RNSS) on numeric entries only.

    As defined in P05 DePlot (§3.1): RNSS looks only at the *unordered set* of numeric
    entries in the predicted and target tables, and measures how well the predicted set
    matches the target set using relative distances and optimal (Hungarian) matching.

    Formula:
        D(p, t) = min(1, |p - t| / |t|)   for each numeric pair
        X = optimal binary matching matrix (Hungarian on D_matrix)
        RNSS = 1 - sum(X_ij * D(p_i, t_j)) / max(N, M)

    Note: RNSS intentionally ignores column/row headers and table structure —
    use RMS-F1 when structure matters.

    Args:
        pred_table: Predicted table representation.
        gt_table: Ground-truth table representation.
        eps: Numerical stability constant for division by zero.

    Returns:
        float: RNSS score in [0.0, 1.0]. 1.0 = perfect numeric match.
    """
    preds = _extract_triplets(pred_table)
    gts = _extract_triplets(gt_table)

    # Filter to numeric values only
    pred_nums = [float(p.value) for p in preds if isinstance(p.value, (int, float))]
    gt_nums = [float(g.value) for g in gts if isinstance(g.value, (int, float))]

    n, m = len(pred_nums), len(gt_nums)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0

    # Build N×M relative-distance matrix
    d_matrix = np.ones((n, m), dtype=np.float64)
    for i, pv in enumerate(pred_nums):
        for j, gv in enumerate(gt_nums):
            d_matrix[i, j] = relative_distance(pv, gv, eps=eps)

    # Hungarian matching on distance matrix (minimize cost = minimize distance)
    row_ind, col_ind = linear_sum_assignment(d_matrix)
    total_d = float(np.sum(d_matrix[row_ind, col_ind]))

    # RNSS = 1 - avg relative error, normalized by max(N, M) per DePlot eq. (1)
    return float(1.0 - total_d / max(n, m))


def compute_value_recall_at_5(
    pred_table: Any,
    gt_table: Any,
    tolerance: float = 0.05,
    eps: float = 1e-6,
) -> float:
    """Computes Value-Recall@5%: the proportion of gold values matched within <= 5% error.

    Args:
        pred_table: Predicted table object.
        gt_table: Ground-truth table object.
        tolerance: Relative tolerance threshold (default: 0.05).
        eps: Stability constant.

    Returns:
        float: Recall proportion in [0.0, 1.0].
    """
    preds = _extract_triplets(pred_table)
    gts = _extract_triplets(gt_table)

    gold_nums = [float(g.value) for g in gts if isinstance(g.value, (int, float))]
    pred_nums = [float(p.value) for p in preds if isinstance(p.value, (int, float))]

    if not gold_nums:
        return 1.0 if not pred_nums else 0.0
    if not pred_nums:
        return 0.0

    matched_count = 0
    used_pred_indices = set()

    for g_val in gold_nums:
        denom = max(abs(g_val), eps)
        for p_idx, p_val in enumerate(pred_nums):
            if p_idx in used_pred_indices:
                continue
            if abs(p_val - g_val) / denom <= tolerance:
                matched_count += 1
                used_pred_indices.add(p_idx)
                break

    return float(matched_count / len(gold_nums))
