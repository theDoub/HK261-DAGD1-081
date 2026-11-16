"""Compare paired Gate 2 reports without changing their scoring semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


COMPARISON_CONDITION_KEYS = (
    "model_name_or_path",
    "device_requested",
    "device_used",
    "quantization",
    "max_new_tokens",
)
REQUIRED_CONDITION_KEYS = COMPARISON_CONDITION_KEYS + ("prompt", "sample_ids")


def _read_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict) or not isinstance(report.get("records"), list):
        raise ValueError(f"Invalid Gate 2 report: {path}")
    return report


def _error_counts(report: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in report["records"]:
        category = record.get("error_category", "unknown")
        counts[category] = counts.get(category, 0) + 1
    return counts


def compare_gate2_reports(baseline_path: Path, candidate_path: Path) -> Dict[str, Any]:
    """Compare metrics and per-sample outcomes for the same inference conditions."""
    baseline = _read_report(baseline_path)
    candidate = _read_report(candidate_path)
    baseline_records = {record["sample_id"]: record for record in baseline["records"]}
    candidate_records = {record["sample_id"]: record for record in candidate["records"]}

    if len(baseline_records) != len(baseline["records"]):
        raise ValueError("Baseline report contains duplicate sample IDs")
    if len(candidate_records) != len(candidate["records"]):
        raise ValueError("Candidate report contains duplicate sample IDs")

    baseline_ids = set(baseline_records)
    candidate_ids = set(candidate_records)
    if baseline_ids != candidate_ids:
        missing_from_candidate = sorted(baseline_ids - candidate_ids)
        missing_from_baseline = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "Reports do not contain the same sample IDs "
            f"(missing from candidate: {missing_from_candidate}; "
            f"missing from baseline: {missing_from_baseline})"
        )
    baseline_order = [record["sample_id"] for record in baseline["records"]]
    candidate_order = [record["sample_id"] for record in candidate["records"]]
    if baseline_order != candidate_order:
        raise ValueError("Reports contain the same sample IDs but in a different order")

    conditions_available = isinstance(baseline.get("conditions"), dict) and isinstance(
        candidate.get("conditions"), dict
    )
    condition_differences: Dict[str, Dict[str, Any]] = {}
    if conditions_available:
        for key in COMPARISON_CONDITION_KEYS:
            baseline_value = baseline["conditions"].get(key)
            candidate_value = candidate["conditions"].get(key)
            if baseline_value != candidate_value:
                condition_differences[key] = {
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                }
        if baseline.get("backend") != candidate.get("backend"):
            condition_differences["backend"] = {
                "baseline": baseline.get("backend"),
                "candidate": candidate.get("backend"),
            }
        if baseline.get("dataset_path") != candidate.get("dataset_path"):
            condition_differences["dataset_path"] = {
                "baseline": baseline.get("dataset_path"),
                "candidate": candidate.get("dataset_path"),
            }
        if baseline["conditions"].get("sample_ids") != baseline_order:
            condition_differences["baseline_sample_ids_metadata"] = {
                "record_order": baseline_order,
                "conditions": baseline["conditions"].get("sample_ids"),
            }
        if candidate["conditions"].get("sample_ids") != candidate_order:
            condition_differences["candidate_sample_ids_metadata"] = {
                "record_order": candidate_order,
                "conditions": candidate["conditions"].get("sample_ids"),
            }
        if condition_differences:
            raise ValueError(f"Inference conditions differ between reports: {condition_differences}")

    conditions_verified = conditions_available and all(
        key in baseline["conditions"] and key in candidate["conditions"]
        for key in REQUIRED_CONDITION_KEYS
    )

    ordered_ids: Sequence[str] = baseline_order
    per_sample: List[Dict[str, Any]] = []
    for sample_id in ordered_ids:
        old = baseline_records[sample_id]
        new = candidate_records[sample_id]
        per_sample.append(
            {
                "sample_id": sample_id,
                "baseline_rms_f1": old.get("rms_f1", 0.0),
                "candidate_rms_f1": new.get("rms_f1", 0.0),
                "rms_f1_delta": round(new.get("rms_f1", 0.0) - old.get("rms_f1", 0.0), 4),
                "baseline_parse_ok": old.get("parse_ok", False),
                "candidate_parse_ok": new.get("parse_ok", False),
                "baseline_error_category": old.get("error_category"),
                "candidate_error_category": new.get("error_category"),
            }
        )

    baseline_errors = _error_counts(baseline)
    candidate_errors = _error_counts(candidate)
    sample_count = len(ordered_ids)
    baseline_header_mismatches = baseline_errors.get("header_mismatch", 0)
    candidate_header_mismatches = candidate_errors.get("header_mismatch", 0)
    error_category_deltas = {
        category: candidate_errors.get(category, 0) - baseline_errors.get(category, 0)
        for category in sorted(set(baseline_errors) | set(candidate_errors))
    }

    return {
        "paired_samples": sample_count,
        "sample_ids": list(ordered_ids),
        "conditions_verified": conditions_verified,
        "prompt_changed": (
            baseline["conditions"].get("prompt") != candidate["conditions"].get("prompt")
            if conditions_available
            else None
        ),
        "condition_verification_warning": (
            None
            if conditions_verified
            else "One or both reports lack complete inference-condition metadata."
        ),
        "condition_differences": condition_differences,
        "baseline": {
            "header_mismatches": baseline_header_mismatches,
            "header_mismatch_rate": round(baseline_header_mismatches / sample_count, 4) if sample_count else 0.0,
            "mean_rms_f1": baseline.get("mean_rms_f1", 0.0),
            "parse_success_rate": baseline.get("parse_success_rate", 0.0),
            "error_distribution": baseline_errors,
        },
        "candidate": {
            "header_mismatches": candidate_header_mismatches,
            "header_mismatch_rate": round(candidate_header_mismatches / sample_count, 4) if sample_count else 0.0,
            "mean_rms_f1": candidate.get("mean_rms_f1", 0.0),
            "parse_success_rate": candidate.get("parse_success_rate", 0.0),
            "error_distribution": candidate_errors,
        },
        "header_mismatch_delta": candidate_header_mismatches - baseline_header_mismatches,
        "mean_rms_f1_delta": round(candidate.get("mean_rms_f1", 0.0) - baseline.get("mean_rms_f1", 0.0), 4),
        "parse_success_rate_delta": round(
            candidate.get("parse_success_rate", 0.0) - baseline.get("parse_success_rate", 0.0), 2
        ),
        "error_category_deltas": error_category_deltas,
        "per_sample": per_sample,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare paired Gate 2 evaluation reports.")
    parser.add_argument("--baseline", required=True, type=Path, help="Frozen baseline evaluation JSON")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate prompt evaluation JSON")
    parser.add_argument("--output", type=Path, help="Optional path for the comparison JSON")
    args = parser.parse_args(argv)

    comparison = compare_gate2_reports(args.baseline, args.candidate)
    serialized = json.dumps(comparison, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
