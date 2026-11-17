import json
from pathlib import Path

import pytest

from scripts.compare_gate2_reports import compare_gate2_reports


def test_compare_gate2_reports_pairs_samples_and_detects_metric_changes(tmp_path: Path):
    conditions = {
        "model_name_or_path": "model",
        "device_requested": "cuda",
        "device_used": "cuda",
        "quantization": "4bit",
        "max_new_tokens": 256,
        "prompt": "baseline prompt",
        "sample_ids": ["a", "b"],
    }
    baseline = {
        "backend": "qwen",
        "dataset_path": "dataset",
        "conditions": conditions,
        "mean_rms_f1": 0.6,
        "parse_success_rate": 100.0,
        "records": [
            {"sample_id": "a", "rms_f1": 0.4, "parse_ok": True, "error_category": "header_mismatch"},
            {"sample_id": "b", "rms_f1": 0.8, "parse_ok": True, "error_category": "correct"},
        ],
    }
    candidate = json.loads(json.dumps(baseline))
    candidate["conditions"]["prompt"] = "candidate prompt"
    candidate["records"][0]["error_category"] = "correct"
    candidate["records"][0]["rms_f1"] = 0.7
    candidate["mean_rms_f1"] = 0.75
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    comparison = compare_gate2_reports(baseline_path, candidate_path)

    assert comparison["conditions_verified"] is True
    assert comparison["prompt_changed"] is True
    assert comparison["paired_samples"] == 2
    assert comparison["header_mismatch_delta"] == -1
    assert comparison["mean_rms_f1_delta"] == 0.15
    assert comparison["error_category_deltas"]["header_mismatch"] == -1
    assert comparison["per_sample"][0]["rms_f1_delta"] == 0.3


def test_compare_gate2_reports_rejects_different_sample_sets(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps({"records": [{"sample_id": "a"}]}), encoding="utf-8")
    candidate_path.write_text(json.dumps({"records": [{"sample_id": "b"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="same sample IDs"):
        compare_gate2_reports(baseline_path, candidate_path)


def test_compare_gate2_reports_rejects_different_inference_settings(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline = {
        "backend": "qwen",
        "dataset_path": "dataset",
        "conditions": {"model_name_or_path": "model-a"},
        "records": [{"sample_id": "a"}],
    }
    candidate = json.loads(json.dumps(baseline))
    candidate["conditions"]["model_name_or_path"] = "model-b"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(ValueError, match="conditions differ"):
        compare_gate2_reports(baseline_path, candidate_path)
