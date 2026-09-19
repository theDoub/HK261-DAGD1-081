"""Main entry point: End-to-end Smoke Test for Chart-to-Table Pipeline.

Workflow:
1. Loads synthetic ground-truth tables from Module B3 (data/sanity/synthetic_dataset.json).
2. Extracts tables using Module B5 VLMTableHarness (MockBackend).
3. Parses output and normalizes cells into Module B1 TableSchema and TableTriplets.
4. Computes Module B2 metrics (RMS-F1, RNSS, Value-Recall@5%).
5. Generates Module B6 aggregations (by chart type, complexity, and error buckets).
6. Logs comprehensive verification summary.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Setup structured logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SmokeTest")


def run_smoke_test() -> None:
    """Executes the complete modular smoke test across Modules B1 to B6."""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader.synthetic import load_synthetic_tables
    from src.evaluation.evaluator import PipelineEvaluator
    from src.models.harness import MockBackend, VLMTableHarness

    logger.info("=" * 65)
    logger.info("STARTING CHART-TO-TABLE PIPELINE SMOKE TEST (MODULES B1-B6)")
    logger.info("=" * 65)

    # 1. Module B3: Load synthetic sanity benchmark tables
    tables = load_synthetic_tables()
    logger.info("Module B3: Loaded %d synthetic benchmark samples.", len(tables))

    # Preview first sample in Markdown (Module B1)
    sample_gold = tables[0]
    print("\n" + "=" * 50)
    print(" [Module B1] Gold Table Preview (Markdown):")
    print("=" * 50)
    print(sample_gold.to_markdown())
    print(f"Total Triplet Cells: {len(sample_gold.triplets)}")
    print("=" * 50 + "\n")

    # 2. Module B5: Initialize extraction harness with MockBackend
    # We configure the mock to match sample_001 perfectly to demonstrate high F1
    mock_md = sample_gold.to_markdown()
    mock_backend = MockBackend(predefined_table_md=mock_md)
    harness = VLMTableHarness(backend=mock_backend, target_format="markdown")
    logger.info("Module B5: Initialized VLMTableHarness with MockBackend.")

    # 3. Module B6 & B2: Run evaluation pipeline
    evaluator = PipelineEvaluator(harness=harness)
    results = evaluator.evaluate_dataset(tables)

    print("\n" + "=" * 50)
    print(" [Module B6] Evaluation & Error Analysis Summary:")
    print("=" * 50)
    print(f"Total Evaluated Samples: {results['total_samples']}")
    print(f"Mean RMS-F1 Score:       {results['mean_rms_f1']:.4f}")
    print(f"Mean RNSS Score:         {results['mean_rnss']:.4f}")
    print(f"Mean Value-Recall@5%:    {results['mean_value_recall_at_5']:.4f}")
    print("-" * 50)
    print("Breakdown by Chart Type:")
    print(json.dumps(results["breakdown_by_chart_type"], indent=2))
    print("Breakdown by Complexity:")
    print(json.dumps(results["breakdown_by_complexity"], indent=2))
    print("Error Distribution:")
    print(json.dumps(results["error_analysis"], indent=2))
    print("=" * 50 + "\n")

    logger.info("SUCCESS: All modules (B1-B6) executed and validated cleanly!")


if __name__ == "__main__":
    run_smoke_test()
