"""Main entry point for the Chart-to-Table -> Table QA pipeline.

Demonstrates a minimal runnable workflow:
1. Loads mock chart data from data/sanity/sample_table.json.
2. Formats and validates the table using Module B1 (src.format.schema).
3. Demonstrates conversions (Markdown, DataFrame, Linearized DePlot format).
4. Executes the two-stage evaluation harness (Module B6) on the sanity sample.
5. Prints a success confirmation log.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Configure structured console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ChartQA-Pipeline")


def run_sanity_pipeline() -> None:
    """Executes the sanity check pipeline."""
    # Ensure root path is in sys.path when running script directly
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader.loader import ChartDataLoader
    from src.evaluation.evaluator import PipelineEvaluator
    from src.format.schema import TableSchema
    from src.models.deplot import DePlotHarness
    from src.models.vlm_client import VLMClient

    logger.info("=" * 60)
    logger.info("Initializing Chart-to-Table -> Table QA Sanity Pipeline")
    logger.info("=" * 60)

    # 1. Load mock data from data/sanity
    sanity_file = project_root / "data" / "sanity" / "sample_table.json"
    logger.info("Loading sanity dataset from: %s", sanity_file)

    loader = ChartDataLoader(data_dir=project_root / "data")
    dataset_items = loader.load_sanity_data(sanity_file)
    sample = dataset_items[0]

    logger.info("Loaded sample ID: %s", sample.id)
    logger.info("Question: '%s'", sample.question)
    logger.info("Ground Truth Answer: '%s'", sample.answer)

    # 2. Validate and format using Module B1 (TableSchema)
    table: TableSchema = sample.table  # type: ignore[assignment]
    is_valid = table.validate_shape()
    logger.info(
        "TableSchema validation: %s (Columns: %d, Rows: %d)",
        "PASSED" if is_valid else "FAILED",
        len(table.columns),
        len(table.rows),
    )

    # 3. Display Markdown formatted table
    print("\n--- [Module B1] Formatted Markdown Table ---")
    print(table.to_markdown())
    print("-" * 45 + "\n")

    # 4. Display Pandas DataFrame conversion
    print("--- [Module B1] Converted pandas DataFrame ---")
    df = table.to_dataframe()
    print(df)
    print("-" * 45 + "\n")

    # 5. Display Linearized Text (DePlot / LLM prompt format)
    print("--- [Module B1] Linearized Table for Prompt Context ---")
    linearized = table.to_linearized_text()
    print(linearized)
    print("-" * 45 + "\n")

    # 6. Execute two-stage evaluation pipeline (Module B6)
    logger.info("Executing two-stage evaluation pipeline stub...")
    evaluator = PipelineEvaluator(
        deplot_harness=DePlotHarness(dry_run=True),
        vlm_client=VLMClient(provider="vlm_stub"),
    )
    summary = evaluator.evaluate_dataset(dataset_items)

    print("\n--- [Module B6] Evaluation Results Summary ---")
    print(json.dumps(summary, indent=2))
    print("-" * 45 + "\n")

    logger.info("SUCCESS: Chart-to-Table -> Table QA pipeline sanity check completed cleanly!")


if __name__ == "__main__":
    run_sanity_pipeline()
