"""Comprehensive Unit Tests for Modules B1 to B6."""

from pathlib import Path
import pytest

from src.data_loader.loader import ChartDataLoader
from src.data_loader.synthetic import SyntheticSanitySet, load_synthetic_tables
from src.evaluation.evaluator import PipelineEvaluator, categorize_error
from src.format.parser import (
    normalize_header,
    normalize_value,
    parse_csv_table,
    parse_deplot_linearized,
    parse_markdown_table,
    parse_table_text,
    strip_markdown_codeblocks,
)
from src.format.schema import TableSchema, TableTriplet
from src.metrics.table_metrics import (
    compute_rms_f1,
    compute_rnss,
    compute_value_recall_at_5,
    normalized_levenshtein,
)
from src.models.harness import MockBackend, VLMTableHarness


# ---------------------------------------------------------
# Module B1 Tests
# ---------------------------------------------------------

def test_normalization():
    """Verifies header and value normalization rules."""
    # Header normalization
    assert normalize_header("  Quarterly   Sales  (USD) ") == "quarterly sales (usd)"
    assert normalize_header("Product\nName") == "product name"

    # Value normalization
    assert normalize_value(" $1,250.50 ") == 1250.50
    assert normalize_value("45.2%") == 45.2
    assert normalize_value("1,000") == 1000
    assert normalize_value("N/A") is None
    assert normalize_value("Category Alpha") == "Category Alpha"


def test_codeblock_stripping():
    """Verifies stripping markdown code fences and conversational preamble."""
    raw = (
        "Here is the result you requested:\n\n"
        "```markdown\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "```\n"
        "Hope this helps!"
    )
    stripped = strip_markdown_codeblocks(raw)
    assert stripped == "| A | B |\n|---|---|\n| 1 | 2 |"


def test_markdown_and_csv_parsers():
    """Verifies parsing markdown and CSV texts into TableSchema with triplets."""
    md_text = (
        "### Revenue\n"
        "| Region | Q1 | Q2 |\n"
        "|---|---|---|\n"
        "| North | $100 | $120 |\n"
        "| South | $80 | $90 |"
    )
    table_md = parse_markdown_table(md_text)
    assert table_md.title == "Revenue"
    assert table_md.columns == ["region", "q1", "q2"]
    assert table_md.rows[0] == ["North", 100, 120]
    assert len(table_md.triplets) == 6
    assert isinstance(table_md.triplets[0], TableTriplet)

    csv_text = "Year,Rate\n2021,5.2%\n2022,6.1%"
    table_csv = parse_csv_table(csv_text)
    assert table_csv.columns == ["year", "rate"]
    assert table_csv.rows[0] == [2021, 5.2]


def test_deplot_linearized_parser():
    """Verifies parsing DePlot output with <0x0A> and TITLE tokens."""
    deplot_text = "TITLE | Energy<0x0A>Type | Amount<0x0A>Solar | 150<0x0A>Wind | 200"
    table = parse_deplot_linearized(deplot_text)
    assert table.title == "Energy"
    assert table.columns == ["type", "amount"]
    assert table.rows[0] == ["Solar", 150]
    assert table.rows[1] == ["Wind", 200]


def test_triplet_reconstruction():
    """Verifies converting to and from TableTriplets."""
    triplets = [
        TableTriplet(row=0, col="Product", value="Laptop"),
        TableTriplet(row=0, col="Price", value=1200),
        TableTriplet(row=1, col="Product", value="Phone"),
        TableTriplet(row=1, col="Price", value=800),
    ]
    reconstructed = TableSchema.from_triplets(triplets, title="Inventory")
    assert reconstructed.columns == ["Product", "Price"]
    assert len(reconstructed.rows) == 2
    assert reconstructed.rows[0] == ["Laptop", 1200]


# ---------------------------------------------------------
# Module B2 Tests
# ---------------------------------------------------------

def test_metrics_computation():
    """Verifies RMS-F1, RNSS, and Value-Recall@5%."""
    t_gold = TableSchema(
        columns=["quarter", "sales"],
        rows=[["Q1", 100.0], ["Q2", 200.0]],
    )
    t_pred = TableSchema(
        columns=["quarter", "sales"],
        rows=[["Q1", 102.0], ["Q2", 195.0]],  # both within 5% error
    )

    rms = compute_rms_f1(t_pred, t_gold)
    assert rms["f1"] > 0.95
    assert rms["precision"] > 0.95
    assert rms["recall"] > 0.95

    rnss = compute_rnss(t_pred, t_gold)
    assert rnss > 0.95

    val_recall = compute_value_recall_at_5(t_pred, t_gold, tolerance=0.05)
    assert val_recall == 1.0


# ---------------------------------------------------------
# Module B3 & B4 Tests
# ---------------------------------------------------------

def test_synthetic_sanity_set():
    """Verifies loading synthetic mock tables."""
    dataset = SyntheticSanitySet()
    tables = dataset.get_tables()
    assert len(tables) >= 5

    simple_tables = dataset.filter_by_complexity("simple")
    assert len(simple_tables) > 0

    bar_tables = dataset.filter_by_chart_type("bar")
    assert len(bar_tables) > 0


def test_real_data_loader_missing_csv(tmp_path):
    """Verifies error handling when a chart image lacks a corresponding CSV table."""
    loader = ChartDataLoader(data_dir=tmp_path)
    # Non-existent CSV should log an error and return None safely
    table = loader.load_table_from_csv(tmp_path / "missing.csv")
    assert table is None


# ---------------------------------------------------------
# Module B5 & B6 Tests
# ---------------------------------------------------------

def test_vlm_table_harness():
    """Verifies VLMTableHarness extraction and failure isolation."""
    mock_table = "| Item | Value |\n|---|---|\n| A | 10 |"
    harness = VLMTableHarness(backend=MockBackend(mock_table))
    res = harness.extract_table()

    assert res.parse_ok is True
    assert res.table is not None
    assert res.table.columns == ["item", "value"]


def test_evaluator_and_error_categorization():
    """Verifies pipeline evaluation aggregations and error categorization."""
    gt = TableSchema(columns=["a", "b"], rows=[[1, 2], [3, 4]])

    # Test error categorization buckets
    assert categorize_error(None, gt, 0.0) == "missing_data"

    pred_wrong_headers = TableSchema(columns=["x", "y"], rows=[[1, 2]])
    assert categorize_error(pred_wrong_headers, gt, 0.0) == "header_mismatch"

    pred_wrong_shape = TableSchema(columns=["a", "b"], rows=[[1, 2]])
    assert categorize_error(pred_wrong_shape, gt, 0.5) == "structural_error"

    pred_good = TableSchema(columns=["a", "b"], rows=[[1, 2], [3, 4]])
    assert categorize_error(pred_good, gt, 1.0) == "none"

    # Test full evaluator run
    evaluator = PipelineEvaluator()
    summary = evaluator.evaluate_dataset([gt])
    assert "mean_rms_f1" in summary
    assert "breakdown_by_chart_type" in summary
    assert "error_analysis" in summary
