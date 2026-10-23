"""Comprehensive Test Suite for Week 6 & 7 deliverables (Việt - SV-C).

Covers:
- Task 6.2 (Module B3 + B2): Loading synthetic sanity set from data/sanity/generated/,
  validating metadata shape consistency, and verifying gold-vs-gold = 1.0
  (direct metrics + multi-format round-trip conversions).
- Task 7.1 (Module B4): Real ChartQA dataset loader, auto-detecting png/ vs images/,
  parsing CSV tables into TableSchema, pairing human/augmented QA annotations,
  and executing gold table self-consistency verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_loader import ChartDataLoader, ChartDatasetItem, ChartQAPair, SyntheticSanitySet
from src.format.schema import TableSchema


@pytest.fixture(scope="module")
def sanity_dataset() -> SyntheticSanitySet:
    project_root = Path(__file__).resolve().parent.parent
    generated_dir = project_root / "data" / "sanity" / "generated"
    if not generated_dir.exists():
        pytest.skip(f"Generated sanity directory not found at: {generated_dir}")
    return SyntheticSanitySet.from_generated_dir(generated_dir)


class TestTask62SanitySetVerification:
    """Verifies Task 6.2 requirements on Sa's generated sanity chart set."""

    def test_sanity_dataset_loading_and_counts(self, sanity_dataset: SyntheticSanitySet):
        """Verifies that at least 10 charts per type (bar, grouped_bar, line) are loaded with assets."""
        tables = sanity_dataset.get_tables()
        assert len(tables) >= 30, f"Expected at least 30 sanity charts, found {len(tables)}"

        bar_tables = sanity_dataset.filter_by_chart_type("bar")
        grouped_tables = sanity_dataset.filter_by_chart_type("grouped_bar")
        line_tables = sanity_dataset.filter_by_chart_type("line")

        assert len(bar_tables) >= 10, f"Expected >= 10 bar charts, found {len(bar_tables)}"
        assert len(grouped_tables) >= 10, f"Expected >= 10 grouped bar charts, found {len(grouped_tables)}"
        assert len(line_tables) >= 10, f"Expected >= 10 line charts, found {len(line_tables)}"

        # Verify all items have associated image and CSV paths that actually exist on disk
        for t in tables:
            meta = t.metadata
            assert "image_path" in meta and meta["image_path"] is not None
            assert Path(meta["image_path"]).exists(), f"Image missing: {meta['image_path']}"
            assert "csv_path" in meta and meta["csv_path"] is not None
            assert Path(meta["csv_path"]).exists(), f"CSV missing: {meta['csv_path']}"

            # Check category & series constraints specified in Module B3
            assert 3 <= meta["n_categories"] <= 6
            assert 1 <= meta["n_series"] <= 3

    def test_gold_vs_gold_and_round_trip_metrics(self, sanity_dataset: SyntheticSanitySet):
        """Executes 3-tier gold verification:

        1. Direct self-comparison (RMS-F1 == 1.0, RNSS == 1.0, Value-Recall == 1.0).
        2. Metadata shape consistency (row count == n_categories, cols == n_series + 1).
        3. Round-trip serialization (CSV, Markdown, Linearized DePlot == 1.0).
        """
        report = sanity_dataset.verify_gold_vs_gold(check_round_trip=True)

        assert report["all_passed"] is True, f"Verification failed with failures: {report['failures']}"
        assert report["passed_count"] == report["total_verified"]
        assert len(report["failures"]) == 0

    def test_verify_gold_vs_gold_catches_tampered_table(self):
        """Ensures that verify_gold_vs_gold fails if a table contains inconsistent metadata or corrupt values."""
        # Create a mock sanity set with an intentional corruption
        corrupted_table = TableSchema(
            title="Corrupted Table",
            columns=["Category", "MetricA"],
            rows=[["A", 10.0], ["B", 20.0]],
            metadata={
                "id": "mock_corrupt",
                "chart_type": "bar",
                "n_categories": 5,  # Deliberate mismatch: metadata says 5, actual rows = 2
                "n_series": 1,
            },
        )

        sanity_set = SyntheticSanitySet.__new__(SyntheticSanitySet)
        sanity_set._tables = [corrupted_table]
        sanity_set._samples = []

        report = sanity_set.verify_gold_vs_gold(check_round_trip=False)
        assert report["all_passed"] is False
        assert any(f["check"] == "metadata_row_count" for f in report["failures"])


class TestTask71ChartQALoader:
    """Verifies Task 7.1 real dataset loader for ChartQA / PlotQA."""

    @pytest.fixture
    def mock_chartqa_dir(self, tmp_path: Path) -> Path:
        """Sets up a realistic ChartQA test split fixture with png/, tables/, and QA JSON files."""
        dataset_root = tmp_path / "chartqa"
        test_dir = dataset_root / "test"
        png_dir = test_dir / "png"
        tables_dir = test_dir / "tables"
        png_dir.mkdir(parents=True)
        tables_dir.mkdir(parents=True)

        # 1. Sample 101: Simple bar chart
        (png_dir / "101.png").write_bytes(b"\x89PNG\r\n\x1a\nfakeimage1")
        (tables_dir / "101.csv").write_text("Quarter,Revenue\nQ1,150.5\nQ2,200.0\n", encoding="utf-8")

        # 2. Sample 102: Grouped bar chart
        (png_dir / "102.png").write_bytes(b"\x89PNG\r\n\x1a\nfakeimage2")
        (tables_dir / "102.csv").write_text("Year,SeriesA,SeriesB\n2020,10,20\n2021,30,40\n", encoding="utf-8")

        # 3. Sample 103: Image without gold CSV (edge case)
        (png_dir / "103.png").write_bytes(b"\x89PNG\r\n\x1a\nfakeimage3")

        # 4. QA Annotations: test_human.json
        human_qa = [
            {"imgname": "101.png", "query": "What is the revenue in Q1?", "label": ["150.5"]},
            {"imgname": "102.png", "query": "What is SeriesA in 2020?", "label": "10"},
        ]
        (test_dir / "test_human.json").write_text(json.dumps(human_qa), encoding="utf-8")

        # 5. QA Annotations: test_augmented.json
        augmented_qa = [
            {"imgname": "101.png", "query": "Is Q2 higher than Q1?", "label": ["Yes"]},
            {"imgname": "102.png", "query": "Sum of SeriesB?", "label": ["60"]},
            {"imgname": "103.png", "query": "Question for missing csv?", "label": ["None"]},
        ]
        (test_dir / "test_augmented.json").write_text(json.dumps(augmented_qa), encoding="utf-8")

        return dataset_root

    def test_chartqa_auto_detection_and_parsing(self, mock_chartqa_dir: Path):
        """Verifies auto-detection of 'png/' folder and parsing into TableSchema."""
        loader = ChartDataLoader(data_dir=mock_chartqa_dir, split="test")

        # Must auto-detect png/
        assert loader.images_dir.name == "png"
        assert loader.tables_dir.name == "tables"

        items = loader.scan_dataset()
        assert len(items) == 3

        item101 = loader.get_item_by_id("101")
        assert item101 is not None
        assert item101.table is not None
        assert item101.table.columns == ["quarter", "revenue"]
        assert item101.table.rows == [["Q1", 150.5], ["Q2", 200.0]]
        # 2 rows x 2 cols = 4 cell triplets
        assert len(item101.table.triplets) == 4

    def test_chartqa_qa_pairs_pairing(self, mock_chartqa_dir: Path):
        """Verifies mapping of human and augmented QA annotations to corresponding dataset items."""
        loader = ChartDataLoader(data_dir=mock_chartqa_dir, split="test")
        items = loader.scan_dataset()

        # Item 101 should have 2 QA pairs (1 human, 1 augmented)
        item101 = loader.get_item_by_id("101")
        assert item101 is not None
        assert len(item101.qa_pairs) == 2

        sources = {qa.source for qa in item101.qa_pairs}
        assert "human" in sources
        assert "augmented" in sources

        human_qa = next(qa for qa in item101.qa_pairs if qa.source == "human")
        assert human_qa.query == "What is the revenue in Q1?"
        assert human_qa.label == "150.5"

        augmented_qa = next(qa for qa in item101.qa_pairs if qa.source == "augmented")
        assert augmented_qa.query == "Is Q2 higher than Q1?"
        assert augmented_qa.label == "Yes"

        # Item 103 (image without CSV) should still have its QA pair
        item103 = loader.get_item_by_id("103")
        assert item103 is not None
        assert item103.table is None
        assert len(item103.qa_pairs) == 1
        assert item103.qa_pairs[0].query == "Question for missing csv?"

        # Verify get_all_qa_pairs
        all_pairs = loader.get_all_qa_pairs()
        assert len(all_pairs) == 5

    def test_chartqa_verify_gold_tables(self, mock_chartqa_dir: Path):
        """Verifies self-consistency check on loaded ground-truth CSV tables."""
        loader = ChartDataLoader(data_dir=mock_chartqa_dir, split="test")
        report = loader.verify_gold_tables()

        # 2 tables exist (101 and 102). 103 has no table.
        assert report["total_tables"] == 2
        assert report["valid_count"] == 2
        assert report["all_valid"] is True
        assert len(report["invalid_ids"]) == 0

    def test_loader_backward_compatibility_with_images_dir(self, tmp_path: Path):
        """Verifies that the loader continues to work seamlessly with an 'images/' folder."""
        data_dir = tmp_path / "legacy"
        images_dir = data_dir / "images"
        tables_dir = data_dir / "tables"
        images_dir.mkdir(parents=True)
        tables_dir.mkdir(parents=True)

        (images_dir / "chart_a.png").write_bytes(b"dummy")
        (tables_dir / "chart_a.csv").write_text("Col1,Col2\nA,1\n", encoding="utf-8")

        loader = ChartDataLoader(data_dir=data_dir, split=".")
        items = loader.scan_dataset()

        assert len(items) == 1
        assert items[0].id == "chart_a"
        assert items[0].table is not None
        assert items[0].table.columns == ["col1", "col2"]

