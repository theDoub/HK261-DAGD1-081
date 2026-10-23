"""Module B3: Synthetic Sanity Set Loader.

Provides utilities for loading synthetic mock chart metadata and ground-truth
tables (bar, line, grouped bar) across varied category counts, series counts, and complexity levels,
as well as verifying gold table self-consistency (gold-vs-gold = 1.0) and round-trip format preservation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.format.formatter import to_csv, to_linearized_deplot, to_markdown
from src.format.parser import parse_csv_table, parse_deplot_linearized, parse_markdown_table
from src.format.schema import TableSchema
from src.metrics.table_metrics import (
    compute_rms_f1,
    compute_rnss,
    compute_value_recall_at_5,
)


class SyntheticSanitySet:
    """Manages synthetic benchmark fixtures for rapid smoke testing and sanity verification."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None) -> None:
        """Initializes the loader with the path to synthetic dataset JSON or generated directory."""
        project_root = Path(__file__).resolve().parent.parent.parent
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Default to data/sanity/synthetic_dataset.json relative to project root
            self.data_path = project_root / "data" / "sanity" / "synthetic_dataset.json"

        self._samples: List[Dict[str, Any]] = []
        self._tables: List[TableSchema] = []
        self._load()

    @classmethod
    def from_generated_dir(cls, dir_path: Optional[Union[str, Path]] = None) -> SyntheticSanitySet:
        """Factory method loading all samples from data/sanity/generated/ directory."""
        if dir_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            dir_path = project_root / "data" / "sanity" / "generated"
        return cls(data_path=dir_path)

    def _load(self) -> None:
        """Loads dataset from either a single JSON catalog or a directory of generated samples."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Synthetic dataset fixture not found at: {self.data_path}")

        self._samples = []
        self._tables = []

        if self.data_path.is_dir():
            # Scan for all individual metadata JSON files in the generated tree
            json_files = sorted(list(self.data_path.rglob("*.json")))
            for j_path in json_files:
                try:
                    with open(j_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    continue

                if not isinstance(meta, dict) or "gold_table" not in meta:
                    continue

                sample_id = meta.get("id", j_path.stem)
                csv_path = j_path.parent / meta.get("gold_table", f"{sample_id}.csv")
                png_path = j_path.parent / meta.get("image", f"{sample_id}.png")

                if not csv_path.exists():
                    continue

                with open(csv_path, "r", encoding="utf-8") as f:
                    csv_content = f.read()

                table = parse_csv_table(csv_content)
                table.title = sample_id

                # Attach comprehensive metadata
                table.metadata = {
                    "id": sample_id,
                    "chart_type": meta.get("chart_type", "unknown"),
                    "complexity": meta.get("complexity", "unknown"),
                    "n_categories": meta.get("n_categories", len(table.rows)),
                    "n_series": meta.get("n_series", max(0, len(table.columns) - 1)),
                    "dpi": meta.get("dpi"),
                    "image_path": str(png_path) if png_path.exists() else None,
                    "csv_path": str(csv_path),
                    "json_path": str(j_path),
                    **meta,
                }

                self._tables.append(table)
                self._samples.append(
                    {
                        "id": sample_id,
                        "title": table.title,
                        "columns": table.columns,
                        "rows": table.rows,
                        "metadata": table.metadata,
                    }
                )
        else:
            # Single JSON catalog format (e.g. data/sanity/synthetic_dataset.json)
            with open(self.data_path, "r", encoding="utf-8") as f:
                self._samples = json.load(f)

            for s in self._samples:
                table = TableSchema(
                    title=s.get("title", ""),
                    columns=s.get("columns", []),
                    rows=s.get("rows", []),
                    metadata={
                        "id": s.get("id"),
                        "chart_type": s.get("chart_type"),
                        "complexity": s.get("complexity"),
                        **s.get("metadata", {}),
                    },
                )
                self._tables.append(table)

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Returns all raw sample dictionaries."""
        return list(self._samples)

    def get_tables(self) -> List[TableSchema]:
        """Returns all parsed TableSchema instances with triplets and metadata.

        Returns:
            List[TableSchema]: Parsed table structures.
        """
        return list(self._tables)

    def filter_by_complexity(self, complexity: str) -> List[TableSchema]:
        """Filters synthetic tables by complexity level ('simple', 'medium', 'complex')."""
        return [t for t in self.get_tables() if t.metadata.get("complexity") == complexity]

    def filter_by_chart_type(self, chart_type: str) -> List[TableSchema]:
        """Filters synthetic tables by chart type ('bar', 'line', 'grouped_bar', 'horizontal_bar')."""
        return [t for t in self.get_tables() if t.metadata.get("chart_type") == chart_type]

    def verify_gold_vs_gold(self, check_round_trip: bool = True) -> Dict[str, Any]:
        """Verifies gold-vs-gold = 1.0 and format round-trip preservation across all tables.

        Evaluation tiers:
            1. Direct self-comparison: compute_rms_f1(gold, gold) == 1.0,
               compute_rnss(gold, gold) == 1.0, compute_value_recall_at_5(gold, gold) == 1.0.
            2. Metadata consistency (if n_categories/n_series are recorded):
               Number of data rows == n_categories, number of columns == n_series + 1.
            3. Round-trip preservation (Markdown, CSV, Linearized DePlot):
               gold -> serialized text -> parsed table -> compute_rms_f1(restored, gold) == 1.0.

        Args:
            check_round_trip: Whether to test format serialization and re-parsing.

        Returns:
            Dict[str, Any]: Verification summary detailing test counts and any failures.
        """
        tables = self.get_tables()
        failures: List[Dict[str, Any]] = []

        for table in tables:
            sample_id = table.metadata.get("id", "unknown")

            # Tier 1: Direct self-comparison
            rms = compute_rms_f1(table, table)
            rnss = compute_rnss(table, table)
            recall = compute_value_recall_at_5(table, table)

            if abs(rms["f1"] - 1.0) > 1e-6:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "check": "direct_rms_f1",
                        "expected": 1.0,
                        "actual": rms["f1"],
                    }
                )

            if abs(rnss - 1.0) > 1e-6:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "check": "direct_rnss",
                        "expected": 1.0,
                        "actual": rnss,
                    }
                )

            if abs(recall - 1.0) > 1e-6:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "check": "direct_value_recall",
                        "expected": 1.0,
                        "actual": recall,
                    }
                )

            # Tier 2: Metadata shape consistency
            if "n_categories" in table.metadata:
                expected_rows = table.metadata["n_categories"]
                if len(table.rows) != expected_rows:
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "check": "metadata_row_count",
                            "expected": expected_rows,
                            "actual": len(table.rows),
                        }
                    )

            if "n_series" in table.metadata:
                expected_series = table.metadata["n_series"]
                # Column 0 is the category/label column; data columns are len(columns) - 1
                actual_series = max(0, len(table.columns) - 1)
                if actual_series != expected_series:
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "check": "metadata_series_count",
                            "expected": expected_series,
                            "actual": actual_series,
                        }
                    )

            # Tier 3: Format Round-Trip
            if check_round_trip:
                # 3a. CSV round-trip
                csv_str = to_csv(table)
                restored_csv = parse_csv_table(csv_str)
                f1_csv = compute_rms_f1(restored_csv, table)["f1"]
                if abs(f1_csv - 1.0) > 1e-6:
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "check": "round_trip_csv",
                            "expected": 1.0,
                            "actual": f1_csv,
                        }
                    )

                # 3b. Markdown round-trip
                md_str = to_markdown(table)
                restored_md = parse_markdown_table(md_str)
                f1_md = compute_rms_f1(restored_md, table)["f1"]
                if abs(f1_md - 1.0) > 1e-6:
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "check": "round_trip_markdown",
                            "expected": 1.0,
                            "actual": f1_md,
                        }
                    )

                # 3c. Linearized DePlot round-trip
                lin_str = to_linearized_deplot(table)
                restored_lin = parse_deplot_linearized(lin_str)
                f1_lin = compute_rms_f1(restored_lin, table)["f1"]
                if abs(f1_lin - 1.0) > 1e-6:
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "check": "round_trip_linearized",
                            "expected": 1.0,
                            "actual": f1_lin,
                        }
                    )

        return {
            "total_verified": len(tables),
            "passed_count": len(tables) - len(failures),
            "all_passed": len(failures) == 0,
            "failures": failures,
        }


def load_synthetic_tables(data_path: Optional[Union[str, Path]] = None) -> List[TableSchema]:
    """Helper function to directly load all synthetic TableSchema instances."""
    return SyntheticSanitySet(data_path).get_tables()
