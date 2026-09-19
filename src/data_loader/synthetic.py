"""Module B3: Synthetic Sanity Set Loader.

Provides utilities for loading small synthetic mock chart metadata and ground-truth
tables (bar, line, grouped bar) across varied category counts, series counts, and complexity levels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.format.schema import TableSchema


class SyntheticSanitySet:
    """Manages synthetic benchmark fixtures for rapid smoke testing and sanity verification."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None) -> None:
        """Initializes the loader with the path to synthetic dataset JSON."""
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Default to data/sanity/synthetic_dataset.json relative to project root
            self.data_path = Path(__file__).resolve().parent.parent.parent / "data" / "sanity" / "synthetic_dataset.json"

        self._samples: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Loads dataset from JSON."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Synthetic dataset fixture not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self._samples = json.load(f)

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Returns all raw sample dictionaries."""
        return list(self._samples)

    def get_tables(self) -> List[TableSchema]:
        """Converts all synthetic samples into TableSchema instances.

        Returns:
            List[TableSchema]: Parsed table structures with triplets and metadata.
        """
        tables: List[TableSchema] = []
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
            tables.append(table)
        return tables

    def filter_by_complexity(self, complexity: str) -> List[TableSchema]:
        """Filters synthetic tables by complexity level ('simple', 'medium', 'complex')."""
        return [t for t in self.get_tables() if t.metadata.get("complexity") == complexity]

    def filter_by_chart_type(self, chart_type: str) -> List[TableSchema]:
        """Filters synthetic tables by chart type ('bar', 'line', 'grouped_bar', 'horizontal_bar')."""
        return [t for t in self.get_tables() if t.metadata.get("chart_type") == chart_type]


def load_synthetic_tables(data_path: Optional[Union[str, Path]] = None) -> List[TableSchema]:
    """Helper function to directly load all synthetic TableSchema instances."""
    return SyntheticSanitySet(data_path).get_tables()
