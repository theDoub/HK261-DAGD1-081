"""Module B4: Dataset Loader Stubs for ChartQA and PlotQA.

Provides structured data ingestion interfaces for multimodal chart question-answering
benchmarks (ChartQA, PlotQA) and local sanity datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from src.format.schema import TableSchema


@dataclass
class ChartDatasetItem:
    """Represents a single multimodal sample in the pipeline.

    Attributes:
        id (str): Unique sample identifier.
        image_path (Optional[Path]): Path to the chart image file (PNG/JPG).
        question (str): Question prompt targeting information in the chart.
        answer (Optional[str]): Ground truth answer string or number.
        table (Optional[TableSchema]): Ground truth underlying table representation if available.
        metadata (Dict[str, Any]): Additional dataset properties (e.g. chart type, split, source).
    """

    id: str
    question: str
    image_path: Optional[Path] = None
    answer: Optional[str] = None
    table: Optional[TableSchema] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChartDataLoader:
    """Dataset loader stub for ChartQA, PlotQA, and local sanity datasets.

    Handles loading annotations, pairing image paths with tabular data,
    and batching samples for two-stage model evaluation.
    """

    def __init__(
        self,
        dataset_name: str = "ChartQA",
        data_dir: Union[str, Path] = "data",
        split: str = "test",
    ) -> None:
        """Initializes the loader with target dataset configuration.

        Args:
            dataset_name: Name of the dataset ('ChartQA', 'PlotQA', or 'sanity').
            data_dir: Root directory for data storage.
            split: Dataset partition ('train', 'val', 'test', 'sanity').
        """
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.split = split
        self._samples: List[ChartDatasetItem] = []

    def load_sanity_data(self, sanity_json_path: Optional[Union[str, Path]] = None) -> List[ChartDatasetItem]:
        """Loads mock table and QA sample from data/sanity for quick smoke testing.

        Args:
            sanity_json_path: Optional path to sanity JSON file. Defaults to data/sanity/sample_table.json.

        Returns:
            List[ChartDatasetItem]: List containing the sanity dataset items.
        """
        path = Path(sanity_json_path) if sanity_json_path else self.data_dir / "sanity" / "sample_table.json"

        if not path.exists():
            raise FileNotFoundError(f"Sanity data file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        table = TableSchema.from_dict(data)

        item = ChartDatasetItem(
            id=meta.get("image_id", "sanity_001"),
            question=meta.get("question", "What is shown in the table?"),
            answer=meta.get("ground_truth_answer", "N/A"),
            table=table,
            image_path=path.parent / meta.get("image_id", "mock_chart.png"),
            metadata=meta,
        )
        self._samples = [item]
        return self._samples

    def load_dataset(self) -> List[ChartDatasetItem]:
        """Loads dataset items from raw directory stub or falls back to sanity data.

        Returns:
            List[ChartDatasetItem]: Loaded dataset items.
        """
        raw_split_dir = self.data_dir / "raw" / self.dataset_name.lower() / self.split
        if not raw_split_dir.exists():
            # In skeleton/stub mode, gracefully fallback to sanity data
            return self.load_sanity_data()

        # Placeholder for full ChartQA/PlotQA json parsing logic
        # Typically annotations are stored as json files:
        # e.g. test_augmented.json, test_human.json
        items: List[ChartDatasetItem] = []
        return items

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[ChartDatasetItem]:
        return iter(self._samples)
