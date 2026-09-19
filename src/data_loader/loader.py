"""Module B4: Real Dataset Loader for ChartQA / PlotQA.

Provides structured ingestion pairing chart images with corresponding gold CSV tables,
parsing them into Module B1 TableSchema representations with robust error logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from src.format.parser import parse_csv_table
from src.format.schema import TableSchema

logger = logging.getLogger(__name__)


@dataclass
class ChartDatasetItem:
    """Represents a multimodal chart sample in the evaluation pipeline.

    Attributes:
        id (str): Unique sample identifier (e.g. image filename or stem).
        image_path: Path to the chart image (PNG/JPG).
        table (Optional[TableSchema]): Parsed ground-truth table representation.
        csv_path (Optional[Path]): Path to the raw gold CSV table.
        metadata (Dict[str, Any]): Additional dataset properties (chart type, split, source).
    """

    id: str
    image_path: Optional[Path] = None
    table: Optional[TableSchema] = None
    csv_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChartDataLoader:
    """Dataset loader pairing chart images with corresponding ground-truth CSV tables."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/raw",
        split: str = "test",
        images_dirname: str = "images",
        tables_dirname: str = "tables",
    ) -> None:
        """Initializes the loader.

        Args:
            data_dir: Root dataset folder (e.g. data/raw/chartqa).
            split: Partition ('train', 'val', 'test').
            images_dirname: Subfolder name for chart images.
            tables_dirname: Subfolder name for gold CSV tables.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split if (self.data_dir / split).exists() else self.data_dir
        self.images_dir = self.split_dir / images_dirname
        self.tables_dir = self.split_dir / tables_dirname
        self._items: List[ChartDatasetItem] = []

    def load_table_from_csv(self, csv_path: Path) -> Optional[TableSchema]:
        """Reads a CSV file and converts it into a TableSchema with normalized cells and triplets.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            Optional[TableSchema]: Parsed table structure or None on failure.
        """
        if not csv_path.exists():
            logger.error("Missing ground-truth CSV table: %s", csv_path)
            return None

        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            table = parse_csv_table(content)
            table.title = csv_path.stem
            return table
        except Exception as e:
            logger.error("Failed to parse CSV table at %s: %s", csv_path, e)
            return None

    def scan_dataset(self) -> List[ChartDatasetItem]:
        """Scans image and table directories, pairing chart images with gold CSV files.

        Returns:
            List[ChartDatasetItem]: List of paired dataset items.
        """
        self._items = []

        if not self.images_dir.exists():
            logger.warning("Images directory does not exist: %s. Returning empty dataset.", self.images_dir)
            return []

        image_files = sorted(
            list(self.images_dir.glob("*.png"))
            + list(self.images_dir.glob("*.jpg"))
            + list(self.images_dir.glob("*.jpeg"))
        )

        for img_path in image_files:
            sample_id = img_path.stem
            expected_csv = self.tables_dir / f"{sample_id}.csv"

            table = None
            if expected_csv.exists():
                table = self.load_table_from_csv(expected_csv)
            else:
                logger.warning("Sample %s has missing gold CSV at: %s", sample_id, expected_csv)

            item = ChartDatasetItem(
                id=sample_id,
                image_path=img_path,
                csv_path=expected_csv if expected_csv.exists() else None,
                table=table,
                metadata={"split": self.split, "source": "real_dataset"},
            )
            self._items.append(item)

        logger.info("Scanned %d paired dataset items from %s", len(self._items), self.split_dir)
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ChartDatasetItem]:
        return iter(self._items)
