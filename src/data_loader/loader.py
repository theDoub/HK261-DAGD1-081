"""Module B4: Real Dataset Loader for ChartQA / PlotQA.

Provides structured ingestion pairing chart images with corresponding gold CSV tables,
parsing them into Module B1 TableSchema representations with robust error logging,
and loading paired QA annotations (human and augmented).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from src.format.parser import parse_csv_table
from src.format.schema import TableSchema
from src.metrics.table_metrics import compute_rms_f1

logger = logging.getLogger(__name__)


@dataclass
class ChartQAPair:
    """Represents a question-answer pair associated with a chart.

    Attributes:
        query (str): The natural language question prompt.
        label (str): The expected ground-truth answer text or value.
        source (str): Origin of the QA pair ('human', 'augmented', or 'unknown').
        metadata (Dict[str, Any]): Additional original fields (raw label, etc.).
    """

    query: str
    label: str
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartDatasetItem:
    """Represents a multimodal chart sample in the evaluation pipeline.

    Attributes:
        id (str): Unique sample identifier (e.g. image filename or stem).
        image_path (Optional[Path]): Path to the chart image (PNG/JPG).
        table (Optional[TableSchema]): Parsed ground-truth table representation.
        csv_path (Optional[Path]): Path to the raw gold CSV table.
        qa_pairs (List[ChartQAPair]): List of associated question-answer pairs.
        metadata (Dict[str, Any]): Additional dataset properties (chart type, split, source).
    """

    id: str
    image_path: Optional[Path] = None
    table: Optional[TableSchema] = None
    csv_path: Optional[Path] = None
    qa_pairs: List[ChartQAPair] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChartDataLoader:
    """Dataset loader pairing chart images with corresponding ground-truth CSV tables and QA annotations."""

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
            images_dirname: Subfolder name for chart images (defaults to 'images', auto-detects 'png').
            tables_dirname: Subfolder name for gold CSV tables.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split if (self.data_dir / split).exists() else self.data_dir

        # Auto-detect ChartQA convention ('png/') vs default ('images/')
        if (self.split_dir / "png").exists() and not (self.split_dir / images_dirname).exists():
            self.images_dir = self.split_dir / "png"
        else:
            self.images_dir = self.split_dir / images_dirname

        self.tables_dir = self.split_dir / tables_dirname
        self._items: List[ChartDatasetItem] = []
        self._qa_dict: Dict[str, List[ChartQAPair]] = {}

    def _find_qa_json_files(self) -> List[Path]:
        """Discovers potential QA annotation JSON files in the split and data directories."""
        candidate_paths: List[Path] = []
        search_dirs = [self.split_dir]
        if self.data_dir.resolve() != self.split_dir.resolve():
            search_dirs.append(self.data_dir)

        # Also search in an 'annotations' subdirectory if present
        for d in list(search_dirs):
            ann_sub = d / "annotations"
            if ann_sub.exists() and ann_sub.is_dir():
                search_dirs.append(ann_sub)

        seen_paths = set()
        for d in search_dirs:
            if not d.exists() or not d.is_dir():
                continue
            for p in d.glob("*.json"):
                if p.resolve() in seen_paths:
                    continue
                name_lower = p.name.lower()
                # Prioritize matching split name or standard QA annotations
                if (
                    self.split.lower() in name_lower
                    or "human" in name_lower
                    or "augmented" in name_lower
                    or "qa" in name_lower
                    or "annotation" in name_lower
                ):
                    seen_paths.add(p.resolve())
                    candidate_paths.append(p)

        return sorted(candidate_paths)

    def _parse_qa_file(self, json_path: Path) -> Dict[str, List[ChartQAPair]]:
        """Parses a QA JSON file and maps sample stems to ChartQAPairs."""
        qa_map: Dict[str, List[ChartQAPair]] = {}
        fname = json_path.stem.lower()

        if "human" in fname:
            source = "human"
        elif "augmented" in fname:
            source = "augmented"
        else:
            source = "unknown"

        try:
            with open(json_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to read QA json file at %s: %s", json_path, e)
            return qa_map

        if not isinstance(data, list):
            logger.debug("QA JSON file %s does not contain a list, skipping.", json_path)
            return qa_map

        for entry in data:
            if not isinstance(entry, dict):
                continue

            img_key = entry.get("imgname") or entry.get("image") or entry.get("id") or entry.get("image_id")
            if not img_key:
                continue

            sample_id = Path(str(img_key)).stem
            query = entry.get("query") or entry.get("question") or ""
            raw_label = entry.get("label") or entry.get("answer") or ""

            if isinstance(raw_label, list):
                label_str = str(raw_label[0]) if raw_label else ""
            else:
                label_str = str(raw_label)

            qa_pair = ChartQAPair(
                query=str(query),
                label=label_str,
                source=entry.get("source", source),
                metadata={"raw_label": raw_label, "original_imgname": img_key, **entry},
            )

            if sample_id not in qa_map:
                qa_map[sample_id] = []
            qa_map[sample_id].append(qa_pair)

        return qa_map

    def _load_qa_annotations(self) -> Dict[str, List[ChartQAPair]]:
        """Scans and aggregates all QA annotations for the split."""
        qa_dict: Dict[str, List[ChartQAPair]] = {}
        qa_files = self._find_qa_json_files()
        for fpath in qa_files:
            parsed = self._parse_qa_file(fpath)
            for sid, pairs in parsed.items():
                if sid not in qa_dict:
                    qa_dict[sid] = []
                qa_dict[sid].extend(pairs)
        return qa_dict

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
        """Scans image and table directories, pairing chart images with gold CSV files and QA pairs.

        Returns:
            List[ChartDatasetItem]: List of paired dataset items.
        """
        self._items = []
        self._qa_dict = self._load_qa_annotations()

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

            qa_pairs = self._qa_dict.get(sample_id, [])

            item = ChartDatasetItem(
                id=sample_id,
                image_path=img_path,
                csv_path=expected_csv if expected_csv.exists() else None,
                table=table,
                qa_pairs=qa_pairs,
                metadata={"split": self.split, "source": "real_dataset"},
            )
            self._items.append(item)

        logger.info("Scanned %d paired dataset items from %s", len(self._items), self.split_dir)
        return self._items

    def verify_gold_tables(self) -> Dict[str, Any]:
        """Verifies that all loaded ground-truth tables achieve gold-vs-gold F1 = 1.0.

        Returns:
            Dict[str, Any]: Summary containing total_tables, valid_count, all_valid, and invalid_ids.
        """
        if not self._items:
            self.scan_dataset()

        total = 0
        valid = 0
        invalid_ids = []

        for item in self._items:
            if item.table is not None:
                total += 1
                rms = compute_rms_f1(item.table, item.table)
                if abs(rms["f1"] - 1.0) < 1e-6:
                    valid += 1
                else:
                    invalid_ids.append(item.id)

        return {
            "total_tables": total,
            "valid_count": valid,
            "all_valid": total > 0 and len(invalid_ids) == 0,
            "invalid_ids": invalid_ids,
        }

    def get_item_by_id(self, sample_id: str) -> Optional[ChartDatasetItem]:
        """Finds a dataset item by sample id."""
        for item in self._items:
            if item.id == sample_id:
                return item
        return None

    def get_all_qa_pairs(self) -> List[Tuple[str, ChartQAPair]]:
        """Returns all QA pairs across all dataset items as (sample_id, qa_pair) tuples."""
        if not self._items:
            self.scan_dataset()
        pairs: List[Tuple[str, ChartQAPair]] = []
        for item in self._items:
            for qa in item.qa_pairs:
                pairs.append((item.id, qa))
        return pairs

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ChartDatasetItem]:
        return iter(self._items)
