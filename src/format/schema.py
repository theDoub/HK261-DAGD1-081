"""Module B1: Common Table Schema.

Defines standard table representation data structures for chart-extracted
tables, supporting round-trip conversions (JSON, pandas DataFrame, Markdown,
linearized prompt format) for multimodal Chart-to-Table -> Table QA pipelines.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from pydantic import BaseModel, Field, ConfigDict

    class _BaseTable(BaseModel):
        """Pydantic base table model when pydantic is available."""
        model_config = ConfigDict(arbitrary_types_allowed=True)
        title: Optional[str] = Field(default=None, description="Optional title or header of the chart/table.")
        columns: List[str] = Field(default_factory=list, description="List of table column headers.")
        rows: List[List[Union[str, int, float, None]]] = Field(
            default_factory=list, description="Matrix of row values."
        )
        metadata: Dict[str, Any] = Field(
            default_factory=dict, description="Arbitrary metadata (e.g. chart type, dataset source, QA pairs)."
        )

    _USE_PYDANTIC = True
except ImportError:  # Fallback for environments before `pip install pydantic`
    from dataclasses import dataclass, field

    @dataclass
    class _BaseTable:  # type: ignore[no-redef]
        """Dataclass fallback table model."""
        title: Optional[str] = None
        columns: List[str] = field(default_factory=list)
        rows: List[List[Union[str, int, float, None]]] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

    _USE_PYDANTIC = False


class TableSchema(_BaseTable):
    """Standardized representation of a table extracted from a chart.

    Attributes:
        title (Optional[str]): The title or caption of the table/chart.
        columns (List[str]): List of column names/headers.
        rows (List[List[Union[str, int, float, None]]]): 2D array representing table rows.
        metadata (Dict[str, Any]): Additional dataset or task-specific metadata.
    """

    def validate_shape(self) -> bool:
        """Checks if all rows match the column length.

        Returns:
            bool: True if table is well-formed, False otherwise.
        """
        num_cols = len(self.columns)
        return all(len(row) == num_cols for row in self.rows)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the table schema into a standard Python dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing title, columns, rows, and metadata.
        """
        if _USE_PYDANTIC and hasattr(self, "model_dump"):
            return self.model_dump()
        return {
            "title": self.title,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableSchema:
        """Constructs a TableSchema instance from a dictionary.

        Args:
            data (Dict[str, Any]): Source dictionary.

        Returns:
            TableSchema: Instantiated table schema.
        """
        return cls(
            title=data.get("title"),
            columns=data.get("columns", []),
            rows=data.get("rows", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> TableSchema:
        """Loads TableSchema from a JSON string.

        Args:
            json_str (str): Raw JSON string.

        Returns:
            TableSchema: Parsed table schema instance.
        """
        return cls.from_dict(json.loads(json_str))

    def to_json(self, indent: int = 2) -> str:
        """Serializes TableSchema to a formatted JSON string.

        Args:
            indent (int): JSON indentation level.

        Returns:
            str: JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_dataframe(self) -> Any:
        """Converts table schema to a pandas DataFrame if pandas is installed.

        Returns:
            pandas.DataFrame: DataFrame representation of the table.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("pandas is required for to_dataframe(). Please install it via requirements.txt.") from err

        return pd.DataFrame(self.rows, columns=self.columns)

    def to_markdown(self) -> str:
        """Renders the table as a clean Markdown string.

        Returns:
            str: Markdown formatted table string.
        """
        lines: List[str] = []
        if self.title:
            lines.append(f"### {self.title}\n")

        if not self.columns:
            return ""

        # Format header row
        headers = [str(col).strip() for col in self.columns]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Format data rows
        for row in self.rows:
            formatted_row: List[str] = []
            for idx in range(len(headers)):
                val = row[idx] if idx < len(row) else ""
                val_str = "" if val is None else str(val).strip()
                formatted_row.append(val_str)
            lines.append("| " + " | ".join(formatted_row) + " |")

        return "\n".join(lines)

    def to_linearized_text(self) -> str:
        """Linearizes table into standard pipe-separated sequence (e.g., DePlot/Pix2Struct style).

        Format:
            TITLE | <title>
            col1 | col2 | ...
            row1_c1 | row1_c2 | ...

        Returns:
            str: Linearized string representation.
        """
        chunks: List[str] = []
        if self.title:
            chunks.append(f"TITLE | {self.title}")

        if self.columns:
            chunks.append(" | ".join(str(col) for col in self.columns))

        for row in self.rows:
            chunks.append(" | ".join("" if v is None else str(v) for v in row))

        return "\n".join(chunks)

    @classmethod
    def from_linearized_text(cls, text: str, default_title: Optional[str] = None) -> TableSchema:
        """Parses a linearized table string (e.g. DePlot model output) back to TableSchema.

        Args:
            text (str): Pipe-delimited linearized string.
            default_title (Optional[str]): Fallback title if TITLE token is missing.

        Returns:
            TableSchema: Reconstructed table schema.
        """
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        title = default_title
        columns: List[str] = []
        rows: List[List[Union[str, int, float, None]]] = []

        for line in lines:
            if line.startswith("TITLE |"):
                title = line.split("TITLE |", 1)[1].strip()
            elif not columns:
                columns = [part.strip() for part in line.split("|")]
            else:
                raw_cells = [part.strip() for part in line.split("|")]
                row_vals: List[Union[str, int, float, None]] = []
                for cell in raw_cells:
                    if cell.lower() in ("none", "null", ""):
                        row_vals.append(None)
                    else:
                        try:
                            if "." in cell:
                                row_vals.append(float(cell))
                            else:
                                row_vals.append(int(cell))
                        except ValueError:
                            row_vals.append(cell)
                rows.append(row_vals)

        return cls(title=title, columns=columns, rows=rows)
