"""Module B1: Common Table Schema and Triplet Representation.

Defines the standard TableSchema and TableTriplet data structures using Pydantic v2.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from tabulate import tabulate


class TableTriplet(BaseModel):
    """Canonical cell representation as a triplet: (row_idx, col_header, value).

    Attributes:
        row (int): Zero-indexed row number.
        col (str): Column header name.
        value (Any): Normalized cell value (float, int, str, or None).
    """

    row: int
    col: str
    value: Any


class TableSchema(BaseModel):
    """Standardized representation of a chart-extracted table.

    Attributes:
        title (Optional[str]): Title or caption of the chart/table.
        columns (List[str]): List of column names/headers.
        rows (List[List[Any]]): 2D matrix of row cell values.
        triplets (List[TableTriplet]): Relational cell triplets (row, col, value).
        metadata (Dict[str, Any]): Additional dataset and task-specific metadata.
    """

    title: Optional[str] = Field(default="", description="Title or caption of the table.")
    columns: List[str] = Field(default_factory=list, description="List of column headers.")
    rows: List[List[Any]] = Field(default_factory=list, description="2D matrix of cell values.")
    triplets: List[TableTriplet] = Field(
        default_factory=list, description="List of (row, col, value) cell triplets."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata.")

    @model_validator(mode="after")
    def sync_triplets(self) -> TableSchema:
        """Ensures triplets are automatically populated from rows/columns if omitted."""
        if not self.triplets and self.rows and self.columns:
            triplets: List[TableTriplet] = []
            for r_idx, row in enumerate(self.rows):
                for c_idx, val in enumerate(row):
                    col_name = self.columns[c_idx] if c_idx < len(self.columns) else f"col_{c_idx}"
                    triplets.append(TableTriplet(row=r_idx, col=str(col_name), value=val))
            self.triplets = triplets
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Converts table rows and columns to a pandas DataFrame.

        Returns:
            pd.DataFrame: Table represented as DataFrame.
        """
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_markdown(self) -> str:
        """Renders table as formatted GitHub-Flavored Markdown.

        Returns:
            str: Markdown formatted table string.
        """
        table_str = tabulate(self.rows, headers=self.columns, tablefmt="github")
        if self.title:
            return f"### {self.title}\n\n{table_str}"
        return table_str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes TableSchema to a standard Python dictionary.

        Returns:
            Dict[str, Any]: Model dictionary dump.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableSchema:
        """Instantiates TableSchema from a dictionary.

        Args:
            data (Dict[str, Any]): Input dictionary.

        Returns:
            TableSchema: Instantiated model.
        """
        return cls(**data)

    @classmethod
    def from_triplets(
        cls,
        triplets: List[TableTriplet],
        title: Optional[str] = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TableSchema:
        """Reconstructs TableSchema matrix (columns, rows) from a list of TableTriplets.

        Args:
            triplets: List of TableTriplet items.
            title: Optional table title.
            metadata: Optional metadata dictionary.

        Returns:
            TableSchema: Constructed table.
        """
        if not triplets:
            return cls(title=title, columns=[], rows=[], triplets=[], metadata=metadata or {})

        # Preserve column appearance order
        cols: List[str] = []
        for t in triplets:
            if t.col not in cols:
                cols.append(t.col)

        max_row = max(t.row for t in triplets)
        col_to_idx = {c: i for i, c in enumerate(cols)}

        grid: List[List[Any]] = [[None for _ in cols] for _ in range(max_row + 1)]
        for t in triplets:
            grid[t.row][col_to_idx[t.col]] = t.value

        return cls(
            title=title,
            columns=cols,
            rows=grid,
            triplets=triplets,
            metadata=metadata or {},
        )
