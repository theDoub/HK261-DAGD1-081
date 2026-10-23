"""Module B1: Table schema, triplet representation, and parsers."""

from src.format.parser import (
    normalize_header,
    normalize_value,
    parse_csv_table,
    parse_deplot_linearized,
    parse_markdown_table,
    parse_table_text,
    strip_markdown_codeblocks,
)

from src.format.formatter import (
    to_linearized_deplot,
    to_markdown,
    to_csv,
)

from src.format.schema import TableSchema, TableTriplet

__all__ = [
    "TableSchema",
    "TableTriplet",
    "normalize_header",
    "normalize_value",
    "strip_markdown_codeblocks",
    "parse_markdown_table",
    "parse_csv_table",
    "parse_deplot_linearized",
    "parse_table_text",
    "to_linearized_deplot",
    "to_markdown",
    "to_csv",
]
