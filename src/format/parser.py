"""Module B1: Table Parsers and Normalization Utilities.

Implements normalization for headers and values, codeblock/conversational stripping,
and parsers for Markdown, CSV, and Linearized DePlot formats into TableSchema.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any, List, Optional

from src.format.schema import TableSchema, TableTriplet


def normalize_header(header: str) -> str:
    """Normalizes table header by lowercasing and collapsing multiple whitespace characters.

    Args:
        header: Raw header string.

    Returns:
        str: Normalized header string.
    """
    if not isinstance(header, str):
        header = str(header)
    # Collapse multiple whitespaces and lowercase
    cleaned = re.sub(r"\s+", " ", header).strip().lower()
    return cleaned


def normalize_value(val: Any) -> Any:
    """Normalizes cell values by stripping currency symbols, commas, percent signs, and converting to numeric where possible.

    Args:
        val: Raw cell value.

    Returns:
        Union[int, float, str, None]: Normalized cell value.
    """
    if val is None or isinstance(val, (int, float)):
        return val

    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan", "n/a", "-"):
        return None

    # Remove currency symbols ($ € £ ¥ ₫ ₹), commas, and percent signs
    cleaned = re.sub(r"[\$,€,£,¥,₫,₹,%]", "", s).strip()
    # Also remove commas used as thousands separators
    cleaned_num = cleaned.replace(",", "")

    # Try parsing as integer or float
    try:
        if "." in cleaned_num or "e" in cleaned_num.lower():
            return float(cleaned_num)
        return int(cleaned_num)
    except ValueError:
        # If not a valid number, return original trimmed string
        return s


def strip_markdown_codeblocks(text: str) -> str:
    """Strips conversational prefixes and extracts content from markdown codeblocks.

    Handles:
        - Conversational prefixes ("Here is the table:", "Sure!...")
        - ```markdown ... ``` or ```csv ... ``` code fences.

    Args:
        text: Raw model output string.

    Returns:
        str: Extracted table content string.
    """
    cleaned = text.strip()

    # Extract block inside ```...``` if present
    code_match = re.search(r"```(?:markdown|csv|table)?\s*\n(.*?)\n\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()

    # If starts with ``` and ends with ```
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Remove conversational preamble if first table line starts with '#', '|', or contains 'TITLE |'
    lines = cleaned.split("\n")
    table_start_idx = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("|") or stripped.startswith("TITLE |") or "," in stripped:
            table_start_idx = idx
            break

    return "\n".join(lines[table_start_idx:]).strip()


def parse_markdown_table(text: str) -> TableSchema:
    """Parses a Markdown table string into a TableSchema.

    Args:
        text: Raw Markdown table string.

    Returns:
        TableSchema: Parsed table structure with normalized headers and values.
    """
    cleaned = strip_markdown_codeblocks(text)
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

    title = ""
    header_line: Optional[str] = None
    data_lines: List[str] = []

    for line in lines:
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            continue
        if "|" in line:
            # Check if this is a separator line (e.g. |---|---|)
            if re.match(r"^\|?[\s\-:|]+\|?$", line):
                continue
            if header_line is None:
                header_line = line
            else:
                data_lines.append(line)

    if header_line is None:
        return TableSchema(title=title, columns=[], rows=[], triplets=[])

    raw_headers = [c.strip() for c in header_line.strip("|").split("|")]
    columns = [normalize_header(h) for h in raw_headers]

    rows: List[List[Any]] = []
    triplets: List[TableTriplet] = []

    for r_idx, line in enumerate(data_lines):
        raw_cells = [c.strip() for c in line.strip("|").split("|")]
        row_vals: List[Any] = []
        for c_idx, col_name in enumerate(columns):
            raw_cell = raw_cells[c_idx] if c_idx < len(raw_cells) else ""
            val = normalize_value(raw_cell)
            row_vals.append(val)
            triplets.append(TableTriplet(row=r_idx, col=col_name, value=val))
        rows.append(row_vals)

    return TableSchema(title=title, columns=columns, rows=rows, triplets=triplets)


def parse_csv_table(text: str) -> TableSchema:
    """Parses a CSV table string into a TableSchema.

    Args:
        text: Raw CSV string.

    Returns:
        TableSchema: Parsed table structure.
    """
    cleaned = strip_markdown_codeblocks(text)
    f = io.StringIO(cleaned)
    reader = csv.reader(f)

    rows_raw = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows_raw:
        return TableSchema(columns=[], rows=[], triplets=[])

    columns = [normalize_header(c) for c in rows_raw[0]]
    rows: List[List[Any]] = []
    triplets: List[TableTriplet] = []

    for r_idx, raw_cells in enumerate(rows_raw[1:]):
        row_vals: List[Any] = []
        for c_idx, col_name in enumerate(columns):
            raw_val = raw_cells[c_idx] if c_idx < len(raw_cells) else ""
            val = normalize_value(raw_val)
            row_vals.append(val)
            triplets.append(TableTriplet(row=r_idx, col=col_name, value=val))
        rows.append(row_vals)

    return TableSchema(columns=columns, rows=rows, triplets=triplets)


def parse_deplot_linearized(text: str) -> TableSchema:
    """Parses a Linearized DePlot output string (using '<0x0A>' or newline delimiters).

    Format:
        TITLE | <title>
        col1 | col2 | ...
        row1_c1 | row1_c2 | ...

    Args:
        text: Linearized table string from DePlot.

    Returns:
        TableSchema: Parsed TableSchema.
    """
    cleaned = strip_markdown_codeblocks(text)
    # Replace '<0x0A>' tokens with newline
    normalized_text = cleaned.replace("<0x0A>", "\n")
    lines = [line.strip() for line in normalized_text.split("\n") if line.strip()]

    title = ""
    columns: List[str] = []
    rows: List[List[Any]] = []
    triplets: List[TableTriplet] = []

    data_row_idx = 0
    for line in lines:
        if line.startswith("TITLE |"):
            title = line.split("TITLE |", 1)[1].strip()
            continue

        parts = [p.strip() for p in line.split("|")]
        if not columns:
            columns = [normalize_header(p) for p in parts]
        else:
            row_vals: List[Any] = []
            for c_idx, col_name in enumerate(columns):
                raw_cell = parts[c_idx] if c_idx < len(parts) else ""
                val = normalize_value(raw_cell)
                row_vals.append(val)
                triplets.append(TableTriplet(row=data_row_idx, col=col_name, value=val))
            rows.append(row_vals)
            data_row_idx += 1

    return TableSchema(title=title, columns=columns, rows=rows, triplets=triplets)


def parse_table_text(text: str, fmt: str = "auto") -> TableSchema:
    """Auto-detects format and parses table text into TableSchema.

    Args:
        text: Input raw text from model or file.
        fmt: Format hint ('auto', 'markdown', 'csv', 'deplot').

    Returns:
        TableSchema: Extracted table schema.
    """
    cleaned = strip_markdown_codeblocks(text)
    if fmt == "markdown" or (fmt == "auto" and "|" in cleaned and "-|-" in cleaned):
        return parse_markdown_table(cleaned)
    elif fmt == "deplot" or (fmt == "auto" and ("<0x0A>" in cleaned or cleaned.startswith("TITLE |"))):
        return parse_deplot_linearized(cleaned)
    elif fmt == "csv" or (fmt == "auto" and "," in cleaned and "\n" in cleaned and "|" not in cleaned):
        return parse_csv_table(cleaned)
    elif "|" in cleaned:
        # Fallback to markdown or pipe parsing
        return parse_markdown_table(cleaned)

    # Fallback to CSV
    return parse_csv_table(cleaned)
