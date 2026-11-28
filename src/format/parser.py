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


def _parse_explicit_linearized(lines: List[str]) -> TableSchema:
    """Strictly parses the explicit Linearized DePlot token grammar."""
    if not lines or lines[-1] != "<END>":
        raise ValueError("Malformed linearized table: missing exactly closing <END>")

    end_count = sum(line == "<END>" for line in lines)
    if end_count != 1:
        raise ValueError("Malformed linearized table: expected exactly one <END>")

    title = ""
    columns: List[str] = []
    rows: List[List[Any]] = []
    triplets: List[TableTriplet] = []
    seen_start = False

    for line in lines[:-1]:
        if re.search(r"<SNEW_ROW>", line, re.IGNORECASE):
            raise ValueError(
                "Malformed linearized table: unknown row token <SNEW_ROW>"
            )

        if line.startswith("TITLE |"):
            title = line.split("|", 1)[1].strip()
            continue

        if line.startswith("<START>"):
            if seen_start:
                raise ValueError("Malformed linearized table: duplicate <START>")
            seen_start = True
            payload = line[len("<START>"):].strip()
            if payload.startswith("TITLE |"):
                title = payload.split("|", 1)[1].strip()
            elif payload:
                if columns:
                    raise ValueError("Malformed linearized table: duplicate header row")
                header_parts = [part.strip() for part in payload.split("|")]
                if any(not part for part in header_parts):
                    raise ValueError(
                        "Malformed linearized table: header contains an empty column"
                    )
                columns = [normalize_header(part) for part in header_parts]
            continue

        if line.startswith("<NEW_ROW>"):
            payload = line[len("<NEW_ROW>"):].strip()
            if not payload:
                raise ValueError("Malformed linearized table: <NEW_ROW> has no values")
            parts = [part.strip() for part in payload.split("|")]

            if not columns:
                if not seen_start:
                    raise ValueError(
                        "Malformed linearized table: header must follow <START>"
                    )
                if any(not part for part in parts):
                    raise ValueError(
                        "Malformed linearized table: header contains an empty column"
                    )
                columns = [normalize_header(part) for part in parts]
                continue

            if len(parts) != len(columns):
                raise ValueError(
                    "Malformed linearized table: row has "
                    f"{len(parts)} values but header has {len(columns)} columns"
                )

            row_idx = len(rows)
            values = [normalize_value(value) for value in parts]
            rows.append(values)
            for col_name, value in zip(columns, values):
                triplets.append(
                    TableTriplet(row=row_idx, col=col_name, value=value)
                )
            continue

        if re.match(r"<[^>]+>", line):
            raise ValueError(
                f"Malformed linearized table: unknown token in line {line!r}"
            )
        raise ValueError(
            f"Malformed linearized table: expected a tokenized line, got {line!r}"
        )

    if not seen_start:
        raise ValueError("Malformed linearized table: missing <START>")
    if not columns:
        raise ValueError("Malformed linearized table: missing header row")

    return TableSchema(
        title=title,
        columns=columns,
        rows=rows,
        triplets=triplets,
    )


def parse_deplot_linearized(text: str) -> TableSchema:
    """
    Parse raw DePlot output or custom Linearized DePlot text.

    Supported examples:

        TITLE | Revenue vs Expenses
        Product | Revenue | Expenses
        Speakers | 112472.5 | 89181.4

    Also supports:

        TITLE | Revenue vs Expenses
        <START> Product | Revenue | Expenses
        <NEW_ROW> Speakers | 112472.5 | 89181.4
        <END>
    """

    if not isinstance(text, str):
        raise TypeError("DePlot input must be a string")

    normalized_text = text.replace("<0x0A>", "\n")

    # Remove optional code fences without interpreting commas
    # or pipes as evidence of a table.
    normalized_text = re.sub(
        r"^\s*```(?:\w+)?\s*\n?",
        "",
        normalized_text,
    )
    normalized_text = re.sub(
        r"\n?\s*```\s*$",
        "",
        normalized_text,
    )

    lines = [
        line.strip()
        for line in normalized_text.splitlines()
        if line.strip()
    ]

    if re.search(r"<(?:START|NEW_ROW|END|SNEW_ROW)>", normalized_text):
        return _parse_explicit_linearized(lines)

    title = ""
    columns: List[str] = []
    rows: List[List[Any]] = []
    triplets: List[TableTriplet] = []

    for line in lines:
        if line == "<END>":
            break

        if line.startswith("TITLE |"):
            title = line.split("|", 1)[1].strip()
            continue

        if line.startswith("<START>"):
            line = line[len("<START>"):].strip()
            if line.startswith("TITLE |"):
                title = line.split("|", 1)[1].strip()
                continue

        is_data_row = line.startswith("<NEW_ROW>")

        if is_data_row:
            line = line[len("<NEW_ROW>"):].strip()

        if not line:
            continue

        parts = [part.strip() for part in line.split("|")]

        if not columns:
            columns = [normalize_header(part) for part in parts]
            continue

        # Explicit row tokens identify data rows.
        # Without them, support the original DePlot format.
        raw_values = parts

        if len(raw_values) != len(columns):
            if len(raw_values) < len(columns):
                raw_values = raw_values + [""] * (len(columns) - len(raw_values))
            else:
                raw_values = raw_values[:len(columns)]

        row_idx = len(rows)

        values = [
            normalize_value(value)
            for value in raw_values
        ]

        rows.append(values)

        for col_name, value in zip(columns, values):
            triplets.append(
                TableTriplet(
                    row=row_idx,
                    col=col_name,
                    value=value,
                )
            )

    return TableSchema(
        title=title,
        columns=columns,
        rows=rows,
        triplets=triplets,
    )


def parse_table_text(text: str, fmt: str = "auto") -> TableSchema:
    """Auto-detects format and parses table text into TableSchema.

    Args:
        text: Input raw text from model or file.
        fmt: Format hint ('auto', 'markdown', 'csv', 'deplot').

    Returns:
        TableSchema: Extracted table schema.
    """
    if fmt in {"deplot", "linearized"}:
        return parse_deplot_linearized(text)

    raw_text = text if isinstance(text, str) else str(text)
    if fmt == "auto" and re.search(
        r"<(?:START|NEW_ROW|END|SNEW_ROW)>|<0x0A>",
        raw_text,
    ):
        return parse_deplot_linearized(raw_text)
    if fmt == "auto":
        first_nonempty = next(
            (line.strip() for line in raw_text.splitlines() if line.strip()),
            "",
        )
        if first_nonempty.startswith("TITLE |"):
            return parse_deplot_linearized(raw_text)

    cleaned = strip_markdown_codeblocks(text)
    if fmt == "markdown" or (fmt == "auto" and "|" in cleaned and "-|-" in cleaned):
        return parse_markdown_table(cleaned)
    elif fmt in {"deplot", "linearized"} or (
        fmt == "auto" and ("<0x0A>" in cleaned or cleaned.startswith("TITLE |"))
    ):
        return parse_deplot_linearized(cleaned)
    elif fmt == "csv" or (fmt == "auto" and "," in cleaned and "\n" in cleaned and "|" not in cleaned):
        return parse_csv_table(cleaned)
    elif "|" in cleaned:
        # Fallback to markdown or pipe parsing
        return parse_markdown_table(cleaned)

    # Fallback to CSV
    return parse_csv_table(cleaned)
