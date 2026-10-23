"""Format TableSchema into Linearized DePlot, Markdown, and CSV."""

import csv
import io

from src.format.schema import TableSchema


def to_linearized_deplot(table: TableSchema) -> str:
    """
    Convert TableSchema into the custom Linearized DePlot format.

    Example:
        <START> TITLE | Revenue vs Expenses
        <NEW_ROW> Product | Revenue | Expenses
        <NEW_ROW> Speakers | 112472.5 | 89181.4
        <END>
    """

    title = "" if table.title is None else str(table.title)
    lines = [
        f"<START> TITLE | {title}",
        "<NEW_ROW> " + " | ".join(table.columns),
    ]

    for row in table.rows:
        values = [
            "" if value is None else str(value)
            for value in row
        ]

        lines.append("<NEW_ROW> " + " | ".join(values))

    lines.append("<END>")

    return "\n".join(lines)


def to_markdown(table: TableSchema) -> str:
    """
    Convert TableSchema into a Markdown table.

    Uses the existing TableSchema.to_markdown() implementation
    to avoid duplicating formatting logic.
    """

    return table.to_markdown()


def to_csv(table: TableSchema) -> str:
    """
    Convert TableSchema into CSV.

    The title is intentionally excluded from the CSV table.
    """

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)

    writer.writerow(table.columns)
    writer.writerows(table.rows)

    return buffer.getvalue()