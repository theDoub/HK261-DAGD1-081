"""Application-level chart extraction and output formatting functions."""

from pathlib import Path
from typing import Dict, Optional, Union

from src.format.formatter import to_csv, to_linearized_deplot, to_markdown
from src.format.schema import TableSchema
from src.models.harness import VLMTableHarness

ImagePath = Union[str, Path]


def _extract_table(
    image_path: ImagePath,
    harness: Optional[VLMTableHarness],
) -> TableSchema:
    active_harness = harness or VLMTableHarness(
        backend="deplot",
        target_format="deplot",
    )
    result = active_harness.extract_table(image_path)
    if not result.parse_ok or result.table is None:
        raise RuntimeError(result.error_message or "Chart extraction or parsing failed")
    return result.table


def process_chart(
    image_path: ImagePath,
    output_format: str = "markdown",
    harness: Optional[VLMTableHarness] = None,
) -> str:
    """Extract one chart and serialize it into a single requested format."""
    serializers = {
        "linearized": to_linearized_deplot,
        "markdown": to_markdown,
        "csv": to_csv,
    }
    try:
        serializer = serializers[output_format]
    except KeyError as error:
        raise ValueError(
            "output_format must be 'linearized', 'markdown', or 'csv'"
        ) from error

    table = _extract_table(image_path, harness)
    return serializer(table)


def process_all_chart(
    image_path: ImagePath,
    harness: Optional[VLMTableHarness] = None,
) -> Dict[str, str]:
    """Extract one chart once and serialize the table into every supported format."""
    table = _extract_table(image_path, harness)
    return {
        "linearized": to_linearized_deplot(table),
        "markdown": to_markdown(table),
        "csv": to_csv(table),
    }