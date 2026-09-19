"""Module B5: VLM Table Extraction Harness.

Provides a pluggable backend architecture for extracting tables from chart images:
- MockBackend: For deterministic offline testing and sanity verification.
- OpenVLMBackend: Stub for open-source vision models (e.g. Qwen2.5-VL, DePlot).
- APIVLMBackend: Stub for hosted APIs (e.g. Gemini, Claude) with lightweight caching.

Features robust error handling (saves raw text, records parse_ok flag, isolates sample failures).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.format.parser import parse_table_text
from src.format.schema import TableSchema

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Encapsulates the output and diagnosis of a single extraction attempt.

    Attributes:
        raw_text (str): Raw unparsed model response.
        table (Optional[TableSchema]): Parsed table representation if parsing succeeded.
        parse_ok (bool): True if parsing succeeded and produced valid tabular rows.
        error_message (Optional[str]): Error description if extraction or parsing failed.
        backend_name (str): Identifier of the extraction backend used.
        metadata (Dict[str, Any]): Additional execution metadata.
    """

    raw_text: str
    table: Optional[TableSchema] = None
    parse_ok: bool = False
    error_message: Optional[str] = None
    backend_name: str = "mock"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExtractionBackend(ABC):
    """Abstract base class for vision-language model extraction backends."""

    @abstractmethod
    def generate(self, prompt: str, image_path: Optional[Union[str, Path]] = None) -> str:
        """Generates raw text extraction from the model."""
        pass


class MockBackend(BaseExtractionBackend):
    """Deterministic mock backend returning valid markdown or linearized tables for smoke testing."""

    def __init__(self, predefined_table_md: Optional[str] = None) -> None:
        self.predefined_table_md = predefined_table_md or (
            "| Quarter | Sales |\n"
            "|---|---|\n"
            "| Q1 | 120 |\n"
            "| Q2 | 150 |\n"
            "| Q3 | 180 |"
        )

    def generate(self, prompt: str, image_path: Optional[Union[str, Path]] = None) -> str:
        """Returns mock markdown table wrapped with optional conversational preamble."""
        return f"Here is the extracted table from the chart:\n\n```markdown\n{self.predefined_table_md}\n```"


class OpenVLMBackend(BaseExtractionBackend):
    """Stub harness for local open-source vision-language models (e.g. Qwen2.5-VL, DePlot)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> None:
        self.model_name = model_name

    def generate(self, prompt: str, image_path: Optional[Union[str, Path]] = None) -> str:
        """Stub generation simulating local open VLM output."""
        logger.info("Simulating local Open VLM generation with model: %s", self.model_name)
        return (
            "| Category | Value A | Value B |\n"
            "|---|---|---|\n"
            "| Alpha | 45.2 | 30.1 |\n"
            "| Beta | 52.8 | 61.4 |"
        )


class APIVLMBackend(BaseExtractionBackend):
    """Stub harness for hosted vision APIs (e.g. Gemini 2.5, Claude 3.7) with in-memory caching."""

    def __init__(self, model_name: str = "gemini-2.5-flash", cache_dir: Optional[Path] = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._memory_cache: Dict[str, str] = {}

    def _get_cache_key(self, prompt: str, image_path: Optional[Union[str, Path]]) -> str:
        raw_key = f"{self.model_name}_{prompt}_{str(image_path)}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, image_path: Optional[Union[str, Path]] = None) -> str:
        """Queries the API with basic response caching."""
        cache_key = self._get_cache_key(prompt, image_path)
        if cache_key in self._memory_cache:
            logger.debug("Cache hit for API VLM query: %s", cache_key)
            return self._memory_cache[cache_key]

        logger.info("Executing API VLM call [%s] for image: %s", self.model_name, image_path)
        # Stub response simulating an API VLM answer
        result = (
            "```markdown\n"
            "| Region | Solar | Wind |\n"
            "|---|---|---|\n"
            "| North | 120 | 180 |\n"
            "| South | 95 | 110 |\n"
            "```"
        )
        self._memory_cache[cache_key] = result
        return result


class VLMTableHarness:
    """Orchestrates chart-to-table extraction across pluggable backends with robust error isolation."""

    def __init__(
        self,
        backend: Union[str, BaseExtractionBackend] = "mock",
        target_format: str = "markdown",
    ) -> None:
        """Initializes the harness.

        Args:
            backend: Backend instance or name ('mock', 'open_vlm', 'api_vlm').
            target_format: Expected output format ('markdown' or 'deplot').
        """
        self.target_format = target_format
        if isinstance(backend, BaseExtractionBackend):
            self.backend = backend
        elif backend == "open_vlm":
            self.backend = OpenVLMBackend()
        elif backend == "api_vlm":
            self.backend = APIVLMBackend()
        else:
            self.backend = MockBackend()

    def format_prompt(self) -> str:
        """Constructs prompt instructing VLM to extract chart data into the target tabular format."""
        if self.target_format == "deplot":
            return (
                "Generate underlying data table of the figure below in linearized format:\n"
                "TITLE | <title>\n<header_1> | <header_2> ...\n<val_1> | <val_2> ..."
            )
        return (
            "Extract all numerical values and categories from this chart into a clean GitHub-Flavored "
            "Markdown table. Output only the markdown table without conversational explanation."
        )

    def extract_table(
        self,
        image_path: Optional[Union[str, Path]] = None,
        prompt: Optional[str] = None,
    ) -> ExtractionResult:
        """Executes extraction on a single image, safely isolating failures.

        Args:
            image_path: Filepath to chart image.
            prompt: Optional prompt override.

        Returns:
            ExtractionResult: Complete record containing raw text, parsed table, and parse_ok flag.
        """
        active_prompt = prompt or self.format_prompt()
        backend_name = self.backend.__class__.__name__

        try:
            raw_text = self.backend.generate(active_prompt, image_path)
        except Exception as e:
            logger.error("Backend generation error on %s: %s", image_path, e)
            return ExtractionResult(
                raw_text="",
                table=None,
                parse_ok=False,
                error_message=f"Backend failure: {str(e)}",
                backend_name=backend_name,
            )

        # Attempt parsing into TableSchema (Module B1)
        try:
            table = parse_table_text(raw_text, fmt=self.target_format)
            parse_ok = bool(table and table.columns and len(table.rows) > 0)
            error_msg = None if parse_ok else "Parsed table contained no rows or columns"

            return ExtractionResult(
                raw_text=raw_text,
                table=table if parse_ok else None,
                parse_ok=parse_ok,
                error_message=error_msg,
                backend_name=backend_name,
            )
        except Exception as e:
            logger.error("Failed to parse raw text from %s: %s", image_path, e)
            return ExtractionResult(
                raw_text=raw_text,
                table=None,
                parse_ok=False,
                error_message=f"Parsing exception: {str(e)}",
                backend_name=backend_name,
            )

    def batch_extract(
        self,
        image_paths: List[Union[str, Path]],
    ) -> List[ExtractionResult]:
        """Extracts tables from a sequence of images. An error on one image will not halt others."""
        results: List[ExtractionResult] = []
        for p in image_paths:
            res = self.extract_table(p)
            results.append(res)
        return results
