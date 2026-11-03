"""Module B5: VLM Table Extraction Harness.

Provides a pluggable backend architecture for extracting tables from chart images:
- MockBackend: For deterministic offline testing and sanity verification.
- OpenVLMBackend: Local open-source vision models (e.g. Qwen2.5-VL).
- DePlotBackend: Google DePlot chart-to-table model.
- APIVLMBackend: Stub for hosted APIs (e.g. Gemini, Claude) with lightweight caching.

Features robust error handling, records parse status, and isolates sample failures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
import logging
import gc
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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
    """Local Transformers backend optimized for limited-VRAM GPUs."""

    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        max_new_tokens: int = 256,
        quantization: str = "4bit",
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if quantization not in {"none", "4bit"}:
            raise ValueError("quantization must be either 'none' or '4bit'")
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization
        self._processor: Any = None
        self._model: Any = None

    def _load_model(self) -> None:
        """Load the model once using a GPU-memory-conscious configuration."""
        import torch
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device}' requested, but CUDA is unavailable"
            )
        if self.quantization == "4bit" and not device.startswith("cuda"):
            raise ValueError(
                "4-bit quantization requires a supported CUDA setup. "
                "Use quantization='none' for CPU inference."
            )
        logger.info("Loading processor: %s", self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name)
        model_kwargs = {
            "low_cpu_mem_usage": True,
        }
        if device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = {"": 0}
            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
        else:
            model_kwargs["torch_dtype"] = torch.float32
        logger.info(
            "Loading model: %s | device=%s | quantization=%s",
            self.model_name,
            device,
            self.quantization,
        )
        start = time.perf_counter()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        if not self.quantization == "4bit" or not device.startswith("cuda"):
            model.to(device)
        model.eval()
        self._processor = processor
        self._model = model
        self.device = device
        elapsed = time.perf_counter() - start
        logger.info("Model loaded in %.2f seconds", elapsed)
        if device.startswith("cuda"):
            logger.info(
                "GPU allocated memory after loading: %.2f GiB",
                torch.cuda.memory_allocated() / (1024**3),
            )

    def generate(
        self,
        prompt: str,
        image_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Run inference on one image and return the generated response."""
        if image_path is None:
            raise ValueError("Open VLM extraction requires an image_path")
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if self._model is None or self._processor is None:
            self._load_model()
        import torch
        from PIL import Image

        start = time.perf_counter()
        inputs = None
        generated_ids = None
        try:
            with Image.open(image_path) as image:
                image_rgb = image.convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_rgb},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[text],
                images=[image_rgb],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            response_ids = generated_ids[:, prompt_length:]
            result = self._processor.batch_decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            logger.info(
                "Inference completed in %.2f seconds",
                time.perf_counter() - start,
            )
            return result
        finally:
            del inputs
            del generated_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class DePlotBackend(BaseExtractionBackend):
    """Backend adapter that exposes DePlot through the extraction interface."""

    def __init__(
        self,
        predictor: Optional[Callable[[Union[str, Path]], str]] = None,
        **model_kwargs: Any,
    ) -> None:
        self._predictor = predictor
        self._model_kwargs = model_kwargs

    def generate(self, prompt: str, image_path: Optional[Union[str, Path]] = None) -> str:
        """Runs DePlot on an image, loading the model only when first needed."""
        if image_path is None:
            raise ValueError("DePlot extraction requires an image_path")

        if self._predictor is None:
            from src.models.deplot import DePlotHarness

            model = DePlotHarness(**self._model_kwargs)
            self._predictor = model.predict

        return self._predictor(image_path)


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
            backend: Backend instance or name ('mock', 'open_vlm', 'api_vlm', 'deplot').
            target_format: Expected output format ('markdown' or 'deplot').
        """
        self.target_format = target_format
        if isinstance(backend, BaseExtractionBackend):
            self.backend = backend
        elif backend == "deplot":
            self.backend = DePlotBackend()
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
