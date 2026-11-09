"""Main entry point: End-to-end Smoke Test for Chart-to-Table Pipeline.

Workflow:
1. Loads synthetic ground-truth tables from Module B3 (data/sanity/synthetic_dataset.json).
2. Extracts tables using Module B5 VLMTableHarness (MockBackend).
3. Parses output and normalizes cells into Module B1 TableSchema and TableTriplets.
4. Computes Module B2 metrics (RMS-F1, RNSS, Value-Recall@5%).
5. Generates Module B6 aggregations (by chart type, complexity, and error buckets).
6. Logs comprehensive verification summary.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, TypeVar, Union

# Setup structured logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SmokeTest")
Sample = TypeVar("Sample")

if TYPE_CHECKING:
    from src.models.harness import VLMTableHarness


def run_smoke_test() -> None:
    """Executes the complete modular smoke test across Modules B1 to B6."""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader.synthetic import load_synthetic_tables
    from src.evaluation.evaluator import PipelineEvaluator
    from src.models.harness import MockBackend, VLMTableHarness

    logger.info("=" * 65)
    logger.info("STARTING CHART-TO-TABLE PIPELINE SMOKE TEST (MODULES B1-B6)")
    logger.info("=" * 65)

    # 1. Module B3: Load synthetic sanity benchmark tables
    tables = load_synthetic_tables()
    logger.info("Module B3: Loaded %d synthetic benchmark samples.", len(tables))

    # Preview first sample in Markdown (Module B1)
    sample_gold = tables[0]
    print("\n" + "=" * 50)
    print(" [Module B1] Gold Table Preview (Markdown):")
    print("=" * 50)
    print(sample_gold.to_markdown())
    print(f"Total Triplet Cells: {len(sample_gold.triplets)}")
    print("=" * 50 + "\n")

    # 2. Module B5: Initialize extraction harness with MockBackend
    # We configure the mock to match sample_001 perfectly to demonstrate high F1
    mock_md = sample_gold.to_markdown()
    mock_backend = MockBackend(predefined_table_md=mock_md)
    harness = VLMTableHarness(backend=mock_backend, target_format="markdown")
    logger.info("Module B5: Initialized VLMTableHarness with MockBackend.")

    # 3. Module B6 & B2: Run evaluation pipeline
    evaluator = PipelineEvaluator(harness=harness)
    results = evaluator.evaluate_dataset(tables)

    print("\n" + "=" * 50)
    print(" [Module B6] Evaluation & Error Analysis Summary:")
    print("=" * 50)
    print(f"Total Evaluated Samples: {results['total_samples']}")
    print(f"Mean RMS-F1 Score:       {results['mean_rms_f1']:.4f}")
    print(f"Mean RNSS Score:         {results['mean_rnss']:.4f}")
    print(f"Mean Value-Recall@5%:    {results['mean_value_recall_at_5']:.4f}")
    print("-" * 50)
    print("Breakdown by Chart Type:")
    print(json.dumps(results["breakdown_by_chart_type"], indent=2))
    print("Breakdown by Complexity:")
    print(json.dumps(results["breakdown_by_complexity"], indent=2))
    print("Error Distribution:")
    print(json.dumps(results["error_analysis"], indent=2))
    print("=" * 50 + "\n")

    logger.info("SUCCESS: All modules (B1-B6) executed and validated cleanly!")


def select_samples(items: Sequence[Sample], limit: Optional[int]) -> Sequence[Sample]:
    """Applies an optional positive sample cap without changing the input sequence."""
    if limit is None:
        return items
    if limit < 1:
        raise ValueError("sample limit must be a positive integer or None")
    return items[:limit]


def load_project_config(project_root: Path) -> Dict[str, Any]:
    """Loads the project YAML configuration."""
    import yaml

    config_path = project_root / "configs" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return config


def build_extraction_harness(
    backend_name: str,
    config: Dict[str, Any],
    project_root: Path,
    *,
    disable_api_cache: bool = False,
) -> VLMTableHarness:
    """Creates the configured extraction harness for the selected backend."""
    from src.models.harness import (
        ClaudeVLMBackend,
        DePlotBackend,
        GeminiVLMBackend,
        OpenVLMBackend,
        VLMTableHarness,
    )

    if backend_name == "deplot":
        deplot_config = config.get("deplot", {})
        return VLMTableHarness(
            backend=DePlotBackend(
                model_name=deplot_config.get("model_name_or_path", "google/deplot"),
                max_new_tokens=deplot_config.get("max_new_tokens", 512),
            ),
            target_format="deplot",
        )

    if backend_name == "qwen":
        qwen_config = config.get("qwen", {})
        return VLMTableHarness(
            backend=OpenVLMBackend(
                model_name=qwen_config.get(
                    "model_name_or_path", OpenVLMBackend.DEFAULT_MODEL_NAME
                ),
                device=qwen_config.get("device", "auto"),
                max_new_tokens=qwen_config.get("max_new_tokens", 256),
                quantization=qwen_config.get("quantization", "4bit"),
            ),
            target_format="markdown",
            prompt=qwen_config.get("prompt"),
        )

    if backend_name not in ("gemini", "claude"):
        raise ValueError(
            "backend_name must be 'deplot', 'qwen', 'gemini', or 'claude'"
        )

    api_config = config.get("api_vlm", {})
    provider_config = api_config.get(backend_name, {})
    cache_dir = Path(
        api_config.get("cache_dir", ".cache/chart_to_table/api_responses")
    )
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir

    backend_class = (
        GeminiVLMBackend if backend_name == "gemini" else ClaudeVLMBackend
    )
    return VLMTableHarness(
        backend=backend_class(
            model_name=provider_config.get(
                "model_name", backend_class.DEFAULT_MODEL
            ),
            cache_dir=cache_dir,
            cache_enabled=api_config.get("cache_enabled", True)
            and not disable_api_cache,
        ),
        target_format="markdown",
    )


def validate_image_path(image_path: Union[str, Path]) -> Path:
    """Validates that the input is a readable, supported static chart image."""
    path = Path(image_path).expanduser()
    supported_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    if path.suffix.lower() not in supported_extensions:
        raise ValueError(
            "Unsupported image extension; use PNG, JPG/JPEG, or WEBP"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    from PIL import Image

    try:
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError(
                    f"Image content must be PNG, JPEG, or WEBP: {path}"
                )
            if getattr(image, "is_animated", False):
                raise ValueError(f"Animated images are not supported: {path}")
            image.verify()
    except (OSError, SyntaxError) as error:
        raise ValueError(f"Unable to read image file {path}: {error}") from error
    return path


def run_single_image(
    image_path: Union[str, Path],
    backend_name: str = "deplot",
    output_path: Optional[Union[str, Path]] = None,
    disable_api_cache: bool = False,
) -> bool:
    """Extracts a table from one image, optionally writing a JSON result."""
    try:
        image_path = validate_image_path(image_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return False
    project_root = Path(__file__).resolve().parent
    config = load_project_config(project_root)
    harness = build_extraction_harness(
        backend_name,
        config,
        project_root,
        disable_api_cache=disable_api_cache,
    )
    result = harness.extract_table(image_path)

    print(f"Image:   {image_path.resolve()}")
    print(f"Backend: {result.backend_name}")
    print(f"Parsed:  {'yes' if result.parse_ok else 'no'}")
    if result.parse_ok and result.table is not None:
        print("\nExtracted table (Markdown):")
        print(result.table.to_markdown())
    if result.error_message:
        print(f"\nError: {result.error_message}")
    print("\nRaw model response:")
    print(result.raw_text or "<empty>")

    record = {
        "image_path": str(image_path.resolve()),
        "backend": result.backend_name,
        "raw_text": result.raw_text,
        "parse_ok": result.parse_ok,
        "error_message": result.error_message,
        "table": result.table.to_dict() if result.table is not None else None,
    }
    if output_path is not None:
        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as result_file:
            json.dump(record, result_file, ensure_ascii=False, indent=2)
            result_file.write("\n")
        print(f"\nResult JSON: {output_file.resolve()}")
    return result.parse_ok


def run_chartqa_evaluation(
    backend_name: str = "deplot",
    max_samples: Optional[int] = None,
    no_sample_limit: bool = False,
    disable_api_cache: bool = False,
) -> None:
    """Runs a local model or hosted vision API across the local ChartQA test split."""
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be a positive integer")

    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader.loader import ChartDataLoader
    from src.evaluation.evaluator import PipelineEvaluator

    config_path = project_root / "configs" / "config.yaml"
    config = load_project_config(project_root)

    data_config = config.get("data", {})
    dataset_root = (
        project_root
        / data_config.get("raw_dir", "data/raw")
        / "ChartQA_Dataset"
    )
    split = data_config.get("split", "test")
    items = ChartDataLoader(data_dir=dataset_root, split=split).scan_dataset()
    if not items:
        raise RuntimeError(f"No ChartQA samples found in {dataset_root}\\{split}")

    if backend_name in ("deplot", "qwen"):
        harness = build_extraction_harness(
            backend_name, config, project_root
        )
        sample_limit = max_samples
    else:
        api_config = config.get("api_vlm", {})
        if backend_name not in ("gemini", "claude"):
            raise ValueError("backend_name must be 'deplot', 'gemini', or 'claude'")
        harness = build_extraction_harness(
            backend_name,
            config,
            project_root,
            disable_api_cache=disable_api_cache,
        )
        if no_sample_limit:
            sample_limit = None
        elif max_samples is not None:
            sample_limit = max_samples
        else:
            sample_limit = api_config.get("sample_limit", 10)

    items = select_samples(items, sample_limit)
    evaluator = PipelineEvaluator(harness=harness, config_path=config_path)

    evaluation_config = config.get("evaluation", {})
    output_dir = project_root / evaluation_config.get(
        "output_dir", "outputs/predictions"
    )
    output_name = (
        f"chartqa_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    )
    predictions_path = output_dir / output_name
    summary = evaluator.evaluate_dataset(items, predictions_path=predictions_path)

    parse_ok_count = sum(record["parse_ok"] for record in summary["records"])
    print(f"Samples attempted: {summary['total_samples']}")
    print(f"Parse successes:   {parse_ok_count}")
    print(f"Parse failures:    {summary['total_samples'] - parse_ok_count}")
    print(f"Predictions:       {predictions_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Chart-to-table pipeline")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--chartqa",
        action="store_true",
        help="Evaluate the local ChartQA test split with a selected backend.",
    )
    input_group.add_argument(
        "--image",
        metavar="PATH",
        help="Extract a table from one PNG, JPG/JPEG, or WEBP image.",
    )
    parser.add_argument(
        "--backend",
        choices=("deplot", "qwen", "gemini", "claude"),
        help="Chart extraction backend for --image or --chartqa (default: deplot).",
    )
    sample_limit_group = parser.add_mutually_exclusive_group()
    sample_limit_group.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of ChartQA samples to evaluate.",
    )
    sample_limit_group.add_argument(
        "--no-sample-limit",
        action="store_true",
        help="Evaluate all ChartQA samples, overriding the hosted-API default cap.",
    )
    parser.add_argument(
        "--no-api-cache",
        action="store_true",
        help="Do not read or write the local response cache for Gemini/Claude.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write single-image extraction details and parsed table to a JSON file.",
    )
    args = parser.parse_args(argv)

    if args.image and (args.max_samples is not None or args.no_sample_limit):
        parser.error(
            "--max-samples and --no-sample-limit can only be used with --chartqa"
        )
    if args.chartqa and args.output:
        parser.error("--output can only be used with --image")
    if not args.chartqa and not args.image and (
        args.backend is not None
        or args.max_samples is not None
        or args.no_sample_limit
        or args.no_api_cache
        or args.output
    ):
        parser.error(
            "--backend, sample-limit options, --no-api-cache, and --output "
            "require --image or --chartqa"
        )

    if args.chartqa:
        run_chartqa_evaluation(
            backend_name=args.backend or "deplot",
            max_samples=args.max_samples,
            no_sample_limit=args.no_sample_limit,
            disable_api_cache=args.no_api_cache,
        )
    elif args.image:
        return 0 if run_single_image(
            image_path=args.image,
            backend_name=args.backend or "deplot",
            output_path=args.output,
            disable_api_cache=args.no_api_cache,
        ) else 1
    else:
        run_smoke_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
