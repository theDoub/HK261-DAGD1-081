"""Run local Qwen chart-to-table extraction on a single image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence

import yaml

from src.models.harness import OpenVLMBackend, VLMTableHarness


def _load_qwen_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    qwen_config = config.get("qwen")
    if not isinstance(qwen_config, dict):
        raise ValueError(f"Missing 'qwen' configuration in {config_path}")
    return qwen_config


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extract a table from a chart image using local Qwen2.5-VL."
    )
    parser.add_argument("--image", required=True, type=Path, help="Chart image path")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "config.yaml",
        help="YAML config path (default: configs/config.yaml)",
    )
    parser.add_argument("--model", help="Override the configured Hugging Face model ID")
    parser.add_argument("--device", help="Override the configured device (auto/cuda/cpu)")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override the configured generation token limit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    qwen_config = _load_qwen_config(args.config)

    model_name = args.model or qwen_config.get(
        "model_name_or_path", OpenVLMBackend.DEFAULT_MODEL_NAME
    )
    device = args.device or qwen_config.get("device", "auto")
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else qwen_config.get("max_new_tokens", 512)
    )
    prompt = qwen_config.get(
        "prompt",
        "Extract all numerical values and categories from this chart into a clean "
        "GitHub-Flavored Markdown table. Output only the markdown table without "
        "conversational explanation.",
    )

    backend = OpenVLMBackend(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        quantization=qwen_config.get("quantization", "4bit"),
    )
    harness = VLMTableHarness(backend=backend, target_format="markdown")
    result = harness.extract_table(args.image, prompt=prompt)

    if not result.parse_ok or result.table is None:
        print(
            f"Qwen extraction failed ({result.backend_name}): "
            f"{result.error_message or 'model response could not be parsed'}",
            file=sys.stderr,
        )
        if result.raw_text:
            print("Raw model response:", file=sys.stderr)
            print(result.raw_text, file=sys.stderr)
        return 1

    print(result.table.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
