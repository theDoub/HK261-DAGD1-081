"""Gemini-specific inference harness."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image

from src.models.api_cache import APIResponseCache


class GeminiHarness:
    """Runs Gemini vision requests and caches successful responses."""

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[Union[str, Path]] = None,
        *,
        cache_enabled: bool = True,
        api_key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self.model_name = model_name
        self.cache = APIResponseCache(cache_dir, enabled=cache_enabled)
        self.client = client if client is not None else self._create_client(api_key)

    @staticmethod
    def _create_client(api_key: Optional[str]) -> Any:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required to use the Gemini API")
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError(
                "Gemini SDK is missing; install the google-genai dependency"
            ) from error
        return genai.Client(api_key=key)

    def _generate_uncached(self, image_bytes: bytes, prompt: str) -> str:
        with Image.open(BytesIO(image_bytes)) as image:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image.convert("RGB")],
            )
        return response.text

    def generate(
        self,
        prompt: str,
        image_path: Optional[Union[str, Path]] = None,
    ) -> str:
        if image_path is None:
            raise ValueError("Gemini extraction requires an image_path")
        return self.cache.get_or_create(
            provider="gemini",
            model_name=self.model_name,
            prompt=prompt,
            image_path=image_path,
            generate=self._generate_uncached,
        )
