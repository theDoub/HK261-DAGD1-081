"""Gemini-specific inference harness."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import random
import time
from typing import Any, Optional, Union

from PIL import Image

from src.models.api_cache import APIResponseCache


class GeminiHarness:
    """Runs Gemini vision requests and caches successful responses."""

    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[Union[str, Path]] = None,
        *,
        cache_enabled: bool = True,
        api_key: Optional[str] = None,
        client: Any = None,
        max_retries: int = 4,
        base_retry_delay: float = 2.0,
    ) -> None:
        self.model_name = model_name
        self.cache = APIResponseCache(cache_dir, enabled=cache_enabled)
        self.client = client if client is not None else self._create_client(api_key)
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay

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

    @staticmethod
    def _is_transient_error(error: Exception) -> bool:
        code = getattr(error, "code", None)
        if code in {429, 500, 503}:
            return True
        message = str(error).upper()
        return any(
            token in message
            for token in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "500")
        )

    def _generate_uncached(self, image_bytes: bytes, prompt: str) -> str:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[prompt, rgb_image],
                    )
                    return response.text
                except Exception as error:
                    if attempt < self.max_retries and self._is_transient_error(error):
                        delay = self.base_retry_delay * (2**attempt) + random.uniform(
                            0.0, 1.0
                        )
                        time.sleep(delay)
                        continue
                    raise
        raise RuntimeError("Gemini generation failed after retries")

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
