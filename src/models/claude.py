"""Claude-specific inference harness."""

from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image

from src.models.api_cache import APIResponseCache


class ClaudeHarness:
    """Runs Claude vision requests and caches successful responses."""

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    MEDIA_TYPES = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }

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
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required to use the Claude API")
        try:
            import anthropic
        except ImportError as error:
            raise RuntimeError(
                "Claude SDK is missing; install the anthropic dependency"
            ) from error
        return anthropic.Anthropic(api_key=key)

    def _generate_uncached(self, image_bytes: bytes, prompt: str) -> str:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format or ""
        media_type = self.MEDIA_TYPES.get(image_format)
        if media_type is None:
            raise ValueError(f"Unsupported image format for Claude API: {image_format}")

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
            and isinstance(getattr(block, "text", None), str)
        )

    def generate(
        self,
        prompt: str,
        image_path: Optional[Union[str, Path]] = None,
    ) -> str:
        if image_path is None:
            raise ValueError("Claude extraction requires an image_path")
        return self.cache.get_or_create(
            provider="claude",
            model_name=self.model_name,
            prompt=prompt,
            image_path=image_path,
            generate=self._generate_uncached,
        )
