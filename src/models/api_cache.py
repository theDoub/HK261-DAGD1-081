"""Persistent response cache shared by hosted vision-model backends."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)


class APIResponseCache:
    """Caches successful responses using model inputs and image bytes as the key."""

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(".cache") / "chart_to_table" / "api_responses"
        )
        self.enabled = enabled

    @staticmethod
    def _cache_key(
        provider: str,
        model_name: str,
        prompt: str,
        image_bytes: bytes,
    ) -> str:
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        key_data = json.dumps(
            [provider, model_name, prompt, image_digest],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(key_data.encode("utf-8")).hexdigest()

    def get_or_create(
        self,
        *,
        provider: str,
        model_name: str,
        prompt: str,
        image_path: Union[str, Path],
        generate: Callable[[bytes, str], str],
    ) -> str:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_bytes = image_path.read_bytes()
        cache_key = self._cache_key(provider, model_name, prompt, image_bytes)
        cache_path = self.cache_dir / f"{cache_key}.txt"
        if self.enabled and cache_path.is_file():
            logger.debug("%s response cache hit: %s", provider, cache_key)
            cached_response = cache_path.read_text(encoding="utf-8")
            if not cached_response.strip():
                raise RuntimeError(
                    f"Cached {provider} API response is empty: {cache_path}"
                )
            return cached_response

        logger.info(
            "Calling %s API model [%s] for image: %s",
            provider,
            model_name,
            image_path,
        )
        result = generate(image_bytes, prompt)
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError(f"{provider} API returned an empty text response")
        result = result.strip()

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.cache_dir,
                    prefix=f".{cache_key}.",
                    suffix=".tmp",
                    delete=False,
                ) as cache_file:
                    cache_file.write(result)
                    temporary_path = Path(cache_file.name)
                os.replace(temporary_path, cache_path)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
        return result
