"""Module B5: Vision-Language Model (VLM) / Large Language Model (LLM) Client Stub.

Interfaces with language models or multimodal reasoning APIs (e.g. GPT-4o, Claude,
or local open-source LLMs) to perform Table Question Answering over extracted tables.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.format.schema import TableSchema

logger = logging.getLogger(__name__)


class VLMClient:
    """Client harness for Table QA reasoning over structured or linearized tables."""

    def __init__(
        self,
        provider: str = "vlm_stub",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 128,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initializes the Table QA client.

        Args:
            provider: Backend provider ('vlm_stub', 'openai', 'anthropic', 'huggingface').
            model_name: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            system_prompt: Optional custom system prompt.
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are an analytical assistant. Answer the user question based strictly on the provided table."
        )

    def build_prompt(self, table: TableSchema, question: str) -> str:
        """Constructs an aligned prompt containing the table representation and question.

        Args:
            table: Input TableSchema instance.
            question: Target query.

        Returns:
            str: Combined prompt string.
        """
        table_md = table.to_markdown()
        return (
            f"{self.system_prompt}\n\n"
            f"### Table Context:\n{table_md}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Answer (concise, factual):"
        )

    def answer_question(self, table: TableSchema, question: str) -> str:
        """Generates an answer to the question using the table context.

        Args:
            table: Extracted or ground-truth table.
            question: Question string.

        Returns:
            str: Predicted answer.
        """
        prompt = self.build_prompt(table, question)
        logger.debug("Prompt constructed for QA:\n%s", prompt)

        if self.provider == "vlm_stub":
            # Stub behavior: checks table metadata for ground truth or mock logic
            if table.metadata and "ground_truth_answer" in table.metadata:
                return str(table.metadata["ground_truth_answer"])
            return "Mock answer based on table context."

        # Real external API or Hugging Face LLM call stub:
        # response = client.chat.completions.create(model=self.model_name, messages=...)
        # return response.choices[0].message.content.strip()
        raise NotImplementedError(f"Provider '{self.provider}' not yet configured.")
