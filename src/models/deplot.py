"""Module B5: DePlot Model Harness Stub.

Provides a lightweight stub for Google's DePlot visual chart-to-table model.
"""
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Pix2StructForConditionalGeneration,
)


class DePlotHarness:
    """
    Extract a linearized table from a chart image using DePlot.
    """

    MODEL_NAME = "google/deplot"

    PROMPT = "Generate underlying data table of the figure below:"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(
            self.model_name
        )

        self.model = (
            Pix2StructForConditionalGeneration.from_pretrained(
                self.model_name
            )
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_path: str | Path) -> str:
        """
        Generate a table representation from a chart image.

        Args:
            image_path: Path to the chart image.

        Returns:
            Model-generated table representation.
        """
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            inputs = self.processor(
                images=image,
                text=self.PROMPT,
                return_tensors="pt",
            )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        predictions = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )

        result = self.processor.decode(
            predictions[0],
            skip_special_tokens=True,
        )

        return result.strip()