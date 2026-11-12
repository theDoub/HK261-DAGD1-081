from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType

from PIL import Image
import pytest

from src.models.harness import OpenVLMBackend, VLMTableHarness


class FakeTensor:
    def __init__(self, values, shape=None):
        self.values = values
        self.shape = shape or (1, len(values[0]) if values and isinstance(values[0], list) else len(values))

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row_idx, col_idx = item
            rows = self.values[row_idx]
            if isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], list):
                selected = [r[col_idx] for r in rows]
            elif isinstance(rows, list):
                selected = rows[col_idx]
            else:
                selected = rows
            return FakeTensor(selected)
        value = self.values[item]
        return FakeTensor(value) if isinstance(value, list) else value

    def to(self, device):
        return self


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.messages = None
        self.processor_args = None
        self.decode_args = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "formatted image prompt"

    def __call__(self, **kwargs):
        self.processor_args = kwargs
        return FakeBatch({"input_ids": FakeTensor([[10, 11]], shape=(1, 2))})

    def decode(self, token_ids, **kwargs):
        self.decode_args = (token_ids.values, kwargs)
        return "```markdown\n| label | value |\n|---|---|\n| A | 9 |\n```"

    def batch_decode(self, token_ids, **kwargs):
        self.decode_args = (token_ids.values, kwargs)
        return ["```markdown\n| label | value |\n|---|---|\n| A | 9 |\n```"]


class FakeModel:
    def __init__(self):
        self.device = None
        self.evaluating = False
        self.generate_args = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True

    def generate(self, **kwargs):
        self.generate_args = kwargs
        return FakeTensor([[10, 11, 20, 21]])


def install_fake_transformers(monkeypatch, processor, model):
    transformers = ModuleType("transformers")

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            assert model_name == "test/qwen"
            return processor

    class FakeQwenModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            assert model_name == "test/qwen"
            if "device_map" in kwargs:
                model.device = "cuda"
            return model

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    transformers.AutoProcessor = FakeAutoProcessor
    transformers.Qwen2_5_VLForConditionalGeneration = FakeQwenModel
    transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    import torch as real_torch
    monkeypatch.setattr(real_torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(real_torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(real_torch.cuda, "memory_allocated", lambda: 0)
    if not hasattr(real_torch, "float16"):
        monkeypatch.setattr(real_torch, "float16", "float16")


def test_open_vlm_runs_qwen_and_returns_only_generated_text(tmp_path, monkeypatch):
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (4, 4), color="white").save(image_path)
    processor = FakeProcessor()
    model = FakeModel()
    install_fake_transformers(monkeypatch, processor, model)

    backend = OpenVLMBackend(
        model_name="test/qwen",
        device="auto",
        max_new_tokens=37,
    )
    result = VLMTableHarness(backend=backend).extract_table(
        image_path,
        prompt="extract the plotted values",
    )

    assert result.parse_ok is True
    assert result.backend_name == "OpenVLMBackend"
    assert result.table is not None
    assert result.table.rows == [["A", 9]]
    assert result.raw_text.startswith("```markdown")
    assert model.device == "cuda"
    assert model.evaluating is True
    assert model.generate_args["max_new_tokens"] == 37
    assert processor.processor_args["text"] == ["formatted image prompt"]
    assert processor.processor_args["padding"] is True
    assert processor.messages[0]["content"][1]["text"] == "extract the plotted values"
    assert processor.decode_args == (
        [[20, 21]],
        {"skip_special_tokens": True, "clean_up_tokenization_spaces": False},
    )


def test_open_vlm_reuses_lazy_loaded_model(tmp_path, monkeypatch):
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (4, 4), color="white").save(image_path)
    processor = FakeProcessor()
    model = FakeModel()
    install_fake_transformers(monkeypatch, processor, model)
    backend = OpenVLMBackend(model_name="test/qwen")

    backend.generate("first prompt", image_path)
    backend.generate("second prompt", image_path)

    assert backend._model is model
    assert backend._processor is processor
    assert processor.messages[0]["content"][1]["text"] == "second prompt"


def test_open_vlm_auto_uses_cpu_when_cuda_is_unavailable(tmp_path, monkeypatch):
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (4, 4), color="white").save(image_path)
    processor = FakeProcessor()
    model = FakeModel()
    install_fake_transformers(monkeypatch, processor, model)
    import torch as real_torch
    monkeypatch.setattr(real_torch.cuda, "is_available", lambda: False)

    backend = OpenVLMBackend(model_name="test/qwen", device="auto", quantization="none")
    backend.generate("chart prompt", image_path)

    assert model.device == "cpu"


def test_open_vlm_rejects_missing_image_before_loading_model(tmp_path):
    backend = OpenVLMBackend()

    with pytest.raises(FileNotFoundError, match="Image not found"):
        backend.generate("prompt", tmp_path / "missing.png")

    assert backend._model is None
    assert backend._processor is None


def test_open_vlm_requires_image_path():
    backend = OpenVLMBackend()

    with pytest.raises(
        ValueError,
        match="Open VLM extraction requires an image_path",
    ):
        backend.generate("prompt")


def test_open_vlm_rejects_non_positive_token_limit():
    with pytest.raises(ValueError, match="max_new_tokens must be greater than zero"):
        OpenVLMBackend(max_new_tokens=0)


def test_open_vlm_cuda_unavailable_is_reported_by_harness(tmp_path, monkeypatch):
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (4, 4), color="white").save(image_path)

    import torch as real_torch
    monkeypatch.setattr(real_torch.cuda, "is_available", lambda: False)

    harness = VLMTableHarness(
        backend=OpenVLMBackend(model_name="test/qwen", device="cuda")
    )
    result = harness.extract_table(image_path)

    assert result.parse_ok is False
    assert result.error_message == (
        "Backend failure: CUDA device 'cuda' requested, but CUDA is unavailable"
    )
