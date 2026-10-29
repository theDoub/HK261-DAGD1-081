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
        self.shape = shape

    def __getitem__(self, item):
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
        return FakeBatch({"input_ids": FakeTensor([10, 11], shape=(1, 2))})

    def decode(self, token_ids, **kwargs):
        self.decode_args = (token_ids.values, kwargs)
        return "```markdown\n| label | value |\n|---|---|\n| A | 9 |\n```"


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
        def from_pretrained(model_name):
            assert model_name == "test/qwen"
            return processor

    class FakeQwenModel:
        @staticmethod
        def from_pretrained(model_name):
            assert model_name == "test/qwen"
            return model

    transformers.AutoProcessor = FakeAutoProcessor
    transformers.Qwen2_5_VLForConditionalGeneration = FakeQwenModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    torch = ModuleType("torch")
    torch.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)


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
        [20, 21],
        {"skip_special_tokens": True},
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
    torch = sys.modules["torch"]
    torch.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: False)})()

    backend = OpenVLMBackend(model_name="test/qwen", device="auto")
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

    torch = ModuleType("torch")
    torch.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: False)})()
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)

    harness = VLMTableHarness(
        backend=OpenVLMBackend(model_name="test/qwen", device="cuda")
    )
    result = harness.extract_table(image_path)

    assert result.parse_ok is False
    assert result.error_message == (
        "Backend failure: CUDA device 'cuda' requested, but CUDA is not available"
    )
