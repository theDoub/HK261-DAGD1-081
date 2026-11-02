import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

import main as cli
from main import select_samples
from src.models.claude import ClaudeHarness
from src.models.gemini import GeminiHarness
from src.models.harness import (
    ClaudeVLMBackend,
    GeminiVLMBackend,
    MockBackend,
    VLMTableHarness,
)


class FakeGeminiClient:
    def __init__(self, response_text="| Category | Value |\n|---|---|\n| A | 1 |"):
        self.calls = []
        self.response_text = response_text
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.response_text)


class FakeClaudeClient:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="| Category | Value |\n|---|---|\n| B | 2 |"),
                SimpleNamespace(type="image", text=None),
            ]
        )


def create_chart(path: Path, color=(20, 40, 60)) -> Path:
    Image.new("RGB", (4, 4), color=color).save(path, format="PNG")
    return path


def test_gemini_request_and_persistent_cache(tmp_path):
    image_path = create_chart(tmp_path / "chart.png")
    cache_dir = tmp_path / "cache"
    first_client = FakeGeminiClient()
    first_backend = GeminiHarness(client=first_client, cache_dir=cache_dir)

    first_response = first_backend.generate("Extract the table", image_path)
    second_client = FakeGeminiClient()
    second_backend = GeminiHarness(client=second_client, cache_dir=cache_dir)
    cached_response = second_backend.generate("Extract the table", image_path)

    assert first_response.startswith("| Category")
    assert cached_response == first_response
    assert len(first_client.calls) == 1
    assert second_client.calls == []
    assert first_client.calls[0]["model"] == GeminiHarness.DEFAULT_MODEL
    assert first_client.calls[0]["contents"][0] == "Extract the table"
    assert isinstance(first_client.calls[0]["contents"][1], Image.Image)


def test_gemini_cache_key_uses_image_and_prompt(tmp_path):
    image_path = create_chart(tmp_path / "chart.png")
    client = FakeGeminiClient()
    backend = GeminiHarness(client=client, cache_dir=tmp_path / "cache")

    backend.generate("Prompt A", image_path)
    backend.generate("Prompt A", image_path)
    backend.generate("Prompt B", image_path)
    create_chart(image_path, color=(60, 40, 20))
    backend.generate("Prompt A", image_path)

    assert len(client.calls) == 3


def test_cache_can_be_disabled(tmp_path):
    image_path = create_chart(tmp_path / "chart.png")
    client = FakeGeminiClient()
    backend = GeminiHarness(
        client=client,
        cache_dir=tmp_path / "cache",
        cache_enabled=False,
    )

    backend.generate("Extract", image_path)
    backend.generate("Extract", image_path)

    assert len(client.calls) == 2
    assert not (tmp_path / "cache").exists()


def test_claude_request_encodes_chart_image(tmp_path):
    image_path = create_chart(tmp_path / "chart.png")
    client = FakeClaudeClient()
    backend = ClaudeHarness(client=client, cache_dir=tmp_path / "cache")

    response = backend.generate("Extract the chart", image_path)

    message = client.calls[0]["messages"][0]["content"]
    assert response.startswith("| Category")
    assert client.calls[0]["model"] == ClaudeHarness.DEFAULT_MODEL
    assert message[0]["type"] == "image"
    assert message[0]["source"]["media_type"] == "image/png"
    assert message[1] == {"type": "text", "text": "Extract the chart"}


def test_claude_uses_shared_persistent_cache(tmp_path):
    image_path = create_chart(tmp_path / "chart.png")
    cache_dir = tmp_path / "cache"
    first_client = FakeClaudeClient()
    first_backend = ClaudeHarness(client=first_client, cache_dir=cache_dir)

    first_response = first_backend.generate("Extract", image_path)
    second_client = FakeClaudeClient()
    second_backend = ClaudeHarness(client=second_client, cache_dir=cache_dir)
    cached_response = second_backend.generate("Extract", image_path)

    assert cached_response == first_response
    assert len(first_client.calls) == 1
    assert second_client.calls == []


def test_cache_is_isolated_by_provider(tmp_path):
    image_path = create_chart(tmp_path / "chart.png")
    cache_dir = tmp_path / "cache"
    gemini_client = FakeGeminiClient()
    claude_client = FakeClaudeClient()

    GeminiHarness(
        client=gemini_client,
        cache_dir=cache_dir,
    ).generate("Extract", image_path)
    ClaudeHarness(
        client=claude_client,
        cache_dir=cache_dir,
    ).generate("Extract", image_path)

    assert len(gemini_client.calls) == 1
    assert len(claude_client.calls) == 1
    assert len(list(cache_dir.glob("*.txt"))) == 2


def test_extraction_record_identifies_selected_api_provider(tmp_path):
    image_path = create_chart(tmp_path / "chart.png")
    backend = GeminiVLMBackend(
        client=FakeGeminiClient(),
        cache_dir=tmp_path / "cache",
    )
    result = VLMTableHarness(backend=backend).extract_table(image_path)

    assert result.backend_name == "GeminiVLMBackend"
    assert result.parse_ok is True


@pytest.mark.parametrize(
    ("backend_name", "backend_class"),
    [
        ("claude", ClaudeVLMBackend),
        ("gemini", GeminiVLMBackend),
    ],
)
def test_vlm_harness_selects_lazy_hosted_backend(backend_name, backend_class):
    harness = VLMTableHarness(backend=backend_name)

    assert isinstance(harness.backend, backend_class)
    assert harness.backend._harness is None


@pytest.mark.parametrize(
    ("backend_class", "environment_variable"),
    [
        (GeminiHarness, "GEMINI_API_KEY"),
        (ClaudeHarness, "ANTHROPIC_API_KEY"),
    ],
)
def test_api_backend_requires_provider_key(
    backend_class,
    environment_variable,
    monkeypatch,
):
    monkeypatch.delenv(environment_variable, raising=False)

    with pytest.raises(ValueError, match=environment_variable):
        backend_class()


def test_api_backend_requires_existing_image(tmp_path):
    backend = GeminiHarness(client=FakeGeminiClient(), cache_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="Image not found"):
        backend.generate("Extract", tmp_path / "missing.png")


def test_select_samples_applies_optional_positive_limit():
    samples = ["one", "two", "three"]

    assert select_samples(samples, 2) == ["one", "two"]
    assert select_samples(samples, None) is samples
    with pytest.raises(ValueError, match="positive integer"):
        select_samples(samples, 0)


def test_validate_image_path_accepts_supported_image_formats(tmp_path):
    for extension, image_format in (
        (".png", "PNG"),
        (".jpg", "JPEG"),
        (".jpeg", "JPEG"),
        (".webp", "WEBP"),
    ):
        image_path = tmp_path / f"chart{extension}"
        Image.new("RGB", (4, 4)).save(image_path, format=image_format)

        assert cli.validate_image_path(image_path) == image_path


def test_validate_image_path_rejects_missing_and_unsupported_images(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        cli.validate_image_path(tmp_path / "missing.png")

    image_path = tmp_path / "chart.gif"
    Image.new("RGB", (4, 4)).save(image_path, format="GIF")
    with pytest.raises(ValueError, match="Unsupported image extension"):
        cli.validate_image_path(image_path)


def test_run_single_image_prints_and_writes_json_result(tmp_path, monkeypatch, capsys):
    image_path = create_chart(tmp_path / "chart.png")
    output_path = tmp_path / "nested" / "result.json"
    harness = VLMTableHarness(
        backend=MockBackend("| Category | Value |\n|---|---|\n| A | 1 |")
    )
    monkeypatch.setattr(cli, "load_project_config", lambda project_root: {})
    monkeypatch.setattr(
        cli,
        "build_extraction_harness",
        lambda *args, **kwargs: harness,
    )

    assert cli.run_single_image(
        image_path,
        backend_name="gemini",
        output_path=output_path,
    )

    output = capsys.readouterr().out
    assert "Extracted table (Markdown)" in output
    assert "Raw model response:" in output
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["image_path"] == str(image_path.resolve())
    assert record["backend"] == "MockBackend"
    assert record["parse_ok"] is True
    assert record["table"]["columns"] == ["category", "value"]
    assert record["error_message"] is None


def test_run_single_image_reports_invalid_file_without_running_backend(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        cli,
        "build_extraction_harness",
        lambda *args, **kwargs: pytest.fail("backend must not run"),
    )

    assert not cli.run_single_image(tmp_path / "missing.png")
    assert "does not exist" in capsys.readouterr().err


def test_cli_routes_single_image_and_preserves_cache_option(monkeypatch, tmp_path):
    received = {}

    def fake_run_single_image(**kwargs):
        received.update(kwargs)
        return True

    monkeypatch.setattr(cli, "run_single_image", fake_run_single_image)

    assert cli.main(
        [
            "--image",
            str(tmp_path / "chart.png"),
            "--backend",
            "claude",
            "--output",
            str(tmp_path / "result.json"),
            "--no-api-cache",
        ]
    ) == 0
    assert received == {
        "image_path": str(tmp_path / "chart.png"),
        "backend_name": "claude",
        "output_path": str(tmp_path / "result.json"),
        "disable_api_cache": True,
    }


def test_cli_rejects_conflicting_modes_and_image_dataset_options():
    with pytest.raises(SystemExit):
        cli.main(["--image", "chart.png", "--chartqa"])
    with pytest.raises(SystemExit):
        cli.main(["--image", "chart.png", "--max-samples", "1"])
    with pytest.raises(SystemExit):
        cli.main(["--backend", "gemini"])


@pytest.mark.parametrize(
    ("backend_class", "model_name"),
    [
        (ClaudeVLMBackend, "claude-test-model"),
        (GeminiVLMBackend, "gemini-test-model"),
    ],
)
def test_backend_lazily_creates_provider_harness(backend_class, model_name, tmp_path):
    image_path = create_chart(tmp_path / "chart.png")
    constructed = []
    client = object()

    class FakeHarness:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def generate(self, prompt, requested_image_path):
            return f"{prompt}: {requested_image_path.name}"

    backend = backend_class(
        model_name=model_name,
        cache_dir=tmp_path / "cache",
        cache_enabled=False,
        api_key="test-key",
        client=client,
        harness_factory=FakeHarness,
    )
    assert constructed == []

    result = backend.generate("Extract", image_path)

    assert result == "Extract: chart.png"
    assert constructed == [
        {
            "model_name": model_name,
            "cache_dir": tmp_path / "cache",
            "cache_enabled": False,
            "api_key": "test-key",
            "client": client,
        }
    ]


@pytest.mark.parametrize(
    "backend_class",
    [ClaudeVLMBackend, GeminiVLMBackend],
)
def test_backend_caches_harness_initialization_failure(backend_class, tmp_path):
    image_path = create_chart(tmp_path / "chart.png")
    factory_calls = []

    def failing_factory(**kwargs):
        factory_calls.append(kwargs)
        raise RuntimeError("SDK initialization failed")

    backend = backend_class(
        harness_factory=failing_factory,
        client=object(),
    )

    with pytest.raises(RuntimeError, match="SDK initialization failed"):
        backend.generate("Extract", image_path)
    with pytest.raises(RuntimeError, match="harness initialization failed"):
        backend.generate("Extract", image_path)

    assert len(factory_calls) == 1
