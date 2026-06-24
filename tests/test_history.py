"""Round-trip tests for imgprompt.history.

LAST_GENERATION_FILE is a module-level constant pointing at the repo root, so we
monkeypatch it to a tmp path to avoid touching the real file.
"""

import imgprompt.history as history
from imgprompt.providers.base import GenerationRequest


def _patch_file(monkeypatch, tmp_path):
    target = tmp_path / ".last_generation.json"
    monkeypatch.setattr(history, "LAST_GENERATION_FILE", str(target))
    return target


def test_save_then_load_roundtrip(monkeypatch, tmp_path):
    _patch_file(monkeypatch, tmp_path)
    req = GenerationRequest(
        prompt="a red fox",
        model="gpt-image-2",
        aspect_ratio="1:1",
        res_key="1K",
        quality_key="high",
        images=["a.png", "b.png"],
        width=1024,
        height=1024,
    )
    history.save_last_generation("openai", req)
    loaded = history.load_last_generation()
    assert loaded is not None
    provider, loaded_req = loaded
    assert provider == "openai"
    assert loaded_req == req


def test_load_missing_file_returns_none(monkeypatch, tmp_path):
    _patch_file(monkeypatch, tmp_path)  # file does not exist yet
    assert history.load_last_generation() is None


def test_load_corrupt_json_returns_none(monkeypatch, tmp_path):
    target = _patch_file(monkeypatch, tmp_path)
    target.write_text("{ not valid json")
    assert history.load_last_generation() is None


def test_load_missing_keys_returns_none(monkeypatch, tmp_path):
    target = _patch_file(monkeypatch, tmp_path)
    target.write_text('{"provider": "openai"}')  # no "request" key
    assert history.load_last_generation() is None


def test_multiline_prompt_survives_roundtrip(monkeypatch, tmp_path):
    _patch_file(monkeypatch, tmp_path)
    req = GenerationRequest(
        prompt="line one\nline two\nline three",
        model="gpt-image-2",
        aspect_ratio="1:1",
        res_key="1K",
        quality_key="high",
    )
    history.save_last_generation("openai", req)
    _, loaded_req = history.load_last_generation()
    assert loaded_req.prompt == "line one\nline two\nline three"
