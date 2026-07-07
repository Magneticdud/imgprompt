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


def test_is_dual_survives_roundtrip(monkeypatch, tmp_path):
    _patch_file(monkeypatch, tmp_path)
    req = GenerationRequest(
        prompt="combine IMG_1 and IMG_2",
        model="bytedance-seed/seedream-4.5",
        aspect_ratio="1:1",
        res_key="1024x1024",
        quality_key="1K",
        images=["a.png", "b.png"],
        is_dual=True,
    )
    history.save_last_generation("OpenRouter", req)
    _, loaded_req = history.load_last_generation()
    assert loaded_req.is_dual is True
    assert loaded_req.is_batch is False


def test_pre_dual_history_file_loads_with_default(monkeypatch, tmp_path):
    """A .last_generation.json written before is_dual existed has no such
    key; loading must default it to False (plain batch semantics)."""
    import dataclasses
    import json

    target = _patch_file(monkeypatch, tmp_path)
    req = GenerationRequest(
        prompt="x",
        model="gpt-image-2",
        aspect_ratio="1:1",
        res_key="1K",
        quality_key="high",
        images=["a.png", "b.png"],
    )
    data = dataclasses.asdict(req)
    del data["is_dual"]
    target.write_text(json.dumps({"provider": "OpenAI", "request": data}))
    _, loaded_req = history.load_last_generation()
    assert loaded_req.is_dual is False
    assert loaded_req.is_batch is True


def test_recraft_style_extras_survive_roundtrip(monkeypatch, tmp_path):
    """--replay must reproduce the exact style/colors of a Recraft run."""
    _patch_file(monkeypatch, tmp_path)
    req = GenerationRequest(
        prompt="a fox logo",
        model="recraft/recraft-v4.1-vector",
        aspect_ratio="Auto",
        res_key="model default",
        quality_key="Standard",
        extras={"style": "vector_illustration", "colors": ["#FFAA00", "#112233"]},
    )
    history.save_last_generation("OpenRouter", req)
    _, loaded_req = history.load_last_generation()
    assert loaded_req.extras["style"] == "vector_illustration"
    assert loaded_req.extras["colors"] == ["#FFAA00", "#112233"]


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
