"""Wizard-step tests for imgedit.py (currently the Recraft style step).

questionary is monkeypatched at the imgedit module boundary: each fake
question returns the next scripted answer, so the tests drive the step
functions without a TTY.
"""

import sys

import pytest

import imgedit
from imgedit import (
    BACK_OPTION,
    RECRAFT_CUSTOM_STYLE,
    is_recraft_model,
    run_replay,
    step_recraft_style,
)
from imgprompt.providers.base import GenerationRequest


class _FakeQuestion:
    def __init__(self, answer):
        self._answer = answer

    def ask(self):
        return self._answer


def _script(monkeypatch, select_answers=(), text_answers=()):
    """Queue scripted answers for questionary.select / questionary.text."""
    selects = list(select_answers)
    texts = list(text_answers)
    monkeypatch.setattr(
        imgedit.questionary,
        "select",
        lambda *a, **k: _FakeQuestion(selects.pop(0)),
    )
    monkeypatch.setattr(
        imgedit.questionary,
        "text",
        lambda *a, **k: _FakeQuestion(texts.pop(0)),
    )


class TestRecraftGate:
    def test_recraft_on_openrouter_is_gated_in(self):
        assert is_recraft_model("OpenRouter", "recraft/recraft-v4.1") is True
        assert is_recraft_model("OpenRouter", "recraft/recraft-v4.1-pro-vector") is True

    def test_non_recraft_openrouter_models_stay_out(self):
        assert is_recraft_model("OpenRouter", "openai/gpt-5.4-image-2") is False
        assert is_recraft_model("OpenRouter", "bytedance-seed/seedream-4.5") is False

    def test_other_providers_stay_out(self):
        assert is_recraft_model("OpenAI", "gpt-image-2") is False
        assert is_recraft_model(None, None) is False


class TestStepRecraftStyle:
    def test_accepting_a_slug_with_no_colors(self, monkeypatch):
        _script(monkeypatch, select_answers=["realistic_image"], text_answers=[""])
        assert step_recraft_style("recraft/recraft-v4.1") == ("realistic_image", [])

    def test_go_back_returns_back_option(self, monkeypatch):
        _script(monkeypatch, select_answers=[BACK_OPTION])
        assert step_recraft_style("recraft/recraft-v4.1") == BACK_OPTION

    def test_cancel_returns_none(self, monkeypatch):
        _script(monkeypatch, select_answers=[None])
        assert step_recraft_style("recraft/recraft-v4.1") is None

    def test_custom_slug_is_accepted(self, monkeypatch):
        _script(
            monkeypatch,
            select_answers=[RECRAFT_CUSTOM_STYLE],
            text_answers=["hand_drawn", ""],
        )
        assert step_recraft_style("recraft/recraft-v4.1") == ("hand_drawn", [])

    def test_invalid_custom_slug_reprompts(self, monkeypatch, capsys):
        _script(
            monkeypatch,
            select_answers=[RECRAFT_CUSTOM_STYLE],
            text_answers=["Not A Slug!", "icon", ""],
        )
        assert step_recraft_style("recraft/recraft-v4.1") == ("icon", [])
        assert "lowercase" in capsys.readouterr().out

    def test_empty_custom_slug_backs_out(self, monkeypatch):
        _script(
            monkeypatch,
            select_answers=[RECRAFT_CUSTOM_STYLE],
            text_answers=["   "],
        )
        assert step_recraft_style("recraft/recraft-v4.1") == BACK_OPTION

    def test_valid_colors_csv_is_parsed_and_trimmed(self, monkeypatch):
        _script(
            monkeypatch,
            select_answers=["digital_illustration"],
            text_answers=[" #FFAA00 , #112233 "],
        )
        slug, colors = step_recraft_style("recraft/recraft-v4.1")
        assert slug == "digital_illustration"
        assert colors == ["#FFAA00", "#112233"]

    def test_invalid_colors_reprompt_until_valid(self, monkeypatch, capsys):
        _script(
            monkeypatch,
            select_answers=["icon"],
            text_answers=["red, blue", "#12FF34"],
        )
        assert step_recraft_style("recraft/recraft-v4.1") == ("icon", ["#12FF34"])
        assert "Error" in capsys.readouterr().out


class TestRecraftDefaults:
    def test_vector_variants_default_to_vector_style(self):
        from imgprompt.presets import recraft_default_style

        assert recraft_default_style("recraft/recraft-v4.1-vector") == (
            "vector_illustration"
        )
        assert recraft_default_style("recraft/recraft-v4.1-pro-vector") == (
            "vector_illustration"
        )

    def test_raster_variants_default_to_realistic(self):
        from imgprompt.presets import recraft_default_style

        for m in (
            "recraft/recraft-v4.1",
            "recraft/recraft-v4.1-pro",
            "recraft/recraft-v4.1-utility",
            "recraft/recraft-v4.1-utility-pro",
        ):
            assert recraft_default_style(m) == "realistic_image"

    def test_default_is_first_choice_in_picker(self, monkeypatch):
        """The documented default must be the highlighted (first non-back)
        entry so plain Enter picks it."""
        captured = {}

        def fake_select(message, choices=None, **kwargs):
            captured["choices"] = choices
            return _FakeQuestion(BACK_OPTION)

        monkeypatch.setattr(imgedit.questionary, "select", fake_select)
        step_recraft_style("recraft/recraft-v4.1-vector")
        first = captured["choices"][1]  # index 0 is Go back
        assert first.value == "vector_illustration"


# --------------------------------------------------------------------------
# --replay --model/--provider override (issue #12)
# --------------------------------------------------------------------------


class _DummyProvider:
    """Stand-in provider recording run() calls; no network, no disk."""

    runs: list = []

    @classmethod
    def provider_name(cls):
        return "Dummy"

    @classmethod
    def supported_models(cls):
        return ["model-a", "model-b"]

    def run(self, request):
        _DummyProvider.runs.append(request)


@pytest.fixture
def replay_env(monkeypatch):
    """Wire run_replay to a dummy provider and an in-memory history."""
    _DummyProvider.runs = []
    saved = []
    req = GenerationRequest(
        prompt="write BORN 2 SHIT in fancy text",
        model="model-a",
        aspect_ratio="9:16",
        res_key="768x1344",
        quality_key="2K",
        images=["photo.jpg"],
        n=1,
        extras={"seed": 7},
    )
    monkeypatch.setattr(imgedit, "PROVIDER_MAP", {"Dummy": _DummyProvider})
    monkeypatch.setattr(imgedit, "load_last_generation", lambda: ("Dummy", req))
    monkeypatch.setattr(
        imgedit, "save_last_generation", lambda p, r: saved.append((p, r))
    )
    # Default: decline the "Edit the prompt?" confirm so the replay runs
    # verbatim. The edit path is exercised separately in TestMaybeEditPrompt.
    monkeypatch.setattr(
        imgedit.questionary, "confirm", lambda *a, **k: _FakeQuestion(False)
    )
    return req, saved


class TestMaybeEditPrompt:
    """The optional one-off prompt tweak offered before a replay re-runs."""

    def _req(self):
        return GenerationRequest(
            prompt="a cat, sitting",
            model="model-a",
            aspect_ratio="1:1",
            res_key="1024x1024",
            quality_key="2K",
        )

    def test_declining_leaves_prompt_untouched(self, monkeypatch):
        monkeypatch.setattr(
            imgedit.questionary, "confirm", lambda *a, **k: _FakeQuestion(False)
        )
        req = self._req()
        imgedit.maybe_edit_prompt(req)
        assert req.prompt == "a cat, sitting"

    def test_accepting_applies_the_edited_prompt(self, monkeypatch):
        monkeypatch.setattr(
            imgedit.questionary, "confirm", lambda *a, **k: _FakeQuestion(True)
        )
        monkeypatch.setattr(
            imgedit, "multiline_prompt", lambda *a, **k: "a cat, sitting, smiling"
        )
        req = self._req()
        imgedit.maybe_edit_prompt(req)
        assert req.prompt == "a cat, sitting, smiling"

    def test_prefills_the_saved_prompt_as_default(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            imgedit.questionary, "confirm", lambda *a, **k: _FakeQuestion(True)
        )

        def fake_multiline(message, default=""):
            seen["default"] = default
            return default

        monkeypatch.setattr(imgedit, "multiline_prompt", fake_multiline)
        req = self._req()
        imgedit.maybe_edit_prompt(req)
        assert seen["default"] == "a cat, sitting"

    def test_empty_edit_keeps_original_prompt(self, monkeypatch):
        monkeypatch.setattr(
            imgedit.questionary, "confirm", lambda *a, **k: _FakeQuestion(True)
        )
        monkeypatch.setattr(imgedit, "multiline_prompt", lambda *a, **k: "   ")
        req = self._req()
        imgedit.maybe_edit_prompt(req)
        assert req.prompt == "a cat, sitting"

    def test_cancelled_edit_keeps_original_prompt(self, monkeypatch):
        monkeypatch.setattr(
            imgedit.questionary, "confirm", lambda *a, **k: _FakeQuestion(True)
        )
        monkeypatch.setattr(imgedit, "multiline_prompt", lambda *a, **k: None)
        req = self._req()
        imgedit.maybe_edit_prompt(req)
        assert req.prompt == "a cat, sitting"


class TestReplayOverride:
    def test_bare_replay_keeps_saved_model(self, replay_env):
        req, saved = replay_env
        run_replay(None)
        assert len(_DummyProvider.runs) == 1
        assert _DummyProvider.runs[0].model == "model-a"

    def test_model_override_rebinds_before_run(self, replay_env):
        req, saved = replay_env
        run_replay(None, model_override="model-b")
        assert _DummyProvider.runs[0].model == "model-b"
        # Everything else is preserved verbatim.
        r = _DummyProvider.runs[0]
        assert r.prompt == req.prompt
        assert r.aspect_ratio == "9:16"
        assert r.quality_key == "2K"
        assert r.images == ["photo.jpg"]
        assert r.extras == {"seed": 7}

    def test_provider_override_is_case_insensitive(self, replay_env):
        run_replay(None, model_override="model-b", provider_override="dummy")
        assert len(_DummyProvider.runs) == 1

    def test_unknown_model_fails_fast_without_running(self, replay_env, capsys):
        with pytest.raises(SystemExit):
            run_replay(None, model_override="nope/never-existed")
        assert _DummyProvider.runs == []
        out = capsys.readouterr().out
        assert "not supported" in out
        assert "model-a" in out  # the supported list is shown

    def test_unknown_provider_fails_fast(self, replay_env, capsys):
        with pytest.raises(SystemExit):
            run_replay(None, provider_override="closedai")
        assert _DummyProvider.runs == []
        assert "unknown provider" in capsys.readouterr().out

    def test_override_is_persisted_for_next_replay(self, replay_env):
        """A further bare --replay must repeat THIS attempt (A→B→B chains)."""
        req, saved = replay_env
        run_replay(None, model_override="model-b")
        assert len(saved) == 1
        saved_provider, saved_req = saved[0]
        assert saved_provider == "Dummy"
        assert saved_req.model == "model-b"

    def test_cli_rejects_model_without_replay(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["imgedit.py", "--model", "model-b"])
        with pytest.raises(SystemExit) as exc:
            imgedit.main()
        assert exc.value.code == 2  # argparse usage error
        assert "--replay" in capsys.readouterr().err
