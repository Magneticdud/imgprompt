"""Wizard-step tests for imgedit.py (currently the Recraft style step).

questionary is monkeypatched at the imgedit module boundary: each fake
question returns the next scripted answer, so the tests drive the step
functions without a TTY.
"""

import pytest

import imgedit
from imgedit import (
    BACK_OPTION,
    RECRAFT_CUSTOM_STYLE,
    is_recraft_model,
    step_recraft_style,
)


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
