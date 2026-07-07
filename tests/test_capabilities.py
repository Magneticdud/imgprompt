"""Tests for the OpenRouter capability catalog (issue #11).

requests.get is stubbed at the module boundary; no test touches the
network. reset_catalog() keeps the per-process cache from leaking between
tests.
"""

from unittest.mock import MagicMock, patch

import pytest

import imgprompt.providers.capabilities as capabilities
from imgprompt.providers.capabilities import (
    ModelCapabilities,
    _parse_item,
    get_capabilities,
    reset_catalog,
)

SEEDREAM_ITEM = {
    "id": "bytedance-seed/seedream-4.5",
    "supported_parameters": {
        "resolution": {"type": "enum", "values": ["1K", "2K", "4K"]},
        "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9", "9:16"]},
        "n": {"type": "range", "min": 1, "max": 10},
        "input_references": {"type": "range", "min": 0, "max": 14},
        "seed": {"type": "boolean"},
    },
    "supports_streaming": False,
}


@pytest.fixture(autouse=True)
def clean_catalog():
    reset_catalog()
    yield
    reset_catalog()


def _stub_get(mock_get, data=None, error=None):
    if error is not None:
        mock_get.side_effect = error
        return
    resp = MagicMock()
    resp.json.return_value = {"data": data or []}
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp


class TestParseItem:
    def test_full_descriptor(self):
        caps = _parse_item(SEEDREAM_ITEM)
        assert caps.model == "bytedance-seed/seedream-4.5"
        assert caps.aspect_ratios == ("1:1", "16:9", "9:16")
        assert caps.resolutions == ("1K", "2K", "4K")
        assert caps.n_min == 1 and caps.n_max == 10
        assert caps.input_refs_max == 14
        assert caps.extra_flags == {"seed"}

    def test_missing_parameters_default_empty(self):
        caps = _parse_item({"id": "recraft/recraft-v4.1"})
        assert caps.aspect_ratios == ()
        assert caps.resolutions == ()
        assert caps.n_max is None
        assert caps.input_refs_max is None

    def test_item_without_id_is_dropped(self):
        assert _parse_item({"supported_parameters": {}}) is None


class TestCatalogFetch:
    def test_hit_returns_parsed_descriptor(self):
        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_get(mock_get, data=[SEEDREAM_ITEM])
            caps = get_capabilities("bytedance-seed/seedream-4.5")
        assert isinstance(caps, ModelCapabilities)
        assert caps.n_max == 10

    def test_unknown_model_returns_none(self):
        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_get(mock_get, data=[SEEDREAM_ITEM])
            assert get_capabilities("nope/never") is None

    def test_single_fetch_per_session(self):
        """The catalog is fetched once and reused across lookups."""
        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_get(mock_get, data=[SEEDREAM_ITEM])
            get_capabilities("bytedance-seed/seedream-4.5")
            get_capabilities("bytedance-seed/seedream-4.5")
            get_capabilities("other/model")
        assert mock_get.call_count == 1

    def test_network_failure_degrades_to_none_with_note(self, capsys):
        import requests as real_requests

        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_get(mock_get, error=real_requests.ConnectionError("down"))
            assert get_capabilities("bytedance-seed/seedream-4.5") is None
            # Failure is cached: one attempt, one note for the whole session.
            assert get_capabilities("bytedance-seed/seedream-4.5") is None
        assert mock_get.call_count == 1
        assert capsys.readouterr().out.count("unavailable") == 1

    def test_malformed_body_degrades_to_none(self):
        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            resp = MagicMock()
            resp.json.side_effect = ValueError("not json")
            resp.raise_for_status.return_value = None
            mock_get.return_value = resp
            assert get_capabilities("bytedance-seed/seedream-4.5") is None

    def test_api_key_forwarded_when_present(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_get(mock_get, data=[])
            get_capabilities("x")
        headers = mock_get.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-test"

    def test_timeout_is_bounded(self):
        """Discovery must never hang the wizard: the 2s budget is pinned."""
        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_get(mock_get, data=[])
            get_capabilities("x")
        assert mock_get.call_args.kwargs["timeout"] == (2.0, 2.0)
