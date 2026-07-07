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
def clean_catalog(monkeypatch, tmp_path):
    """Fresh catalog per test, with the file cache redirected to tmp so no
    test reads or writes the real .cache/openrouter_models.json."""
    monkeypatch.setattr(
        capabilities, "CACHE_FILE", str(tmp_path / "openrouter_models.json")
    )
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

    def test_stringy_range_values_are_coerced(self):
        """External API data: numeric strings must not crash the min()/max()
        clamps downstream."""
        caps = _parse_item(
            {
                "id": "a/b",
                "supported_parameters": {
                    "n": {"type": "range", "min": "1", "max": "10"},
                },
            }
        )
        assert caps.n_min == 1
        assert caps.n_max == 10

    def test_garbage_range_values_degrade_to_no_bound(self):
        caps = _parse_item(
            {
                "id": "a/b",
                "supported_parameters": {
                    "n": {"type": "range", "min": "lots", "max": None},
                },
            }
        )
        assert caps.n_min == 1  # unparseable min falls back to the default
        assert caps.n_max is None

    def test_zero_n_min_is_preserved(self):
        """`0` is falsy but it is still the API's answer — no `or 1` here."""
        caps = _parse_item(
            {
                "id": "a/b",
                "supported_parameters": {
                    "n": {"type": "range", "min": 0, "max": 10},
                },
            }
        )
        assert caps.n_min == 0


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


# --------------------------------------------------------------------------
# File cache with 24h TTL (issue #3): fresh cache → no network; stale cache
# → served immediately + background refresh; cold start writes the cache.
# --------------------------------------------------------------------------


def _write_cache_file(fetched_at, data):
    import json
    import os

    os.makedirs(os.path.dirname(capabilities.CACHE_FILE), exist_ok=True)
    with open(capabilities.CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": fetched_at, "data": data}, f)


class TestFileCache:
    def test_fresh_cache_hit_skips_network(self):
        import time

        _write_cache_file(time.time() - 60, [SEEDREAM_ITEM])
        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            caps = get_capabilities("bytedance-seed/seedream-4.5")
        assert caps is not None and caps.n_max == 10
        mock_get.assert_not_called()

    def test_cold_start_fetches_and_writes_cache(self):
        import json
        import os

        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_get(mock_get, data=[SEEDREAM_ITEM])
            caps = get_capabilities("bytedance-seed/seedream-4.5")
        assert caps is not None
        assert os.path.exists(capabilities.CACHE_FILE)
        with open(capabilities.CACHE_FILE) as f:
            cached = json.load(f)
        assert cached["data"] == [SEEDREAM_ITEM]
        assert cached["fetched_at"] > 0

    def test_stale_cache_served_and_refreshed_in_background(self, monkeypatch):
        """A stale catalog is preferable to blocking the wizard: it is used
        for this session while a daemon thread rewrites the cache file."""
        import json
        import time

        stale_item = dict(SEEDREAM_ITEM)
        _write_cache_file(
            time.time() - capabilities.CACHE_TTL_SECONDS - 60, [stale_item]
        )

        # Run the "background" refresh synchronously for determinism.
        started = []

        class _SyncThread:
            def __init__(self, target=None, daemon=None):
                self._target = target
                started.append(self)

            def start(self):
                self._target()

        monkeypatch.setattr(capabilities.threading, "Thread", _SyncThread)

        fresh_item = {"id": "new/model", "supported_parameters": {}}
        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_get(mock_get, data=[fresh_item])
            caps = get_capabilities("bytedance-seed/seedream-4.5")

        # Session still answers from the stale catalog...
        assert caps is not None and caps.n_max == 10
        assert len(started) == 1
        # ...but the cache file now holds the fresh data for the next run.
        with open(capabilities.CACHE_FILE) as f:
            assert json.load(f)["data"] == [fresh_item]

    def test_no_cache_and_network_down_falls_back_hardcoded(self, capsys):
        import os
        import requests as real_requests

        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_get(mock_get, error=real_requests.ConnectionError("down"))
            assert get_capabilities("bytedance-seed/seedream-4.5") is None
        assert not os.path.exists(capabilities.CACHE_FILE)
        assert "built-in defaults" in capsys.readouterr().out

    def test_fresh_empty_catalog_is_honoured_not_refetched(self, capsys):
        """A fresh cache with data: [] is a valid upstream answer, not a
        missing cache — no re-fetch, and no 'unavailable' note (discovery
        worked; the catalog is just empty)."""
        import time

        _write_cache_file(time.time() - 60, [])
        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            assert get_capabilities("bytedance-seed/seedream-4.5") is None
        mock_get.assert_not_called()
        assert "unavailable" not in capsys.readouterr().out

    def test_corrupt_cache_file_treated_as_cold_start(self):
        import os

        os.makedirs(os.path.dirname(capabilities.CACHE_FILE), exist_ok=True)
        with open(capabilities.CACHE_FILE, "w") as f:
            f.write("{ not valid json")
        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_get(mock_get, data=[SEEDREAM_ITEM])
            caps = get_capabilities("bytedance-seed/seedream-4.5")
        assert caps is not None
        mock_get.assert_called_once()


# --------------------------------------------------------------------------
# Live pricing from /endpoints (issue #3).
# --------------------------------------------------------------------------

GROK_PRICING = [
    {"billable": "input_image", "unit": "image", "cost_usd": 0.01},
    {"billable": "output_image", "unit": "image", "cost_usd": 0.05, "variant": "1k"},
    {"billable": "output_image", "unit": "image", "cost_usd": 0.07, "variant": "2k"},
]

RECRAFT_PRICING = [
    {"billable": "output_image", "unit": "image", "cost_usd": 0.035},
]

MAI_PRICING = [
    {"billable": "input_text", "unit": "token", "cost_usd": 5e-06},
    {"billable": "output_image", "unit": "token", "cost_usd": 4.7e-05},
]


def _stub_endpoints(mock_get, pricing):
    resp = MagicMock()
    resp.json.return_value = {"data": {"endpoints": [{"pricing": pricing}]}}
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp


class TestLivePricing:
    def test_variant_priced_tiers(self):
        from imgprompt.providers.capabilities import output_image_price

        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_endpoints(mock_get, GROK_PRICING)
            assert output_image_price("x-ai/grok-imagine-image-quality", "1K") == 0.05
            assert output_image_price("x-ai/grok-imagine-image-quality", "2K") == 0.07
        # Session cache: one endpoints call for both lookups.
        assert mock_get.call_count == 1

    def test_flat_price_applies_to_any_tier(self):
        from imgprompt.providers.capabilities import output_image_price

        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_endpoints(mock_get, RECRAFT_PRICING)
            assert output_image_price("recraft/recraft-v4.1", "Standard") == 0.035

    def test_token_billed_model_returns_none(self):
        from imgprompt.providers.capabilities import output_image_price

        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            _stub_endpoints(mock_get, MAI_PRICING)
            assert output_image_price("microsoft/mai-image-2.5", "Standard") is None

    def test_network_failure_prints_note_once(self, capsys):
        import requests as real_requests

        from imgprompt.providers.capabilities import output_image_price

        with patch("imgprompt.providers.capabilities.requests.get") as mock_get:
            mock_get.side_effect = real_requests.ConnectionError("down")
            assert output_image_price("a/b", "1K") is None
            assert output_image_price("c/d", "1K") is None
        out = capsys.readouterr().out
        assert out.count("Live pricing unavailable") == 1
