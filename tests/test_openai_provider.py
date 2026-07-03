"""Tests for the OpenAI provider's usage/cost reporting (issue #4).

The network-facing piece of the OpenAI provider is the SDK client (`OpenAI`),
so tests stub `_run_single` and `_run_batch`'s API call site by feeding
hand-crafted response objects directly into `_report_usage`. This keeps the
parsing/fallback surface area honest without needing either a network
fixture or an OpenAI SDK roundtrip.
"""

from unittest.mock import MagicMock

import pytest

from imgprompt.providers.openai_provider import OpenAIProvider
from imgprompt.presets import GPT_IMAGE_2_PRICE_PER_MTOK


def _resp_with_usage(usage) -> MagicMock:
    """Build a mock OpenAI ImagesResponse carrying the given `usage` value.

    Passing `None` for usage simulates the legacy non-gpt-image-2 path
    (dall-e-style) where the API returns no token accounting.
    """
    resp = MagicMock()
    resp.usage = usage
    # ``response.data`` carries the actual image. _report_usage short-
    # circuits on an empty data list, so we hand it a single fake item —
    # the test doesn't care what's in it, only that the field is truthy.
    fake_item = MagicMock()
    fake_item.url = None
    fake_item.b64_json = ""
    resp.data = [fake_item]
    return resp


@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider()


class TestReportUsage:
    """The core of issue #4: extract real tokens and a USD cost from the
    `usage` block, and tolerate every shape the SDK has shown so far."""

    def test_total_tokens_used_directly(self, provider, capsys):
        """The canonical happy path: total_tokens present, no need to fall
        back to summing output+input."""
        usage = MagicMock()
        usage.total_tokens = 1_000_000  # exactly one million
        usage.output_tokens = 1_000_000
        usage.input_tokens = 0
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)

        assert tokens == 1_000_000
        assert cost == pytest.approx(GPT_IMAGE_2_PRICE_PER_MTOK)  # = $30
        captured = capsys.readouterr()
        assert "actual usage" in captured.out
        assert "1,000,000 tokens" in captured.out
        assert "$30.0000" in captured.out

    def test_total_tokens_skipped_when_output_and_input_present(
        self, provider, capsys
    ):
        """When `total_tokens` is missing the fallback sums output+input.
        This mirrors how the OpenAI SDK occasionally omits the aggregate
        field on streaming/error paths."""

        class _Usage:
            output_tokens = 750
            input_tokens = 250
            # No total_tokens set: getattr -> AttributeError -> None.

        resp = _resp_with_usage(_Usage())

        tokens, cost = provider._report_usage(resp)
        # Sum used, not the sum's component parts, so the per-token cost
        # line in the wizard still lines up with the API's actual charge.
        assert tokens == 1000
        assert cost == pytest.approx(
            1000 * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000
        )

    def test_dict_shaped_usage_is_treated_like_attribute_object(
        self, provider, capsys
    ):
        """Some OpenAI SDK versions and mock servers return a plain dict
        in `response.usage`. The helper has to read both shapes the same
        way or we'd silently lose data on upgrades."""
        usage = {"total_tokens": 500}
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)
        assert tokens == 500
        assert cost == pytest.approx(500 * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000)

    def test_dict_shape_with_only_output_tokens(self, provider, capsys):
        """Dict variant of the output+input fallback."""
        usage = {"output_tokens": 300, "input_tokens": 200}
        resp = _resp_with_usage(usage)

        tokens, _cost = provider._report_usage(resp)
        assert tokens == 500

    def test_missing_usage_attribute_returns_none_silently(
        self, provider, capsys
    ):
        """Legacy / non-gpt-image-2 image endpoints never set `usage`.
        The function must return None and *not* print anything — the user
        keeps relying on the wizard's pre-call estimate, which was already
        in the terminal. Crashing here would be the worst kind of surprise."""

        class _NoUsage:
            # Deliberately no `usage` attribute.
            data = []

        tokens, cost = provider._report_usage(_NoUsage())
        assert tokens is None
        assert cost is None
        captured = capsys.readouterr()
        assert "actual usage" not in captured.out

    def test_usage_explicit_none_returns_none_silently(self, provider, capsys):
        """`response.usage is None` (set but empty) must behave the same as
        a missing attribute — both are the documented fallback."""
        resp = _resp_with_usage(None)
        tokens, cost = provider._report_usage(resp)
        assert tokens is None
        assert cost is None
        captured = capsys.readouterr()
        assert "actual usage" not in captured.out

    def test_zero_tokens_everywhere_returns_none(self, provider, capsys):
        """Defensive: when the API sends back a usage block but every count
        field is 0, treating that as a real "no charge" line would be
        misleading. We treat empty-zero usage the same as missing usage."""
        usage = MagicMock()
        usage.total_tokens = None
        usage.output_tokens = 0
        usage.input_tokens = 0
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)
        assert tokens is None
        assert cost is None
        captured = capsys.readouterr()
        assert "actual usage" not in captured.out

    def test_non_numeric_tokens_returns_none(self, provider, capsys):
        """A malformed payload (e.g. a string instead of an int) must not
        crash — we treat unparseable token counts as missing data and let
        the wizard's pre-call estimate cover it."""
        usage = MagicMock()
        usage.total_tokens = "not-a-number"
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)
        assert tokens is None
        assert cost is None
        captured = capsys.readouterr()
        assert "actual usage" not in captured.out

    def test_returned_cost_matches_simple_token_math(self, provider):
        """Pin the math: cost is tokens * $30 / 1_000_000. A regression
        in the price-per-MTok constant or the division would diverge from
        this closed-form expectation immediately."""
        usage = MagicMock()
        usage.total_tokens = 123_456
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)
        assert tokens == 123_456
        assert cost == pytest.approx(123_456 * 30.0 / 1_000_000)
