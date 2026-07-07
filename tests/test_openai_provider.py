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
from imgprompt.presets import (
    GPT_IMAGE_2_INPUT_PRICE_PER_MTOK,
    GPT_IMAGE_2_PRICE_PER_MTOK,
)


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

    def test_split_pricing_with_input_only(self, provider, capsys):
        """gpt-image-2 bills input at $8/MTok and output at $30/MTok, so
        a response where all tokens are input should charge at the input
        rate, not the (much higher) output rate. Regression test for the
        flat-$30 math that previously masked this asymmetry."""
        usage = MagicMock()
        usage.total_tokens = 1_000_000
        usage.input_tokens = 1_000_000
        usage.output_tokens = 0
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)

        assert tokens == 1_000_000
        assert cost == pytest.approx(GPT_IMAGE_2_INPUT_PRICE_PER_MTOK)  # = $8
        captured = capsys.readouterr()
        assert "actual usage" in captured.out
        assert "in=1,000,000" in captured.out
        assert "$8.0000" in captured.out

    def test_split_pricing_with_output_only(self, provider, capsys):
        """The complementary case: all-output usage should charge at the
        output rate ($30/MTok), not the input rate ($8/MTok)."""
        usage = MagicMock()
        usage.total_tokens = 1_000_000
        usage.input_tokens = 0
        usage.output_tokens = 1_000_000
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)

        assert tokens == 1_000_000
        assert cost == pytest.approx(GPT_IMAGE_2_PRICE_PER_MTOK)  # = $30
        captured = capsys.readouterr()
        assert "$30.0000" in captured.out

    def test_split_pricing_with_mixed_input_and_output(self, provider, capsys):
        """Mixed input+output tokens charge at their respective rates. This
        is the realistic edit case: a 250-token input image + a 750-token
        output image; cost = 250*$8 + 750*$30 = $24,500 / 1M = $0.0245."""
        usage = MagicMock()
        usage.total_tokens = 1_000
        usage.input_tokens = 250
        usage.output_tokens = 750
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)

        assert tokens == 1_000
        assert cost == pytest.approx(
            250 * GPT_IMAGE_2_INPUT_PRICE_PER_MTOK / 1_000_000
            + 750 * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000
        )
        captured = capsys.readouterr()
        assert "in=250" in captured.out
        assert "out=750" in captured.out

    def test_total_tokens_used_when_split_missing(self, provider, capsys):
        """When only `total_tokens` is reported (no in/out split), the
        tokens are billed at the output rate — the higher tier — so we
        over-estimate rather than under-charge. gpt-image-2 always
        returns the split in practice; this branch is the defensive
        fallback for older mock servers / SDK shapes."""
        usage = MagicMock()
        usage.total_tokens = 1_000_000
        # Pop input_tokens / output_tokens so getattr / dict .get returns
        # None, not a stale MagicMock attribute.
        del usage.input_tokens
        del usage.output_tokens
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)

        assert tokens == 1_000_000
        assert cost == pytest.approx(GPT_IMAGE_2_PRICE_PER_MTOK)  # = $30
        captured = capsys.readouterr()
        assert "out=1,000,000" in captured.out

    def test_dict_shaped_usage_with_split(self, provider, capsys):
        """Some OpenAI SDK versions and mock servers return a plain dict
        in `response.usage`. The split pricing must apply just the same
        way as for SDK-object shapes."""
        usage = {
            "total_tokens": 500,
            "input_tokens": 100,
            "output_tokens": 400,
        }
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)
        assert tokens == 500
        assert cost == pytest.approx(
            100 * GPT_IMAGE_2_INPUT_PRICE_PER_MTOK / 1_000_000
            + 400 * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000
        )

    def test_dict_shape_total_only_falls_back_to_output_rate(self, provider, capsys):
        """Dict variant of the total-only fallback path."""
        usage = {"total_tokens": 300}
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)
        assert tokens == 300
        # All treated as output (higher rate), per the documented fence.
        assert cost == pytest.approx(300 * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000)

    def test_missing_usage_attribute_returns_none_silently(self, provider, capsys):
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
        usage.input_tokens = "also-not-a-number"
        usage.output_tokens = "nope"
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)
        assert tokens is None
        assert cost is None
        captured = capsys.readouterr()
        assert "actual usage" not in captured.out

    def test_returned_cost_matches_simple_token_math(self, provider):
        """Pin the math on a single-modality token count: cost is
        tokens * rate / 1_000_000 with the appropriate per-direction
        rate. A regression in either pricing constant or the division
        would diverge from this closed-form expectation immediately."""
        usage = MagicMock()
        usage.total_tokens = 123_456
        usage.input_tokens = 0
        usage.output_tokens = 123_456  # all output, so output rate
        resp = _resp_with_usage(usage)

        tokens, cost = provider._report_usage(resp)
        assert tokens == 123_456
        assert cost == pytest.approx(123_456 * 30.0 / 1_000_000)
