"""Tests for the OpenRouter provider.

All network calls are stubbed via monkeypatch on
`imgprompt.providers.openrouter_provider.requests.post`. The adapter is the
only network-facing piece in this module, so mocking at that boundary is
sufficient to assert payload structure, response parsing and fan-out semantics
without touching disk or hitting OpenRouter.
"""

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from imgprompt.providers.base import GenerationRequest
from imgprompt.providers.openrouter_provider import OpenRouterProvider

# A real, PIL-generated 1x1 PNG, base64-encoded at module load. A hardcoded
# base64 string risks subtle truncation during refactors; PIL generates a
# fully-valid PNG (correct zlib IDAT + CRC) so Image.open().verify() succeeds
# reliably without false positives on test infrastructure.
_buf = io.BytesIO()
Image.new("RGB", (1, 1), (10, 20, 30)).save(_buf, format="PNG")
_TINY_PNG_B64 = base64.b64encode(_buf.getvalue()).decode()


def _png_bytes() -> bytes:
    return base64.b64decode(_TINY_PNG_B64)


def _make_response(data=None, cost=None, status_error=None):
    """Build a MagicMock standing in for a `requests.Response`."""
    body = {}
    if data is not None:
        body["data"] = data
    if cost is not None:
        body["usage"] = {"prompt_tokens": 0, "completion_tokens": 1, "cost": cost}
    resp = MagicMock()
    resp.json.return_value = body
    if status_error is not None:
        resp.raise_for_status.side_effect = status_error
    else:
        resp.raise_for_status.return_value = None
    return resp


def _stub_post(mock_post, data=None, cost=None, status_error=None):
    mock_post.return_value = _make_response(
        data=data, cost=cost, status_error=status_error
    )


def _fake_image(tmp_path, name="img.png", size=(64, 64), fmt="PNG"):
    p = tmp_path / name
    Image.new("RGB", size, (10, 20, 30)).save(p, format=fmt)
    return str(p)


@pytest.fixture
def provider_with_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return OpenRouterProvider()


# --------------------------------------------------------------------------
# Payload construction
# --------------------------------------------------------------------------


class TestBuildPayload:
    def test_text_to_image_uses_aspect_ratio_and_resolution(self, provider_with_key):
        req = GenerationRequest(
            prompt="a fox",
            model="sourceful/riverflow-v2.5-pro",
            aspect_ratio="16:9",
            res_key="1344x768",
            quality_key="2K",
        )
        body = provider_with_key._build_payload(req)

        assert body["model"] == "sourceful/riverflow-v2.5-pro"
        assert body["prompt"] == "a fox"
        assert body["aspect_ratio"] == "16:9"
        assert body["resolution"] == "2K"
        assert "size" not in body  # size shorthand must NOT coexist
        assert "input_references" not in body
        # `n` is owned by _call_api (so it can honour the per-call
        # override from the input-batch fan-out).
        assert "n" not in body

    def test_custom_dims_use_size_shorthand_instead_of_ratio(self, provider_with_key):
        req = GenerationRequest(
            prompt="a fox",
            model="sourceful/riverflow-v2.5-pro",
            aspect_ratio="1:1",  # ignored when width/height set
            res_key="custom",
            quality_key="2K",
            width=2048,
            height=1024,
        )
        body = provider_with_key._build_payload(req)

        assert body["size"] == "2048x1024"
        assert "aspect_ratio" not in body
        assert "resolution" not in body

    def test_quality_key_1k_is_passed_as_resolution(self, provider_with_key):
        """Legacy code only set image_size for 2K/4K. The new endpoint
        accepts 1K too — verify we forward it."""
        req = GenerationRequest(
            prompt="x",
            model="sourceful/riverflow-v2.5-pro",
            aspect_ratio="1:1",
            res_key="1024x1024",
            quality_key="1K",
        )
        body = provider_with_key._build_payload(req)
        assert body["resolution"] == "1K"

    def test_non_tier_quality_key_omits_resolution(self, provider_with_key):
        req = GenerationRequest(
            prompt="x",
            model="sourceful/riverflow-v2.5-pro",
            aspect_ratio="1:1",
            res_key="1024x1024",
            quality_key="high",  # OpenAI-style legacy value, not a tier
        )
        body = provider_with_key._build_payload(req)
        assert "resolution" not in body
        assert body["aspect_ratio"] == "1:1"

    def test_input_references_use_documented_shape(self, provider_with_key, tmp_path):
        img_path = _fake_image(tmp_path)
        req = GenerationRequest(
            prompt="convert to watercolor",
            model="openai/gpt-5.4-image-2",
            aspect_ratio="1:1",
            res_key="1024x1024",
            quality_key="2K",
            images=[img_path],
        )
        body = provider_with_key._build_payload(req, img_paths=[img_path])

        assert "input_references" in body
        assert len(body["input_references"]) == 1
        ref = body["input_references"][0]
        assert ref["type"] == "image_url"
        assert ref["image_url"]["url"].startswith("data:image/")

    def test_extras_are_forwarded_verbatim(self, provider_with_key):
        req = GenerationRequest(
            prompt="x",
            model="bytedance-seed/seedream-4.5",
            aspect_ratio="1:1",
            res_key="1024x1024",
            quality_key="2K",
            extras={
                "output_format": "webp",
                "background": "transparent",
                "seed": 42,
                "provider": {"options": {"bytedance-seed": {"steps": 30}}},
            },
        )
        body = provider_with_key._build_payload(req)
        assert body["output_format"] == "webp"
        assert body["background"] == "transparent"
        assert body["seed"] == 42
        assert body["provider"] == {"options": {"bytedance-seed": {"steps": 30}}}

    def test_n_is_not_set_in_build_payload(self, provider_with_key):
        """`n` lives in _call_api so the per-call override from a fan-out
        is honoured without duplicating clamp logic in two places."""
        req = GenerationRequest(
            prompt="x",
            model="bytedance-seed/seedream-4.5",
            aspect_ratio="1:1",
            res_key="1024x1024",
            quality_key="2K",
            n=4,
        )
        body = provider_with_key._build_payload(req)
        assert "n" not in body

    def test_authorization_header_carries_api_key(self, provider_with_key):
        req = GenerationRequest(
            prompt="x",
            model="bytedance-seed/seedream-4.5",
            aspect_ratio="1:1",
            res_key="1024x1024",
            quality_key="1K",
        )
        body = provider_with_key._build_payload(req)
        _ = body  # just touching to prove no header needed for build
        headers = provider_with_key._headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        # App-identification headers are kept from the legacy adapter so
        # existing OpenRouter dashboards still see the same user-agent.
        assert headers["HTTP-Referer"].startswith("https://github.com/")
        assert headers["X-OpenRouter-Title"] == "IMGPrompt"


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


class TestCallApi:
    def test_single_image_returned_as_bytes(self, provider_with_key):
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}], cost=0.04)

            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="2K",
            )
            out = provider_with_key._call_api(req, n=1)

        assert len(out) == 1
        img_bytes, media_type = out[0]
        assert media_type is None  # PNG path doesn't set it
        # BytesIO wrapper is required: Image.open(bytes) treats them as a
        # filename on some inputs and chokes on embedded null bytes.
        Image.open(io.BytesIO(img_bytes)).verify()

    def test_multiple_images_returned_in_order(self, provider_with_key, capsys):
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(
                mock_post,
                data=[{"b64_json": _TINY_PNG_B64}, {"b64_json": _TINY_PNG_B64}],
                cost=0.08,
            )

            req = GenerationRequest(
                prompt="x",
                model="openai/gpt-5.4-image-2",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="2K",
                n=2,
            )
            out = provider_with_key._call_api(req, n=2)

        assert len(out) == 2
        # usage.cost once, not twice.
        captured = capsys.readouterr()
        assert captured.out.count("reported cost") == 1
        assert "$0.0800" in captured.out

    @pytest.mark.parametrize("prefix", ["data:image/png;base64,", ""])
    def test_data_url_prefix_is_stripped(self, provider_with_key, prefix):
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": prefix + _TINY_PNG_B64}])

            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
            )
            out = provider_with_key._call_api(req, n=1)
            assert len(out) == 1
            Image.open(io.BytesIO(out[0][0])).verify()

    def test_http_error_is_raised_for_caller(self, provider_with_key):
        import requests as real_requests

        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(
                mock_post,
                status_error=real_requests.HTTPError("400 Client Error"),
            )
            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
            )
            with pytest.raises(real_requests.HTTPError):
                provider_with_key._call_api(req, n=1)

    def test_cost_reporting_only_when_present(self, provider_with_key, capsys):
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}])
            # No usage.cost in body → no summary line.
            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
            )
            provider_with_key._call_api(req, n=1)

        captured = capsys.readouterr()
        assert "reported cost" not in captured.out

    def test_n_requested_clamped_to_ten_in_payload(self, provider_with_key):
        """Provider clamps request.n > 10 before posting."""
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}])
            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
                n=999,
            )
            provider_with_key._call_api(req, n=999)
        assert mock_post.call_args.kwargs["json"]["n"] == 10

    def test_n_zero_floored_to_one_in_payload(self, provider_with_key):
        """Negative or zero n gets floored to 1 before the API call."""
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}])
            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
                n=0,
            )
            provider_with_key._call_api(req, n=0)
        assert mock_post.call_args.kwargs["json"]["n"] == 1

    def test_timeout_is_passed_to_requests_post(self, provider_with_key):
        """Without a timeout, requests.post would hang indefinitely on
        stalled peers — guard against a future regression removing it."""
        from imgprompt.providers.openrouter_provider import _DEFAULT_TIMEOUT

        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}])
            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
            )
            provider_with_key._call_api(req, n=1)
        # Assert the exact value, not just presence — a future change that
        # accidentally removes the timeout (or sets it to None) must be caught.
        assert mock_post.call_args.kwargs["timeout"] == _DEFAULT_TIMEOUT

    def test_fewer_images_than_requested_logs_warning(self, provider_with_key, capsys):
        """If the server caps `n` server-side and returns fewer, we surface
        a warning so silent truncation doesn't surprise the user."""
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}])
            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="2K",
                n=4,
            )
            out = provider_with_key._call_api(req, n=4)

        assert len(out) == 1
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "4" in captured.out
        assert "1" in captured.out

    def test_svg_response_carries_media_type(self, provider_with_key):
        """The Recraft / vector endpoints set media_type=image/svg+xml on
        each item. The provider forwards it so save_one can pick the right
        extension."""
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(
                mock_post,
                data=[{"b64_json": _TINY_PNG_B64, "media_type": "image/svg+xml"}],
            )
            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
            )
            out = provider_with_key._call_api(req, n=1)

        assert out[0][1] == "image/svg+xml"

    def test_mixed_svg_and_raster_in_one_response(self, provider_with_key):
        """A single response can contain both vector + raster items (mixed
        Recraft outputs); each item's media_type must come through so _save_one
        can route to the right extension per item."""
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(
                mock_post,
                data=[
                    {"b64_json": _TINY_PNG_B64, "media_type": "image/png"},
                    {"b64_json": _TINY_PNG_B64, "media_type": "image/svg+xml"},
                    {"b64_json": _TINY_PNG_B64},  # raster default
                ],
            )
            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
            )
            out = provider_with_key._call_api(req, n=3)

        assert [m for _, m in out] == [
            "image/png",
            "image/svg+xml",
            None,  # absent → save_image_bytes handles raster default
        ]

    def test_empty_b64_in_response_skipped_with_warning(
        self, provider_with_key, capsys
    ):
        """Defensive: a response item with no decodable payload must not
        crash the whole call. Counted separately so the user sees why
        fewer images arrived."""
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(
                mock_post,
                data=[
                    {"b64_json": _TINY_PNG_B64},
                    {"b64_json": ""},  # empty → skip
                ],
            )
            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="2K",
                n=2,
            )
            out = provider_with_key._call_api(req, n=2)

        assert len(out) == 1
        captured = capsys.readouterr()
        assert "skipped" in captured.out
        assert "1" in captured.out  # the count of skipped items


# --------------------------------------------------------------------------
# Run flow (single + batch + n)
# --------------------------------------------------------------------------


class TestRunVariants:
    def test_single_input_n1_makes_one_save(
        self, provider_with_key, tmp_path, monkeypatch
    ):
        """n=1 with a single image: one API call, one file on disk."""
        saved_paths = []

        def fake_save(b, src):
            p = tmp_path / f"save_{len(saved_paths)}.png"
            p.write_bytes(b)
            saved_paths.append(str(p))
            return str(p)

        monkeypatch.setattr(
            "imgprompt.providers.openrouter_provider.save_image_bytes",
            fake_save,
        )

        img = _fake_image(tmp_path)
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}])

            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
                images=[img],
                n=1,
            )
            provider_with_key._run_variants(req)

        assert len(saved_paths) == 1

    def test_single_input_n3_returns_three_variants_one_call(
        self, provider_with_key, tmp_path, monkeypatch
    ):
        """n=3 with a single image: still one HTTP call, three files."""
        saved_paths = []

        def fake_save(b, src):
            p = tmp_path / f"save_{len(saved_paths)}.png"
            p.write_bytes(b)
            saved_paths.append(str(p))
            return str(p)

        monkeypatch.setattr(
            "imgprompt.providers.openrouter_provider.save_image_bytes",
            fake_save,
        )

        img = _fake_image(tmp_path)
        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(
                mock_post,
                data=[
                    {"b64_json": _TINY_PNG_B64},
                    {"b64_json": _TINY_PNG_B64},
                    {"b64_json": _TINY_PNG_B64},
                ],
            )

            req = GenerationRequest(
                prompt="x",
                model="openai/gpt-5.4-image-2",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="2K",
                images=[img],
                n=3,
            )
            provider_with_key._run_variants(req)

        # one HTTP call returning three images
        assert mock_post.call_count == 1
        # three separate save_image_bytes invocations (3 distinct outputs)
        assert len(saved_paths) == 3

    def test_text_to_image_n1(self, provider_with_key, tmp_path, monkeypatch):
        saved_paths = []

        def fake_save(b, src):
            p = tmp_path / f"gen_{len(saved_paths)}.png"
            p.write_bytes(b)
            saved_paths.append(str(p))
            return str(p)

        monkeypatch.setattr(
            "imgprompt.providers.openrouter_provider.save_image_bytes",
            fake_save,
        )

        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}])

            req = GenerationRequest(
                prompt="a fox",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
                n=1,
            )
            provider_with_key._run_variants(req)

        assert len(saved_paths) == 1
        # Payload must NOT carry input_references for text-to-image.
        kwargs = mock_post.call_args.kwargs
        assert "input_references" not in kwargs["json"]


class TestRunInputBatch:
    def test_multi_input_n1_calls_once_per_input(
        self, provider_with_key, tmp_path, monkeypatch
    ):
        saved_paths = []

        def fake_save(b, src):
            p = tmp_path / f"save_{len(saved_paths)}.png"
            p.write_bytes(b)
            saved_paths.append(str(p))
            return str(p)

        monkeypatch.setattr(
            "imgprompt.providers.openrouter_provider.save_image_bytes",
            fake_save,
        )

        img_a = _fake_image(tmp_path, name="a.png")
        img_b = _fake_image(tmp_path, name="b.png")

        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}])

            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
                images=[img_a, img_b],
                n=1,
            )
            provider_with_key._run_input_batch(req)

        # K inputs × 1 variant = 2 saves.
        assert mock_post.call_count == 2
        assert len(saved_paths) == 2

    def test_multi_input_n2_calls_per_input_with_n2_payload(
        self, provider_with_key, tmp_path, monkeypatch
    ):
        saved_paths = []

        def fake_save(b, src):
            p = tmp_path / f"save_{len(saved_paths)}.png"
            p.write_bytes(b)
            saved_paths.append(str(p))
            return str(p)

        monkeypatch.setattr(
            "imgprompt.providers.openrouter_provider.save_image_bytes",
            fake_save,
        )

        img_a = _fake_image(tmp_path, name="a.png")
        img_b = _fake_image(tmp_path, name="b.png")

        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(
                mock_post,
                data=[
                    {"b64_json": _TINY_PNG_B64},
                    {"b64_json": _TINY_PNG_B64},
                ],
            )

            req = GenerationRequest(
                prompt="x",
                model="openai/gpt-5.4-image-2",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="2K",
                images=[img_a, img_b],
                n=2,
            )
            provider_with_key._run_input_batch(req)

        # Two inputs → two HTTP calls, each carrying n=2 in the body.
        assert mock_post.call_count == 2
        assert len(saved_paths) == 4  # 2 inputs × 2 variants
        for call in mock_post.call_args_list:
            assert call.kwargs["json"]["n"] == 2


# --------------------------------------------------------------------------
# seedream-4.5 upstream pixel floor (issue #10): the Seed provider 400s any
# output below 3,686,400 px, so the provider resolves an explicit `size`
# that clears the floor instead of the aspect_ratio+resolution shorthand.
# --------------------------------------------------------------------------


class TestSeedreamPixelFloor:
    FLOOR = 3_686_400

    def _payload(self, provider, ratio, tier, **kw):
        req = GenerationRequest(
            prompt="x",
            model="bytedance-seed/seedream-4.5",
            aspect_ratio=ratio,
            res_key="768x1344",
            quality_key=tier,
            **kw,
        )
        return provider._build_payload(req)

    @pytest.mark.parametrize("ratio", ["9:16", "16:9", "2:3", "3:2", "21:9", "1:1"])
    @pytest.mark.parametrize("tier", ["1K", "2K", "4K"])
    def test_explicit_size_clears_floor(self, provider_with_key, ratio, tier):
        body = self._payload(provider_with_key, ratio, tier)
        assert "aspect_ratio" not in body
        assert "resolution" not in body
        w, h = map(int, body["size"].split("x"))
        assert w * h >= self.FLOOR
        assert w % 16 == 0 and h % 16 == 0

    def test_9_16_1k_lands_exactly_on_floor(self, provider_with_key, capsys):
        """1K is below the floor for every ratio; 9:16 resolves to the
        canonical 1440x2560 = 3,686,400 px minimum, with a visible note."""
        body = self._payload(provider_with_key, "9:16", "1K")
        assert body["size"] == "1440x2560"
        assert "3,686,400" in capsys.readouterr().out

    def test_2k_preserves_requested_ratio(self, provider_with_key):
        body = self._payload(provider_with_key, "9:16", "2K")
        w, h = map(int, body["size"].split("x"))
        # 2K targets ~4.19MP which already clears the floor; shape must
        # still be ~9:16 (16px rounding allows small drift).
        assert abs(w / h - 9 / 16) < 0.02
        assert w * h >= 2048 * 2048

    def test_no_note_when_tier_already_clears_floor(self, provider_with_key, capsys):
        self._payload(provider_with_key, "1:1", "4K")
        assert "minimum" not in capsys.readouterr().out

    def test_custom_dims_below_floor_are_raised(self, provider_with_key, capsys):
        body = self._payload(provider_with_key, "1:1", "2K", width=768, height=1344)
        w, h = map(int, body["size"].split("x"))
        assert w * h >= self.FLOOR
        assert w % 16 == 0 and h % 16 == 0
        assert abs(w / h - 768 / 1344) < 0.02
        assert "minimum" in capsys.readouterr().out

    def test_unknown_ratio_falls_back_to_shorthand(self, provider_with_key):
        """A ratio we can't resolve numerically falls back to the plain
        aspect_ratio+resolution fields rather than guessing a size."""
        body = self._payload(provider_with_key, "7:5", "2K")
        assert body["aspect_ratio"] == "7:5"
        assert body["resolution"] == "2K"
        assert "size" not in body

    def test_models_without_floor_are_untouched(self, provider_with_key):
        req = GenerationRequest(
            prompt="x",
            model="black-forest-labs/flux.2-pro",
            aspect_ratio="9:16",
            res_key="768x1344",
            quality_key="2K",
        )
        body = provider_with_key._build_payload(req)
        assert body["aspect_ratio"] == "9:16"
        assert "size" not in body

    def test_wizard_labels_flag_floored_tiers(self, provider_with_key):
        choices, _ = provider_with_key.get_quality_choices(
            "bytedance-seed/seedream-4.5", "768x1344", None, None, None
        )
        one_k = next(c for c in choices if c.startswith("1K"))
        two_k = next(c for c in choices if c.startswith("2K"))
        assert "minimum" in one_k  # 1K (~1MP) sits below the 3.69MP floor
        assert "minimum" not in two_k  # 2K (~4.19MP) clears it


# --------------------------------------------------------------------------
# Dual mode (issue #2): two input images must reach the API in ONE combined
# call, never fan out to one call per image like batch mode does.
# --------------------------------------------------------------------------


class TestDualDispatch:
    def test_is_dual_two_images_is_not_batch(self):
        req = GenerationRequest(
            prompt="combine IMG_1 and IMG_2",
            model="bytedance-seed/seedream-4.5",
            aspect_ratio="1:1",
            res_key="1024x1024",
            quality_key="1K",
            images=["a.png", "b.png"],
            is_dual=True,
        )
        assert req.is_batch is False

    def test_two_images_without_dual_flag_stays_batch(self):
        req = GenerationRequest(
            prompt="x",
            model="bytedance-seed/seedream-4.5",
            aspect_ratio="1:1",
            res_key="1024x1024",
            quality_key="1K",
            images=["a.png", "b.png"],
        )
        assert req.is_batch is True

    def test_dual_run_makes_single_call_with_both_references(
        self, provider_with_key, tmp_path, monkeypatch
    ):
        saved_paths = []

        def fake_save(b, src):
            p = tmp_path / f"save_{len(saved_paths)}.png"
            p.write_bytes(b)
            saved_paths.append(str(p))
            return str(p)

        monkeypatch.setattr(
            "imgprompt.providers.openrouter_provider.save_image_bytes",
            fake_save,
        )

        img_a = _fake_image(tmp_path, name="a.png")
        img_b = _fake_image(tmp_path, name="b.png")

        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}])

            req = GenerationRequest(
                prompt="use the composition of IMG_1 and the style of IMG_2",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
                images=[img_a, img_b],
                is_dual=True,
            )
            provider_with_key.run(req)

        # The whole point of dual mode: ONE HTTP call carrying BOTH images.
        assert mock_post.call_count == 1
        refs = mock_post.call_args.kwargs["json"]["input_references"]
        assert len(refs) == 2
        assert len(saved_paths) == 1

    def test_batch_run_still_fans_out(self, provider_with_key, tmp_path, monkeypatch):
        """Regression guard: the dual fix must not collapse real batch mode
        (is_dual=False) into a single combined call."""
        monkeypatch.setattr(
            "imgprompt.providers.openrouter_provider.save_image_bytes",
            lambda b, src: str(tmp_path / "out.png"),
        )
        img_a = _fake_image(tmp_path, name="a.png")
        img_b = _fake_image(tmp_path, name="b.png")

        with patch(
            "imgprompt.providers.openrouter_provider.requests.post"
        ) as mock_post:
            _stub_post(mock_post, data=[{"b64_json": _TINY_PNG_B64}])

            req = GenerationRequest(
                prompt="x",
                model="bytedance-seed/seedream-4.5",
                aspect_ratio="1:1",
                res_key="1024x1024",
                quality_key="1K",
                images=[img_a, img_b],
            )
            provider_with_key.run(req)

        assert mock_post.call_count == 2
        for call in mock_post.call_args_list:
            assert len(call.kwargs["json"]["input_references"]) == 1


# --------------------------------------------------------------------------
# Gemini 3.1 Flash Lite image (the new budget model, $0.034/1K, 14 ratios).
# Added when Nano Banana 2 Lite shipped (June 2026); gemini-2.5-flash-image
# was retired at the same time and the tests below also pin its removal so
# a future re-introduction is caught.
# --------------------------------------------------------------------------


class TestGeminiFlashLite:
    """Nano Banana 2 Lite: 1K only, exposes all 14 documented Gemini 3.x
    aspect ratios. Also pins that the retired 2.5-flash-image entry is gone.
    """

    def test_lite_is_in_supported_models(self):
        assert (
            "google/gemini-3.1-flash-lite-image"
            in OpenRouterProvider.supported_models()
        )

    def test_legacy_2_5_flash_image_is_removed(self):
        # gemini-2.5-flash-image shutdown is 2 Oct 2026; the model is gone
        # from `supported_models()` so the wizard never surfaces it. If a
        # re-introduction is attempted, this test fails loudly.
        assert (
            "google/gemini-2.5-flash-image"
            not in OpenRouterProvider.supported_models()
        )

    def test_lite_quality_choices_are_1k_only(self, provider_with_key):
        choices, default = provider_with_key.get_quality_choices(
            "google/gemini-3.1-flash-lite-image",
            "1024x1024",
            None,
            None,
            None,
        )
        # Only 1K — Lite does not accept 2K or 4K.
        assert len(choices) == 1
        assert choices[0].startswith("1K ")
        assert default == choices[0]

    def test_lite_resolution_choices_exposes_all_14(self, provider_with_key):
        from imgprompt.presets import OPENROUTER_RESOLUTIONS

        choices, default = provider_with_key.get_resolution_choices(
            "google/gemini-3.1-flash-lite-image", None
        )
        # All 14 documented for Gemini 3.x image family (incl. 21:9 and the
        # four extreme ratios 1:4, 4:1, 1:8, 8:1).
        assert choices == list(OPENROUTER_RESOLUTIONS.keys())
        assert len(choices) == 14
        assert "1:4" in choices and "1:8" in choices
        assert default == "1:1"

    def test_lite_resolution_choices_honours_image_default(self, provider_with_key, tmp_path):
        # An input image's aspect ratio should still bias the default even
        # when the full 14-ratio table is exposed.
        p = tmp_path / "wide.png"
        Image.new("RGB", (3200, 900), (10, 20, 30)).save(p)
        choices, default = provider_with_key.get_resolution_choices(
            "google/gemini-3.1-flash-lite-image", str(p)
        )
        # 3200x900 ≈ 3.56:1. Across the 14-ratio table, 4:1 (4.0) is the
        # closest, beating 21:9 (≈2.33) by ≈1.22. Lock down the exact answer
        # here so a regression in get_closest_aspect_ratio is caught against
        # Lite's bigger ratio set — the broader table only widens what the
        # closest-ratio helper has to discriminate between.
        assert default == "4:1", (
            f"expected 3200x900 (≈3.56:1) to map to '4:1', got {default!r}"
        )

    def test_lite_payload_passes_resolution_1k(self, provider_with_key):
        req = GenerationRequest(
            prompt="x",
            model="google/gemini-3.1-flash-lite-image",
            aspect_ratio="1:1",
            res_key="1024x1024",
            quality_key="1K",
        )
        body = provider_with_key._build_payload(req)
        assert body["resolution"] == "1K"
        assert body["aspect_ratio"] == "1:1"
        # `size` shorthand is only triggered when width/height are set
        # explicitly; under the default flow it stays out, which is what
        # Lite's API expects.
        assert "size" not in body

    @pytest.mark.parametrize(
        "ratio",
        ["1:1", "2:3", "3:2", "4:5", "5:4", "21:9", "1:4", "4:1", "1:8", "8:1"],
    )
    def test_lite_supports_extreme_ratios(self, provider_with_key, ratio):
        """Lite's 14 ratios span 1:8 to 8:1. resolve_resolution must not
        regress any of them back to a default; the lookup is exact-match."""

        from imgprompt.presets import OPENROUTER_RESOLUTIONS

        out = provider_with_key.resolve_resolution(
            "google/gemini-3.1-flash-lite-image", ratio
        )
        assert out == (OPENROUTER_RESOLUTIONS[ratio], None, None)


# --------------------------------------------------------------------------
# Gemini 3.1 Flash Image (non-Lite). Google's model page documents the same
# 14-ratio set as Lite (Nano Banana 2), so we expose it on OpenRouter.
# Quality choices remain 1K/2K/4K (no 0.5K — left for a future change).
# 3-Pro Image stays on the conservative 10+21:9 list pending per-model
# Google confirmation or a clean upstream test through OpenRouter.
# --------------------------------------------------------------------------


class TestGeminiFlashNonLite:
    """Gemini 3.1 Flash Image (non-Lite): same 14-ratio surface as Lite.
    Added when the OpenRouter gate was extended from Lite-only to Lite +
    non-Lite Flash; mirrors the Lite tests so a future regression that
    reverts the gate is caught here too.
    """

    def test_flash_in_supported_models(self):
        assert (
            "google/gemini-3.1-flash-image"
            in OpenRouterProvider.supported_models()
        )

    def test_flash_resolution_choices_exposes_all_14(self, provider_with_key):
        from imgprompt.presets import OPENROUTER_RESOLUTIONS

        choices, default = provider_with_key.get_resolution_choices(
            "google/gemini-3.1-flash-image", None
        )
        # Same 14-entry set Lite exposes (incl. 21:9 and the four extreme
        # ratios 1:4, 4:1, 1:8, 8:1). Pinned so a regression to the
        # Lite-only gate is caught here too.
        assert choices == list(OPENROUTER_RESOLUTIONS.keys())
        assert len(choices) == 14
        assert "1:4" in choices and "1:8" in choices
        assert "4:1" in choices and "8:1" in choices
        assert default == "1:1"

    def test_flash_resolution_choices_honours_image_default(
        self, provider_with_key, tmp_path
    ):
        # Mirror of Lite test: 3200x900 on the full 14-ratio table should
        # map to 4:1 (not 21:9) — exercises get_closest_aspect_ratio's
        # discrimination on this model.
        p = tmp_path / "wide.png"
        Image.new("RGB", (3200, 900), (10, 20, 30)).save(p)
        choices, default = provider_with_key.get_resolution_choices(
            "google/gemini-3.1-flash-image", str(p)
        )
        assert default == "4:1", (
            f"expected 3200x900 (≈3.56:1) to map to '4:1', got {default!r}"
        )

    @pytest.mark.parametrize(
        "ratio",
        ["1:1", "2:3", "3:2", "4:5", "5:4", "21:9", "1:4", "4:1", "1:8", "8:1"],
    )
    def test_flash_supports_extreme_ratios(self, provider_with_key, ratio):
        from imgprompt.presets import OPENROUTER_RESOLUTIONS

        out = provider_with_key.resolve_resolution(
            "google/gemini-3.1-flash-image", ratio
        )
        assert out == (OPENROUTER_RESOLUTIONS[ratio], None, None)

    def test_flash_quality_choices_include_1k_2k_4k(self, provider_with_key):
        # Regression pin: extending the ratio set must not silently drop
        # any of the three resolution tiers.
        choices, default = provider_with_key.get_quality_choices(
            "google/gemini-3.1-flash-image",
            "1024x1024",
            None,
            None,
            None,
        )
        assert [c.split(" ")[0] for c in choices] == ["1K", "2K", "4K"]
        assert default.startswith("1K ")

    def test_3_pro_still_on_conservative_set(self, provider_with_key):
        # Regression pin: 3-Pro Image's per-model docs at Google do not
        # enumerate the four extreme ratios, so it stays on the standard
        # list + 21:9. If this gets flipped without per-model confirmation,
        # the user-facing wizard might submit ratios the upstream silently
        # clamps or 400s.
        # Compare against the derived expected list (not a hard-coded
        # length), so the assertion survives a future growth of
        # OPENROUTER_STANDARD_RATIOS without us touching this test.
        from imgprompt.presets import OPENROUTER_STANDARD_RATIOS

        choices, _ = provider_with_key.get_resolution_choices(
            "google/gemini-3-pro-image", None
        )
        expected = list(OPENROUTER_STANDARD_RATIOS) + ["21:9"]
        assert set(choices) == set(expected)
        for extreme in ("1:4", "4:1", "1:8", "8:1"):
            assert extreme not in choices, (
                f"3-Pro conservative set regressed: {extreme!r} sneaked in"
            )

    def test_flash_payload_passes_extreme_ratio_and_resolution(self, provider_with_key):
        # Smoke test: extreme ratios flow through _build_payload unchanged
        # (no size shorthand; resolution tier still forwarded).
        req = GenerationRequest(
            prompt="x",
            model="google/gemini-3.1-flash-image",
            aspect_ratio="4:1",
            res_key="2048x512",
            quality_key="1K",
        )
        body = provider_with_key._build_payload(req)
        assert body["aspect_ratio"] == "4:1"
        assert body["resolution"] == "1K"
        assert "size" not in body
