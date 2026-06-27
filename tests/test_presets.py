"""Tests for the pure dimension/pricing logic in imgprompt.presets.

These are the functions behind the recent 'custom resolution' and
'physical units + DPI' work, so they're the highest-value things to lock down.
"""

import math

import pytest

from imgprompt.presets import (
    GPT_IMAGE_2_MAX_ASPECT,
    GPT_IMAGE_2_MAX_EDGE,
    GPT_IMAGE_2_MAX_PIXELS,
    GPT_IMAGE_2_MIN_PIXELS,
    GPT_IMAGE_2_PRESETS,
    auto_adjust_gpt_image2_dims,
    calc_gpt_image2_tokens,
    physical_to_pixels,
    round_to_multiple_of_16,
    validate_gpt_image2_dims,
)


class TestRoundToMultipleOf16:
    def test_exact_multiple_unchanged(self):
        assert round_to_multiple_of_16(1024) == 1024

    @pytest.mark.parametrize(
        "value,expected",
        [
            # NOTE: round() uses banker's rounding, so exact .5 cases round to
            # the nearest even multiple (8 -> 0, not 16; 1000 -> 992, not 1008).
            (7, 0),
            (8, 0),
            (24, 32),
            (1000, 992),
            (1007, 1008),
            (1020, 1024),
            (0, 0),
        ],
    )
    def test_rounding(self, value, expected):
        assert round_to_multiple_of_16(value) == expected

    def test_result_always_divisible_by_16(self):
        for v in range(0, 5000, 37):
            assert round_to_multiple_of_16(v) % 16 == 0


class TestPhysicalToPixels:
    def test_inches(self):
        # 1 inch at 300 dpi == 300 px
        assert physical_to_pixels(1, "in", 300) == 300

    def test_cm(self):
        # 2.54 cm == 1 inch -> 300 px at 300 dpi
        assert physical_to_pixels(2.54, "cm", 300) == 300

    def test_mm(self):
        # 25.4 mm == 1 inch -> 300 px at 300 dpi
        assert physical_to_pixels(25.4, "mm", 300) == 300

    def test_unknown_unit_treated_as_inches(self):
        assert physical_to_pixels(2, "furlong", 100) == 200

    def test_result_is_int(self):
        assert isinstance(physical_to_pixels(3.3, "cm", 150), int)


class TestValidateGptImage2Dims:
    def test_valid_square_passes(self):
        assert validate_gpt_image2_dims(1024, 1024) == []

    def test_not_divisible_by_16(self):
        errors = validate_gpt_image2_dims(1000, 1024)
        assert any("divisible by 16" in e for e in errors)

    def test_below_min_pixels(self):
        errors = validate_gpt_image2_dims(256, 256)
        assert any("Minimum" in e for e in errors)

    def test_above_max_pixels(self):
        errors = validate_gpt_image2_dims(3840, 3840)
        assert any("Maximum" in e and "pixels" in e for e in errors)

    def test_aspect_ratio_too_extreme(self):
        # 3520x16 is way past 3:1, divisible by 16, within pixel range on min side
        errors = validate_gpt_image2_dims(3520, 1088)
        assert any("aspect ratio" in e for e in errors)


class TestAutoAdjustGptImage2Dims:
    @pytest.mark.parametrize(
        "w,h",
        [
            (1024, 1024),
            (100, 100),       # far too small -> scaled up
            (5000, 5000),     # too big -> scaled down + clamped
            (4000, 100),      # extreme aspect -> corrected, regression case
            (100, 4000),      # extreme aspect, tall orientation
            (1, 1),
            (1920, 1088),
        ],
    )
    def test_output_is_always_valid(self, w, h):
        aw, ah = auto_adjust_gpt_image2_dims(w, h)
        assert aw % 16 == 0 and ah % 16 == 0
        assert aw <= GPT_IMAGE_2_MAX_EDGE and ah <= GPT_IMAGE_2_MAX_EDGE
        assert aw > 0 and ah > 0
        # The structural constraints (÷16, max edge, aspect) must always hold.
        aspect = max(aw, ah) / min(aw, ah)
        assert aspect <= GPT_IMAGE_2_MAX_ASPECT + 1e-9

    def test_already_valid_stays_close(self):
        aw, ah = auto_adjust_gpt_image2_dims(1024, 1024)
        assert (aw, ah) == (1024, 1024)

    def test_extreme_aspect_ratio_stays_within_max_edge(self):
        # Regression: (4000,100) used to return (5120,128), over the 3840 edge.
        aw, ah = auto_adjust_gpt_image2_dims(4000, 100)
        assert aw <= GPT_IMAGE_2_MAX_EDGE and ah <= GPT_IMAGE_2_MAX_EDGE
        assert validate_gpt_image2_dims(aw, ah) == []

    def test_long_edge_over_max_preserves_aspect(self):
        # Regression: a 330x1200mm @ 150 DPI request (1949x7087) used to clamp
        # only the long edge to 3840, collapsing the capped 3:1 ratio down to
        # ~1.6:1. It must instead scale both edges and stay at the 3:1 cap.
        aw, ah = auto_adjust_gpt_image2_dims(1949, 7087)
        assert validate_gpt_image2_dims(aw, ah) == []
        aspect = max(aw, ah) / min(aw, ah)
        assert aspect == pytest.approx(GPT_IMAGE_2_MAX_ASPECT, abs=0.05)

    def test_output_always_passes_validation_over_a_grid(self):
        # The adjusted dimensions must satisfy every gpt-image-2 constraint for
        # any plausible input, not just the hand-picked cases above.
        for w in range(1, 8001, 137):
            for h in range(1, 8001, 137):
                aw, ah = auto_adjust_gpt_image2_dims(w, h)
                errors = validate_gpt_image2_dims(aw, ah)
                assert errors == [], f"{w}x{h} -> {aw}x{ah}: {errors}"


class TestCalcGptImage2Tokens:
    def test_returns_positive_int(self):
        assert calc_gpt_image2_tokens(1024, 1024, "high") > 0

    def test_higher_quality_costs_more(self):
        low = calc_gpt_image2_tokens(1024, 1024, "low")
        med = calc_gpt_image2_tokens(1024, 1024, "medium")
        high = calc_gpt_image2_tokens(1024, 1024, "high")
        assert low < med < high

    def test_quality_is_case_insensitive(self):
        assert calc_gpt_image2_tokens(1024, 1024, "HIGH") == calc_gpt_image2_tokens(
            1024, 1024, "high"
        )

    def test_larger_image_costs_more(self):
        small = calc_gpt_image2_tokens(1024, 1024, "high")
        big = calc_gpt_image2_tokens(2048, 2048, "high")
        assert big > small

    def test_returns_ceiling(self):
        val = calc_gpt_image2_tokens(1920, 1088, "medium")
        assert val == math.ceil(val)


class TestPresetTableConsistency:
    """The shipped presets should themselves satisfy the model constraints."""

    @pytest.mark.parametrize("dims", list(GPT_IMAGE_2_PRESETS.values()))
    def test_every_preset_is_valid(self, dims):
        w, h = dims
        assert validate_gpt_image2_dims(w, h) == [], f"{w}x{h} violates constraints"
