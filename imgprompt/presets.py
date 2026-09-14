import math

BACK_OPTION = "← Go back"
CUSTOM_DIMS = "Custom dimensions"
GPT_IMAGE_2_AUTO = "Auto (model decides)"

# gpt-image-2 constraints
GPT_IMAGE_2_MIN_PIXELS = 655_360
GPT_IMAGE_2_MAX_PIXELS = 8_294_400
GPT_IMAGE_2_MAX_EDGE = 3840
GPT_IMAGE_2_MAX_ASPECT = 3
GPT_IMAGE_2_MULTIPLE = 16
# gpt-image-2 bills per modality and per direction. Output tokens (the
# generated image) are the more expensive tier; input tokens (uploaded
# image + text prompt) are roughly a quarter of the output rate. Cached
# input is a third tier priced server-side that we don't surface here
# (the API's input_tokens count already excludes the discount).
GPT_IMAGE_2_PRICE_PER_MTOK = 30.0  # output rate (image tokens generated)
GPT_IMAGE_2_INPUT_PRICE_PER_MTOK = 8.0  # input rate (image + text tokens)

# gpt-image-2 presets: (ratio, size) -> (width, height)
GPT_IMAGE_2_PRESETS = {
    ("1:1", "1K"): (1024, 1024),
    ("1:1", "2K"): (1536, 1536),
    ("1:1", "4K"): (2048, 2048),
    ("16:9", "2K"): (1920, 1088),
    ("16:9", "4K"): (2560, 1440),
    ("9:16", "2K"): (1088, 1920),
    ("9:16", "4K"): (1440, 2560),
    ("4:3", "2K"): (1600, 1200),
    ("4:3", "4K"): (2304, 1728),
    ("3:4", "2K"): (1200, 1600),
    ("3:4", "4K"): (1728, 2304),
    ("3:2", "2K"): (1728, 1152),
    ("3:2", "4K"): (2496, 1664),
    ("2:3", "2K"): (1152, 1728),
    ("2:3", "4K"): (1664, 2496),
}

# Ordered preset choices for CLI display
GPT_IMAGE_2_PRESET_CHOICES = [
    ("1:1 — 1K (1024×1024)", "1:1", "1K", 1024, 1024),
    ("1:1 — 2K (1536×1536)", "1:1", "2K", 1536, 1536),
    ("1:1 — 4K (2048×2048)", "1:1", "4K", 2048, 2048),
    ("16:9 — 2K (1920×1088)", "16:9", "2K", 1920, 1088),
    ("16:9 — 4K (2560×1440)", "16:9", "4K", 2560, 1440),
    ("9:16 — 2K (1088×1920)", "9:16", "2K", 1088, 1920),
    ("9:16 — 4K (1440×2560)", "9:16", "4K", 1440, 2560),
    ("4:3 — 2K (1600×1200)", "4:3", "2K", 1600, 1200),
    ("4:3 — 4K (2304×1728)", "4:3", "4K", 2304, 1728),
    ("3:4 — 2K (1200×1600)", "3:4", "2K", 1200, 1600),
    ("3:4 — 4K (1728×2304)", "3:4", "4K", 1728, 2304),
    ("3:2 — 2K (1728×1152)", "3:2", "2K", 1728, 1152),
    ("3:2 — 4K (2496×1664)", "3:2", "4K", 2496, 1664),
    ("2:3 — 2K (1152×1728)", "2:3", "2K", 1152, 1728),
    ("2:3 — 4K (1664×2496)", "2:3", "4K", 1664, 2496),
]


# Quality -> grid-cells-on-the-long-side, per model family. The families
# share the grid formula below but NOT this mapping: GPT Image 2.5 shifted
# the whole ladder down a rung and inserted two new steps, so the same
# word means different work in each.
#
#   gpt-image-2       low 16          medium 48   high 96
#   gpt-image-2.5     low 16  med 24  high   48   xhigh 64  max 96
#
# i.e. 2.5's `high` costs what 2's `medium` cost, and 2.5's `max` costs
# what 2's `high` cost — the two new names (`medium`, `xhigh`) fill gaps
# that did not exist before. Priced identically per token in both.
#
# These are not guesses. OpenAI documents the 2.5 quality enum, and the
# published 1024x1024 output-token counts (196 / 439 / 1,756 / 3,122 /
# 7,024) invert through the formula below onto exactly these integers —
# all five, to the token. The independent check is a real OpenRouter call:
# 1824x1024 at `low` reported image_tokens=140, and this function returns
# 140.
GPT_IMAGE_2_Q_MAP = {"low": 16, "medium": 48, "high": 96}
GPT_IMAGE_2_5_Q_MAP = {"low": 16, "medium": 24, "high": 48, "xhigh": 64, "max": 96}

# The rungs each family offers, cheapest first, as the wizard shows them.
# gpt-image-2 has shown title case since before 2.5 existed and is
# lowercased on the wire; the 2.5 names are used verbatim because the API
# enum is lowercase and "Xhigh" reads like a typo.
GPT_IMAGE_2_QUALITIES = ["Low", "Medium", "High"]
GPT_IMAGE_2_5_QUALITIES = ["low", "medium", "high", "xhigh", "max"]

# Every model billed by this token model, on the DIRECT OpenAI provider.
# OpenRouter's copies carry an "openai/" prefix and are listed in the
# OpenRouter provider instead; `gpt_image_2_quality_ladder` below handles
# both spellings.
GPT_IMAGE_2_FAMILY = (
    "gpt-image-2",
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
)


def gpt_image_2_quality_ladder(model: str) -> tuple[list[str], dict]:
    """(rungs to offer, q_map to price them with) for a model id.

    One lookup for both providers and the wizard summary, so a model can
    never be offered one ladder and billed against another — the mistake
    that costs 4x, since 2.5 reused the names `medium` and `high` for
    different amounts of work.

    Matches on the id substring so it works for the direct spelling
    ("gpt-image-2.5-flare") and OpenRouter's prefixed one
    ("openai/gpt-image-2.5-flare") alike. Note "gpt-image-2" is NOT a
    substring match for the 2.5 ids' family test — "gpt-image-2.5" is
    checked, and plain "gpt-image-2" does not contain it.
    """
    if "gpt-image-2.5" in model:
        return list(GPT_IMAGE_2_5_QUALITIES), GPT_IMAGE_2_5_Q_MAP
    return list(GPT_IMAGE_2_QUALITIES), GPT_IMAGE_2_Q_MAP


def calc_gpt_image2_tokens(
    width: int, height: int, quality: str, q_map: dict | None = None
) -> int:
    """Calculate output tokens for gpt-image-2.

    Uses a proportional grid model where quality determines the number of cells
    along the long side, scaled by aspect ratio on the short side.
    A pixel factor normalises the result relative to a 2 MP baseline.

    Args:
        width:   Output image width in pixels.
        height:  Output image height in pixels.
        quality: A key of `q_map` (case-insensitive).
        q_map:   Quality -> grid-cells mapping; defaults to
                 :data:`GPT_IMAGE_2_Q_MAP`. Pass
                 :data:`GPT_IMAGE_2_5_Q_MAP` for the GPT Image 2.5
                 family, whose ladder is shifted and two rungs longer.

    Returns:
        Estimated number of output tokens (ceiling).
    """
    q = (q_map or GPT_IMAGE_2_Q_MAP)[quality.lower()]

    long_side = max(width, height)
    short_side = min(width, height)

    q_scaled = round(q * short_side / long_side)

    q_width = q if width >= height else q_scaled
    q_height = q_scaled if width >= height else q

    grid_area = q_width * q_height

    pixel_factor = (2_000_000 + width * height) / 4_000_000

    return math.ceil(grid_area * pixel_factor)


def gpt_image_2_token_cost(
    width: int, height: int, quality: str, q_map: dict | None = None
) -> tuple[int, float]:
    """(output tokens, USD) for one image of the gpt-image-2 token family.

    The single home for "tokens x output rate": OpenAI's ``gpt-image-2``
    (direct) and OpenRouter's GPT Image 2.5 family bill identically — per
    output image token at :data:`GPT_IMAGE_2_PRICE_PER_MTOK` — so both
    providers price through here instead of repeating the arithmetic.

    Exact, not approximate: real 1824x1024 ``low`` calls on OpenRouter
    (2026-09-12) reported ``image_tokens: 140`` and a completions cost of
    $0.0042 — on BOTH 2.5 tiers, with ``reasoning_tokens: 0``; this
    returns (140, 0.0042).

    Pass ``q_map=GPT_IMAGE_2_5_Q_MAP`` for the GPT Image 2.5 family —
    the rates are identical but the quality ladder is not.
    """
    tokens = calc_gpt_image2_tokens(width, height, quality, q_map)
    return tokens, tokens * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000


def gpt_image_2_quality_label(
    width: int, height: int, quality: str, q_map: dict | None = None
) -> str:
    """The wizard's quality-step label for a token-billed image model.

    Shared so the OpenAI and OpenRouter steps stay worded identically;
    ``quality`` is echoed verbatim, so each provider keeps its own casing
    ("High" direct, "high" on OpenRouter). Four decimals because a `low`
    render lands under half a cent.
    """
    tokens, cost = gpt_image_2_token_cost(width, height, quality, q_map)
    return f"{quality} (~{tokens:,} tokens, ${cost:.4f})"


def gpt_image_2_quality_cost(
    width: int | None, height: int | None, quality: str, q_map: dict | None = None
) -> float:
    """USD for one image, or 0.0 when the output size isn't known yet.

    The ``None`` dimensions case is real on the OpenAI provider: picking
    "Auto (model decides)" leaves the wizard with nothing to bill against
    until the response comes back.
    """
    if width is None or height is None:
        return 0.0
    return gpt_image_2_token_cost(width, height, quality, q_map)[1]


def gpt_image_2_quality_choices(
    width: int | None,
    height: int | None,
    qualities: list[str],
    q_map: dict | None = None,
) -> list[str]:
    """The wizard's quality-step labels for a token-billed image model.

    The whole step, shared: both the OpenAI provider (``gpt-image-2``
    direct) and the OpenRouter provider (GPT Image 2.5 family) build their
    quality menu from here, so the two paths differ only in the data that
    genuinely is per-family — the ladder in ``qualities`` and its
    ``q_map``. Unknown dimensions degrade to an unpriced label rather than
    quoting a number we cannot stand behind.
    """
    if width is None or height is None:
        return [f"{q} (cost depends on output size)" for q in qualities]
    return [gpt_image_2_quality_label(width, height, q, q_map) for q in qualities]


def round_to_multiple_of_16(value: int) -> int:
    """Round a dimension to the nearest multiple of 16."""
    return round(value / 16) * 16


def _floor_to_multiple_of_16(value: float) -> int:
    """Round a dimension *down* to a multiple of 16 (used to stay under a cap)."""
    return math.floor(value / 16) * 16


def _ceil_to_multiple_of_16(value: float) -> int:
    """Round a dimension *up* to a multiple of 16 (used to stay above a floor)."""
    return math.ceil(value / 16) * 16


INCHES_PER_CM = 1 / 2.54
INCHES_PER_MM = 1 / 25.4


def physical_to_pixels(value: float, unit: str, dpi: int) -> int:
    """Convert a physical measurement to pixels at a given DPI.

    Args:
        value: The physical dimension value.
        unit: One of 'cm', 'mm', or 'in'.
        dpi: Dots per inch.

    Returns:
        Number of pixels.
    """
    if unit == "cm":
        return round(value * dpi * INCHES_PER_CM)
    if unit == "mm":
        return round(value * dpi * INCHES_PER_MM)
    return round(value * dpi)  # inches


def auto_adjust_gpt_image2_dims(width: int, height: int) -> tuple[int, int]:
    """Automatically adjust dimensions to meet gpt-image-2 requirements.

    Returns:
        Tuple of (adjusted_width, adjusted_height)
    """
    # All snapping to multiples of 16 is direction-aware: we floor when we need
    # to stay *under* a cap (max edge, max pixels) and ceil when we need to stay
    # *above* a floor (min pixels, the shorter side of an aspect cap). Plain
    # nearest rounding is what let earlier versions cross a bound by a few px
    # right after scaling to it.
    adj_width = max(round_to_multiple_of_16(width), 16)
    adj_height = max(round_to_multiple_of_16(height), 16)

    # Fix the aspect ratio first. An extreme ratio is the root cause of the
    # other constraints fighting each other (clamping the long edge then scaling
    # up for the pixel minimum can otherwise push the long edge back over the
    # max), so cap it to MAX_ASPECT before any pixel-count scaling. Ceil the
    # shorter side so the resulting ratio is <= MAX_ASPECT, never slightly over.
    if max(adj_width, adj_height) / min(adj_width, adj_height) > GPT_IMAGE_2_MAX_ASPECT:
        if adj_width > adj_height:
            adj_height = _ceil_to_multiple_of_16(adj_width / GPT_IMAGE_2_MAX_ASPECT)
        else:
            adj_width = _ceil_to_multiple_of_16(adj_height / GPT_IMAGE_2_MAX_ASPECT)

    # Scale proportionally if the longest edge exceeds the max. Clamping each
    # edge independently would shrink only the long side and leave the short
    # side untouched, collapsing the aspect ratio (e.g. a capped 3:1 turning
    # into 1.6:1). Floor so the longest edge lands at or below MAX_EDGE.
    longest = max(adj_width, adj_height)
    if longest > GPT_IMAGE_2_MAX_EDGE:
        scale = GPT_IMAGE_2_MAX_EDGE / longest
        adj_width = max(_floor_to_multiple_of_16(adj_width * scale), 16)
        adj_height = max(_floor_to_multiple_of_16(adj_height * scale), 16)

    # Scale proportionally to land within the pixel-count bounds.
    pixels = adj_width * adj_height
    if pixels > GPT_IMAGE_2_MAX_PIXELS:
        scale = math.sqrt(GPT_IMAGE_2_MAX_PIXELS / pixels)
        adj_width = max(_floor_to_multiple_of_16(adj_width * scale), 16)
        adj_height = max(_floor_to_multiple_of_16(adj_height * scale), 16)
    elif pixels < GPT_IMAGE_2_MIN_PIXELS:
        scale = math.sqrt(GPT_IMAGE_2_MIN_PIXELS / pixels)
        adj_width = _ceil_to_multiple_of_16(adj_width * scale)
        adj_height = _ceil_to_multiple_of_16(adj_height * scale)

    # The min-pixel ceil (and the edge clamp) can nudge the ratio back over the
    # cap; correct it by growing the shorter side, which only adds pixels.
    if max(adj_width, adj_height) / min(adj_width, adj_height) > GPT_IMAGE_2_MAX_ASPECT:
        if adj_width > adj_height:
            adj_height = _ceil_to_multiple_of_16(adj_width / GPT_IMAGE_2_MAX_ASPECT)
        else:
            adj_width = _ceil_to_multiple_of_16(adj_height / GPT_IMAGE_2_MAX_ASPECT)

    # Final guard: keep both edges within [16, MAX_EDGE].
    adj_width = min(max(adj_width, 16), GPT_IMAGE_2_MAX_EDGE)
    adj_height = min(max(adj_height, 16), GPT_IMAGE_2_MAX_EDGE)

    return adj_width, adj_height


def validate_gpt_image2_dims(width, height):
    """Returns list of error strings, empty if valid."""
    errors = []
    if width % 16 != 0 or height % 16 != 0:
        errors.append("Width and height must be divisible by 16")
    pixels = width * height
    if pixels < GPT_IMAGE_2_MIN_PIXELS:
        errors.append(f"Minimum {GPT_IMAGE_2_MIN_PIXELS:,} pixels")
    if pixels > GPT_IMAGE_2_MAX_PIXELS:
        errors.append(f"Maximum {GPT_IMAGE_2_MAX_PIXELS:,} pixels")
    if max(width, height) > GPT_IMAGE_2_MAX_EDGE:
        errors.append(f"Maximum edge: {GPT_IMAGE_2_MAX_EDGE}px")
    if max(width, height) / min(width, height) > GPT_IMAGE_2_MAX_ASPECT:
        errors.append("Maximum aspect ratio: 3:1")
    return errors


# Pricing constants (approximate, for reference only)
COSTS = {
    "gemini-3.1-flash-image": {
        "1K": {"fixed": 0.07},
        "2K": {"fixed": 0.10},
        "4K": {"fixed": 0.15},
    },
    "gemini-3-pro-image": {
        "1K": {"fixed": 0.14},
        "2K": {"fixed": 0.14},
        "4K": {"fixed": 0.25},
    },
    # Nano Banana 2 Lite: 1K only, all 14 aspect ratios of the Gemini 3.x
    # image family. Cheapest 1K option — supersedes the now-removed
    # gemini-2.5-flash-image.
    "gemini-3.1-flash-lite-image": {
        "1K": {"fixed": 0.034},
    },
    # Riverflow 2.5, re-verified 2026-09-11 against /endpoints. Both tiers
    # had drifted well below the figures recorded here (fast 2K was $0.04,
    # pro 4K was $0.33 — roughly double the real charge). The drift went
    # unnoticed because the live-pricing override that should have corrected
    # it at runtime was reading the wrong response envelope; with that fixed
    # these numbers are the offline fallback, so keep them honest.
    "sourceful/riverflow-v2.5-fast": {
        "1K": {"fixed": 0.019},
        "2K": {"fixed": 0.021},
    },
    "sourceful/riverflow-v2.5-pro": {
        "1K": {"fixed": 0.13},
        "2K": {"fixed": 0.15},
        "4K": {"fixed": 0.17},
    },
    # Seedream 5.0 Lite: one flat price at every tier, like the 4.5 it
    # replaces, but cheaper ($0.035 vs $0.04) and with no 1K tier — its
    # `resolution` enum is ["2K","4K"]. Input references are free.
    "bytedance-seed/seedream-5-0-lite": {
        "2K": {"fixed": 0.035},
        "4K": {"fixed": 0.035},
    },
    # Seedream 5.0 Pro: the first Seed tier with a genuine per-resolution
    # price — 2K bills as the `high_resolution` pricing variant at double
    # the 1K rate — and the first to charge for input references at all
    # ($0.003 flat each, like Qwen). Caps at 2K (no 4K).
    "bytedance-seed/seedream-5-0-pro": {
        "1K": {"fixed": 0.045},
        "2K": {"fixed": 0.09},
        "input_flat": 0.003,
    },
    "black-forest-labs/flux.2-klein-4b": {
        "1K": {"fixed": 0.014},
        "2K": {"fixed": 0.017},
    },
    "black-forest-labs/flux.2-flex": {
        "1K": {"fixed": 0.06},
        "2K": {"fixed": 0.24},
        "input_mp_rate": 0.06,
    },
    "black-forest-labs/flux.2-pro": {
        "1K": {"fixed": 0.03},
        "2K": {"fixed": 0.075},
        "input_mp_rate": 0.015,
    },
    "black-forest-labs/flux.2-max": {
        "1K": {"fixed": 0.07},
        "2K": {"fixed": 0.16},
        "input_mp_rate": 0.03,
    },
}

# OpenRouter prefixed versions for Gemini models
# OpenRouter-prefixed Gemini models (share COSTS entries with their bare-name
# counterparts so the wizard — which only ever sees raw model IDs — works the
# same on both providers).
COSTS["google/gemini-3.1-flash-image"] = COSTS["gemini-3.1-flash-image"]
COSTS["google/gemini-3-pro-image"] = COSTS["gemini-3-pro-image"]
COSTS["google/gemini-3.1-flash-lite-image"] = COSTS["gemini-3.1-flash-lite-image"]


# OpenRouter OpenAI models (using token-based pricing like gpt-image-2).
#
# `openai/gpt-5.4-image-2` used to live here with 1K/2K/4K rows. It was
# retired from the OpenRouter provider once a survey of all eight OpenAI
# entries in /api/v1/images/models (2026-09-12) showed that not one of them
# advertises a `resolution` parameter: those three tiers were never real,
# the field went out un-advertised, `quality` was left at the upstream
# default, and the prices below corresponded to nothing billable.

# The GPT Image 2.5 family (openai/gpt-image-2.5-flare, -sunburst) has NO
# COSTS row on purpose. It is billed purely per output token at $30/Mtok —
# the same rate as gpt-image-2, snapshot 2026-09-12 — and the OpenRouter
# provider always sends an explicit `size`, so its price is a function of
# (width, height, quality) that `calc_gpt_image2_tokens` computes exactly
# rather than a per-tier constant to look up. See
# `OpenRouterProvider._gpt_image_25_price`.
#
# This was measured, not assumed: a 1824x1024 `low` call reported
# image_tokens=140 / completions cost $0.0042, and calc_gpt_image2_tokens
# predicts exactly 140 tokens -> $0.0042. A flat per-quality table would
# have quoted $0.006 for the same render (41% high) and tripped the >10%
# reconciliation warning on essentially every run.

# Microsoft MAI Image 2.6 (Azure, via OpenRouter). Token-billed, not
# per-image: output_image $38/Mtok, input_image $8/Mtok, input_text $5/Mtok
# (snapshot 2026-09-11 from /api/v1/images/models/microsoft/mai-image-2.6/
# endpoints). No resolution tiers — the single "Standard" entry estimates a
# typical ~1MP output at ≈4,000 image tokens, so 4,000 x $38/Mtok ≈ $0.15;
# the post-call usage.cost line reports the real charge.
COSTS["microsoft/mai-image-2.6"] = {
    "Standard": {"fixed": 0.15},
}

# MAI Image 2.6 Flash (same Azure endpoint family, half the output price):
# output_image $19/Mtok, input_image $2.50/Mtok, input_text $1.75/Mtok
# (snapshot 2026-09-11 from /api/v1/images/models/microsoft/mai-image-2.6-flash/
# endpoints). Same ~4,000-image-token assumption as the precision tier, so
# 4,000 x $19/Mtok ≈ $0.08; usage.cost reports the real charge.
COSTS["microsoft/mai-image-2.6-flash"] = {
    "Standard": {"fixed": 0.08},
}

# Krea 2 family (via OpenRouter). Flat per-image pricing; 1K is the only
# resolution tier the descriptor advertises, so each row has a single "1K"
# key. NOTE: unlike every other OpenRouter model here, the /endpoints
# response carries an EMPTY `pricing` array (checked 2026-07-26), so live
# discovery cannot price these — `_tier_price` always falls back to this
# table. Figures are the "from $X/image" headline on each model page
# (snapshot 2026-07-26); usage.cost reports the real charge after the call.
COSTS["krea/krea-2-medium-turbo"] = {"1K": {"fixed": 0.015}}
COSTS["krea/krea-2-medium"] = {"1K": {"fixed": 0.03}}
COSTS["krea/krea-2-large"] = {"1K": {"fixed": 0.06}}

# xAI Grok Imagine 2.0 (via OpenRouter). Unique in the catalog: the price
# depends on TWO axes — the resolution tier (1K/2K) AND a `quality` enum
# (low/medium) the retired image-quality tier didn't have — so the keys
# here are the compound "<tier> <quality>" wizard keys, not bare tiers.
# Per-image pricing from
# /api/v1/images/models/x-ai/grok-imagine-image-2.0/endpoints, snapshot
# 2026-09-12 (variants low_1k / medium_1k / low_2k / medium_2k); input
# images are a flat $0.01 each regardless of size (input_flat, not
# per-megapixel), unchanged from the retired tier.
COSTS["x-ai/grok-imagine-image-2.0"] = {
    "1K low": {"fixed": 0.04},
    "1K medium": {"fixed": 0.06},
    "2K low": {"fixed": 0.06},
    "2K medium": {"fixed": 0.08},
    "input_flat": 0.01,
}

# Qwen Image 3 family (via OpenRouter). Two tiers, flat per-image pricing
# from /api/v1/images/models/qwen/qwen-image-3{,-pro}/endpoints, snapshot
# 2026-08-06: base $0.03 flat (1K and 2K bill the same), pro $0.04 (1K) /
# $0.075 (2K); input reference images cost a flat $0.003 each on both tiers.
COSTS["qwen/qwen-image-3"] = {
    "1K": {"fixed": 0.03},
    "2K": {"fixed": 0.03},
    "input_flat": 0.003,
}
COSTS["qwen/qwen-image-3-pro"] = {
    "1K": {"fixed": 0.04},
    "2K": {"fixed": 0.075},
    "input_flat": 0.003,
}

# Meta Muse Image (via OpenRouter). Flat $0.01/image — the cheapest model
# in the catalog. Like Krea, live discovery CANNOT price it: its
# /api/v1/images/models/meta/muse-image/endpoints response carries an empty
# `endpoints` array outright (checked 2026-09-12), so `_tier_price` always
# falls back to this table. The figure is the model page's headline and was
# confirmed against three real calls (usage.cost == 0.01 exactly, every
# time), so the >10% reconciliation warning should never fire.
#
# Single "Standard" key: the model exposes no resolution tiers — see the
# _META_MODELS notes in the OpenRouter provider for why.
COSTS["meta/muse-image"] = {"Standard": {"fixed": 0.01}}

# Recraft v4.1 family (via OpenRouter). Two axes: output (raster vs. SVG
# vector) × tier (base/utility vs. pro). Flat per-image pricing from each
# model's /endpoints entry, snapshot 2026-07-07; no resolution tiers (the
# API exposes no resolution/aspect_ratio parameter for the family), hence
# the single "Standard" key. No input megapixel billing.
COSTS["recraft/recraft-v4.1"] = {"Standard": {"fixed": 0.035}}
COSTS["recraft/recraft-v4.1-pro"] = {"Standard": {"fixed": 0.21}}
COSTS["recraft/recraft-v4.1-utility"] = {"Standard": {"fixed": 0.035}}
COSTS["recraft/recraft-v4.1-utility-pro"] = {"Standard": {"fixed": 0.21}}
COSTS["recraft/recraft-v4.1-vector"] = {"Standard": {"fixed": 0.08}}
COSTS["recraft/recraft-v4.1-pro-vector"] = {"Standard": {"fixed": 0.30}}

# Recraft style axis (issue #9). The slug set is shared across the v4.1
# family and comes from Recraft's public style taxonomy; it changes rarely,
# so a hardcoded list is the right cut until live discovery (#3) grows a
# style source. Forwarded verbatim through GenerationRequest.extras.
RECRAFT_STYLE_SLUGS = [
    "any",
    "realistic_image",
    "digital_illustration",
    "vector_illustration",
    "icon",
]

RECRAFT_STYLE_LABELS = {
    "any": "any — model decides",
    "realistic_image": "realistic_image — photographic look",
    "digital_illustration": "digital_illustration — painted/drawn look",
    "vector_illustration": "vector_illustration — flat geometric shapes",
    "icon": "icon — simple pictogram",
}

RECRAFT_BRAND_COLOR_HELP = (
    "comma-separated #RRGGBB values, e.g. #FFAA00, #112233 (empty = none)"
)


def recraft_default_style(model: str) -> str:
    """Default style slug for a Recraft variant.

    Vector variants pair naturally with vector_illustration; every raster
    variant defaults to realistic_image (Recraft's own default). Guarantees
    extras["style"] is never silently empty when the user skips the picker.
    """
    return "vector_illustration" if "vector" in model else "realistic_image"


RATIO_TO_RESOLUTION = {
    "1:1": "1024x1024",
    "2:3": "832x1248",
    "3:2": "1248x832",
    "3:4": "864x1184",
    "4:3": "1184x864",
    "4:5": "896x1152",
    "5:4": "1152x896",
    "9:16": "768x1344",
    "16:9": "1344x768",
    "21:9": "1536x672",
    "1:4": "512x2048",
    "4:1": "2048x512",
    "1:8": "512x4096",
    "8:1": "4096x512",
}

GEMINI_RESOLUTIONS = RATIO_TO_RESOLUTION
OPENROUTER_RESOLUTIONS = RATIO_TO_RESOLUTION

OPENROUTER_STANDARD_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
]

ASPECT_RATIO_VALUES = {
    "1:1": 1.0,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "3:4": 3 / 4,
    "4:3": 4 / 3,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
    "1:4": 1 / 4,
    "4:1": 4.0,
    "1:8": 1 / 8,
    "8:1": 8.0,
    "1024x1024 (Square)": 1.0,
    "1024x1536 (Vertical)": 1024 / 1536,
    "1536x1024 (Horizontal)": 1536 / 1024,
}

PRESET_PROMPTS_EDIT = [
    "Outpaint the provided image, maintain all existing details. Preserve the exact composition and identity.",
    "The quality of this logo is poor, recreate it faithfully as if it were vector-based, with sharp edges and limited colors.",
    "Upscale this photo 4x. Preserve the exact composition and identity. Remove JPEG artifacts and noise, enhance real details only. Do not add or remove objects. Do not change facial features. Do not hallucinate text or logos; if unreadable, keep it unreadable. High-resolution output.",
    "Object Removal (High Quality)",
    "Convert this photo into a classic oil painting style.",
    'Restore this photograph using **strict conservation restoration**. Remove only physical damage: **tears, scratches, scuffs, dust spots, stains, crease lines, and fold marks**. **Do not change anything else.** Keep **exactly** the original composition, framing, geometry, perspective, colors, white balance, exposure, contrast, saturation, grain, sharpness, and texture. Do **not** add, remove, or alter any objects, people, faces, hair, clothing, background details, text, logos, or patterns. Do **not** beautify, retouch skin, or "improve" lighting. Reconstruct missing areas by copying/repairing from the **nearest surrounding pixels** so the result matches the original. Output a **1:1 faithful restoration** at the same resolution.',
    "Transform the input photo into a Japanese manga illustration. Preserve the person identity, pose, clothing, and background composition. Clean black ink lineart, confident contours, simplified shapes, screentone shading, high-contrast black and white, crisp lines, minimal gray tones, manga panel style, detailed eyes and hair with ink strokes, no photorealistic texture.",
    "Convert the input photo into a high-quality anime illustration. Preserve identity and facial features. Cel shading, clean linework, smooth gradient highlights, stylized but realistic proportions, vibrant but controlled colors, sharp eyes, defined hair shapes, studio anime lighting.",
    "Give this portrait a 1950s vintage film look.",
    "Turn the photo into a shoujo manga style illustration. Delicate lineart, soft screentones, elegant facial features, sparkly eyes, light blush, airy hair highlights, romantic composition, clean black-and-white manga look.",
    "Transform the uploaded photo into a black-and-white graphite pencil drawing. Use clean line art with cross-hatching for shadows and volume, visible paper texture, and no solid black fills. Keep the exact composition, subject identity, pose, proportions, and camera framing from the original photo. Simplify the background slightly but keep it consistent. No color. Avoid: color, watercolor, oil paint, digital painting, CGI/3D, cartoon/anime, vector-clean outlines, automatic sketch filter look, blur, noisy artifacts, soft airbrushed shading, heavy solid blacks.",
    "Convert the photo into a classic pencil illustration style: precise ink-like pencil outlines, diagonal and cross-hatching for skies/shadows, graphite-only shading (no smooth airbrush gradients), detailed textures on hair/clothing, and a sketchbook look. Maintain the original photo composition and subject identity exactly. Monochrome only. Avoid: color, watercolor, oil paint, digital painting, CGI/3D, cartoon/anime, vector-clean outlines, automatic sketch filter look, blur, noisy artifacts, soft airbrushed shading, heavy solid blacks.",
    "Restore this scanned page with maximum fidelity. Only perform non-destructive cleanup: remove dust/specks, scan noise, paper texture and stains; normalize the halftone/screen pattern to be uniform; correct slight skew. Do NOT redraw, reinterpret, or invent any content. Preserve all original linework, shapes, proportions, fonts, and text exactly. No style changes. Output a clean, flat, high-resolution image that matches the original as closely as possible.",
    "Clean this scan, removing halftones and making everything more uniform.",
    "Rebuild the business card as a flat print file. Canvas size: 91×61 mm including 3 mm bleed on all sides (final trim 85×55 mm). Keep all text inside a 4 mm safe margin from the trim edge. Match the original layout from the reference photo. Output: 300 DPI. Do NOT draw any crop marks, trim marks, registration marks, cut lines, corner marks, rulers, or measurement annotations; extend the artwork to the full canvas edges as a single flat design.",
    "Perform conservative color restoration only on the provided 1970s photo. Correct color cast (yellow/magenta/green), restore faded colors, and rebalance white balance to a natural analog-photo look. Do not change any details: keep identical geometry, composition, crop, perspective, faces, skin texture, hair, edges, background, text, film grain, dust, scratches, stains, and any imperfections. No enhancement: no denoise, no sharpening, no deblur, no upscaling, no HDR, no relighting, no beautification. Output must match the original framing and resolution; only chroma/tonal color values may change.",
    "Convert this image into a clean, black and white line art. Use sharp black outlines on a pure white background. Remove all shading, colors, and gradients. It must look like a high-quality adult coloring book page, staying faithful to the original subject and background details.",
    """Perform a strictly conservative photo restoration on the provided image. Goal: improve readability while maintaining absolute faithfulness to the original photo. Allowed adjustments only:
1. neutralize the strong blue/purple color cast with a realistic daylight white balance
2. exposure and contrast correction (no HDR, no dramatic changes)
3. gentle noise/grain reduction while preserving natural film grain
Hard constraints: do not add, remove, move, or alter any real objects or people; do not change faces, bodies, clothing, background, geometry, perspective, cropping, or composition. Do not invent missing details. No style transfer. Output must look like the same photograph, only corrected.""",
    "Upscale without any deformation, and add an empty 10% outpainting margin around the edges, keeping the same background style.",
    "Custom Prompt",
]

PRESET_PROMPTS_GENERATE = [
    "A futuristic cyberpunk cityscape at night, neon lights, rain, high detail.",
    "A cute minimalist vector logo of a fox.",
    "A photorealistic portrait of an astronaut on Mars, cinematic lighting.",
    "Abstract geometric patterns, vibrant colors, 3d render style.",
    "A serene japanese garden with cherry blossoms, watercolor style.",
    "Isometric view of a cozy coffee shop interior.",
    "A retro-style BW lettering with thick outline",
    "1990s Memphis Style Logo",
    "Business Card",
    "APPROVED Stamp",
    "Generic Logotype",
    "Comic Book Style Text",
    "Pokémon Style Lettering",
    "Minecraft Style Lettering",
    "Custom Prompt",
]

PRESET_PROMPTS_DUAL = [
    "Combine the contents of IMG_1 and IMG_2 into a coherent scene.",
    "Use the composition of IMG_1 and the style of IMG_2.",
    "IMG_1 is the subject, IMG_2 is the background.",
    "Create a vintage etching / engraved illustration double exposure using two input photos. Use IMG_1 as the main subject silhouette and keep its pose, proportions, and outline faithful. Use IMG_2 as the internal scene, visible only inside the silhouette of IMG_1 (no spill outside the outline). Convert everything to black-and-white ink linework with cross-hatching and etched shading, consistent line weight, high detail. Fit and scale IMG_2 to the silhouette while preserving its aspect ratio; adjust position for a pleasing composition. Clean white background, no text, no frame, no extra objects.The outer area must remain blank white; all texture must be inside the silhouette only.",
    "Custom Prompt",
]
