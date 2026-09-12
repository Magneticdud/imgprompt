import os
import sys
import io
import base64
import math
from datetime import datetime

import requests
from PIL import Image

from imgprompt.providers.base import ImageProvider, GenerationRequest
from imgprompt.providers.capabilities import get_capabilities, output_image_price
from imgprompt.images import save_image_bytes

# OpenRouter Image API base URL (replaces the legacy /chat/completions+modalities path).
OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"

# Per-call payload limit (same as legacy: OpenRouter rejects >4.5MB).
_MAX_REQUEST_SIZE = 4.5 * 1024 * 1024
# Black Forest Labs models resize to ~4MP anyway — pre-downscale large inputs.
_MAX_BFL_MEGAPIXELS = 4
# The new API caps `n` at 10. We clamp regardless of the input.
_MAX_N = 10
# Per-model caps below the global one. Recraft's descriptor pins n at 1..6
# (verified 2026-07-07); the full descriptor-driven clamp is issue #11.
#
# Krea is the one family whose descriptor omits `n` ENTIRELY (verified
# 2026-07-26) rather than pinning it to a 1..1 range like the other
# single-image models. That means `caps.n_max` is None for it and the
# descriptor-driven clamp in `_call_api` falls through to this table — so
# without the entry below Krea would inherit the global cap of 10 and the
# wizard would offer (and price) batches the model cannot produce.
#
# Qwen's descriptor DOES pin n (1..6, verified 2026-08-06), so live
# discovery already clamps it correctly — the entry here is only an
# offline/cache-miss safety net, same role as Recraft's.
#
# Meta Muse Image ships an EMPTY descriptor — `supported_parameters` is
# `{}`, not merely missing `n` (verified 2026-09-12) — so `caps.n_max` is
# None and the clamp falls through to this table, same as Krea. The cap of
# 1 is CONSERVATIVE, not measured: the probes that verified the model were
# all single-image, and a reason-then-render agentic model is an unlikely
# fit for server-side variant fan-out. If a future probe shows it honours
# n > 1, raise this number — nothing else has to change.
#
# The two Seedream 5.0 entries are full model ids rather than family
# prefixes: the tiers disagree (Lite 1..4, Pro 1..1, verified 2026-09-11)
# so `bytedance-seed/` cannot carry one number for both. `startswith`
# matches an exact id as a prefix of itself, and neither id is a prefix
# of the other, so the lookup stays unambiguous. Without these the
# offline path would inherit the global cap of 10 and the wizard would
# price a 10-variant batch Pro can never produce.
_MODEL_MAX_N_PREFIXES = {
    "recraft/": 6,
    "krea/": 1,
    "meta/": 1,
    "qwen/": 6,
    "bytedance-seed/seedream-5-0-lite": 4,
    "bytedance-seed/seedream-5-0-pro": 1,
    "x-ai/grok-imagine-image-2.0": 1,
}


def _max_n_for(model: str) -> int:
    for prefix, cap in _MODEL_MAX_N_PREFIXES.items():
        if model.startswith(prefix):
            return cap
    return _MAX_N


# (connect_timeout, read_timeout) for POST /api/v1/images. The legacy adapter
# inherited the OpenAI SDK's built-in defaults; bare `requests.post` has no
# timeout by default, so without this the CLI would hang indefinitely on
# stalled peers.
#
# Image generation on OpenRouter routinely takes 30–90s per image and gets
# even slower with `n > 1` variants, so a 60s read budget is too tight. 300s
# (5min) matches the lower end of the OpenAI SDK's defaults and covers the
# long tail without letting a truly wedged server hang the CLI forever.
_DEFAULT_TIMEOUT = (10, 300)

# Some upstream providers enforce hard bounds on the OUTPUT pixel count
# (Seed 4K family, see issues #10 and #23): a *floor* below which the
# request 400s ("image size must be at least 3686400 pixels"), and a
# *ceiling* above which it also 400s ("image size must be at most
# 16777216 pixels"). OpenRouter's own (aspect_ratio, resolution) → size
# translation lands both above and below those limits for most non-square
# ratios, so for these models we resolve an explicit `size` client-side,
# shaping the box to the aspect ratio with edges rounded to multiples of
# 16 (the /api/v1/images grid), then clamping to the floor (ceil-16) and
# the ceiling (floor-16, with a defensive re-check).
#
# Carried over from the retired seedream-4.5 to Seedream 5.0 Lite: same
# upstream endpoint (`provider_slug: seed`), and Lite's descriptor DROPS
# the 1K tier it inherited — exactly what you would expect from a model
# whose smallest legal output is 3.69MP. Both of Lite's tiers (2K =
# 4.19MP, 4K = 16.78MP) clear the floor on their own, so the floor is
# inert for menu-driven runs and only bites on the custom-dimension path.
# Keeping it is the safe direction: an unnecessary raise costs nothing
# (Lite is flat-priced at any size) whereas a missing one is a 400.
#
# Seedream 5.0 Pro is deliberately in NEITHER table. It advertises 1K
# (1.05MP, well under the old floor) AND prices it separately from its
# `high_resolution` tier, so the floor demonstrably does not apply to it;
# and it caps at 2K (4.19MP), nowhere near the ceiling. Do not add
# entries for it without verifying against a real upstream response.
_MODEL_PIXEL_FLOORS = {
    "bytedance-seed/seedream-5-0-lite": 3_686_400,
}
_MODEL_PIXEL_CEILINGS = {
    "bytedance-seed/seedream-5-0-lite": 16_777_216,
}

# Microsoft MAI family on /api/v1/images. Both tiers ship the SAME
# descriptor (seven concrete aspect ratios + "auto", no `resolution`
# parameter at all, n pinned to 1, up to 5 input references) — only the
# per-token price differs, so the wizard branches below treat them as one.
# Verified 2026-09-11 against /api/v1/images/models. Note the 2.6 family
# lifted the input-reference cap from 1 (the retired 2.5 tiers) to 5, so
# dual-image mode works here.
_MAI_MODELS = ("microsoft/mai-image-2.6", "microsoft/mai-image-2.6-flash")

# Krea 2 family on /api/v1/images. All three tiers ship an IDENTICAL
# descriptor (verified 2026-07-26 against /api/v1/images/models) — same
# seven aspect ratios, `resolution` pinned to the single value "1K", at
# most 1 input reference, no `n` parameter — and differ only in price,
# so the wizard branches below treat them as one.
_KREA_MODELS = (
    "krea/krea-2-medium-turbo",
    "krea/krea-2-medium",
    "krea/krea-2-large",
)

# The seven ratios Krea advertises, in the wizard's canonical
# OPENROUTER_RESOLUTIONS order. Note the asymmetry: 4:5 is supported but
# its mirror 5:4 is not, and 2:3 is supported but 3:4 is not — this is
# what the descriptor says, not an oversight. Anything else 400s upstream.
_KREA_RATIOS = ["1:1", "2:3", "3:2", "4:3", "4:5", "9:16", "16:9"]

# Qwen Image 3 family on /api/v1/images. Both tiers ship the SAME
# descriptor (verified 2026-08-06 against /api/v1/images/models) — thirteen
# aspect ratios, `resolution` enum {"1K", "2K"}, `n` range 1..6, up to 4
# input references — and differ only in price, so the wizard branches below
# treat them as one.
_QWEN_MODELS = ("qwen/qwen-image-3", "qwen/qwen-image-3-pro")

# Eleven of Qwen's thirteen ratios, in the wizard's canonical
# OPENROUTER_RESOLUTIONS order. `1:2` and `2:1` are also in the descriptor
# but have no RATIO_TO_RESOLUTION preset for the wizard's pixel preview —
# same treatment as Grok's phone-screen ratios — so they stay off the picker.
_QWEN_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "1:4",
    "4:1",
]

# Meta Muse Image on /api/v1/images. The odd one out of the whole catalog:
# its capability descriptor is EMPTY (`supported_parameters: {}`) and
# /api/v1/images/models/meta/muse-image/endpoints returns `{"endpoints":
# []}` — no descriptor, no live pricing, nothing. Krea at least ships a
# descriptor with an empty `pricing` array; this model ships neither, so
# every wizard branch below has to come from measurement instead.
#
# Measured against three real calls on 2026-09-12 (the ONLY source of truth
# for this model — re-measure before changing any of it):
#
#   aspect_ratio 1:1  -> 1600x1600 (1:1)
#   aspect_ratio 16:9 -> 1920x1280 (3:2)   <- clamped, NOT 16:9
#   aspect_ratio 9:16 -> 1280x1920 (2:3)   <- clamped, NOT 9:16
#
# So `aspect_ratio` IS honoured, but only as an orientation intent that
# Meta snaps to its own three shapes (~2.5MP each). The gateway's Zod
# schema accepts all 22 catalog ratios, so sending 16:9 never errors — it
# just silently returns 3:2. Offering only the three shapes the model can
# actually produce keeps the picker honest; anything wider would let the
# user pick a ratio and pay for a different one.
#
# `resolution` is accepted by the schema and then IGNORED: the 16:9 probe
# sent resolution "1K" and got 1920x1280 back. Hence the synthetic
# "Standard" tier in get_quality_choices (same treatment as MAI/Recraft) —
# deliberately outside {512,1K,2K,4K} so _build_payload never puts a
# `resolution` field on the wire.
#
# Output arrives as image/webp, not PNG; save_image_bytes already picks the
# extension from the decoded format, so nothing special is needed there.
_META_MODELS = ("meta/muse-image",)

# The three shapes Muse actually produces, in the wizard's canonical
# OPENROUTER_RESOLUTIONS order. Note these are the OUTPUT ratios measured
# above, not a descriptor enum — there is no descriptor to read.
_META_RATIOS = ["1:1", "2:3", "3:2"]

# The exact WxH each of those ratios came back as, so the wizard's summary
# can print the real shape instead of the RATIO_TO_RESOLUTION preset (which
# would claim 1024x1024 for a 1:1 that actually arrives at 1600x1600).
# These are measured constants, not a computation: Muse picks the geometry
# itself and takes no size/resolution input, so there is nothing to derive
# them from. Re-measure if the upstream model version changes.
_META_OUTPUT_SIZES = {
    "1:1": (1600, 1600),
    "2:3": (1280, 1920),
    "3:2": (1920, 1280),
}

# The OpenAI image models on /api/v1/images. All three ship the SAME
# aspect_ratio enum — eight concrete ratios plus "auto" (which the wizard
# doesn't surface for OpenRouter) — and NONE of them advertises a
# `resolution` parameter at all. Verified 2026-09-12 against
# /api/v1/images/models.
#
# The eight matter because the generic fallback branch
# (OPENROUTER_STANDARD_RATIOS + "21:9") offers 4:5 and 5:4, which this
# family does NOT support: every one of those picks 400s upstream whenever
# live discovery is unavailable (cold cache + no network). The descriptor
# override further down already trimmed them on the online path; this
# branch makes the offline path agree.
_GPT_IMAGE_MODELS = (
    "openai/gpt-5.4-image-2",
    "openai/gpt-image-2.5-flare",
    "openai/gpt-image-2.5-sunburst",
)

# Those eight ratios in the wizard's canonical OPENROUTER_RESOLUTIONS order.
_GPT_IMAGE_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]

# GPT Image 2.5 family on /api/v1/images. Both tiers ship an IDENTICAL
# descriptor AND identical /endpoints pricing (verified 2026-09-12) — same
# eight ratios, no `resolution`, `quality` enum
# {auto,low,medium,high,xhigh,max}, n 1..10, up to 16 input references,
# output_image $30/Mtok — and differ only in positioning: Flare is the
# speed tier, Sunburst the precision tier. So the wizard branches below
# treat them as one.
#
# These are the first models in the catalog whose ONLY price axis is
# `quality`: there is no resolution tier to pick, so the wizard's single
# tier step offers the quality enum directly and `quality_key` holds a bare
# quality value ("high"), not a tier ("2K") or a compound "<tier>
# <quality>" (Grok). `_split_quality_key` keys off this tuple to return
# (None, quality) for them, which is what keeps `resolution` off the wire
# and puts `quality` on it.
_GPT_IMAGE_25_MODELS = (
    "openai/gpt-image-2.5-flare",
    "openai/gpt-image-2.5-sunburst",
)

# The five concrete quality steps, cheapest first. "auto" is in the
# descriptor too but stays off the picker for the same reason "auto" is
# kept off the ratio picker: the wizard quotes a price before spending, and
# a model-chosen tier has no price to quote.
_GPT_IMAGE_25_QUALITIES = ["low", "medium", "high", "xhigh", "max"]

# Nominal pixel targets per resolution tier (the square each tier names).
# Used to derive an explicit `size` for models in _MODEL_PIXEL_FLOORS.
_TIER_PIXELS = {
    "512": 512 * 512,
    "1K": 1024 * 1024,
    "2K": 2048 * 2048,
    "4K": 4096 * 4096,
}

# Canonical resolution-tier keys for the OpenRouter wizard and the
# payload gate. Kept in lockstep with `_TIER_PIXELS` above (per-tier
# pixel targets) — this is the pairing the load-time guard enforces.
# The per-model `COSTS` rows in `presets.py` carry per-tier prices,
# but their priceable tiers are only a SUBSET of these keys (e.g.
# `512` is a resolution floor with no `COSTS` entry, and most rows
# price just `1K`/`2K`/`4K`), so `COSTS` is intentionally NOT part of
# the guarded contract. Used at the four module callsites below with
# deliberately distinct semantics — see the per-callsite comments for
# why each one tests the membership.
_VALID_TIER_QUALITY_KEYS = ("512", "1K", "2K", "4K")

# Pin the constant/`_TIER_PIXELS` pairing: if a future commit adds a
# new tier to one but not the other, the import below fails
# immediately (and `python -m pytest` short-circuits on collection)
# instead of silently producing a tier with no pixel target or a
# missing `resolution` field on the wire. Cheap insurance.
#
# We use `raise` instead of a bare `assert` so the check
# survives `python -O` (CI optimisation flag strips asserts).
# The error message includes BOTH sides so the mismatch is
# diagnosable without re-reading the source.
if set(_VALID_TIER_QUALITY_KEYS) != set(_TIER_PIXELS):
    raise RuntimeError(
        "_VALID_TIER_QUALITY_KEYS and _TIER_PIXELS drifted — "
        "every tier key must be listed in BOTH "
        f"(constant={_VALID_TIER_QUALITY_KEYS!r}, "
        f"_TIER_PIXELS keys={tuple(_TIER_PIXELS)!r})."
    )


# Models whose price depends on TWO axes: the resolution tier AND a
# separate `quality` enum. Grok Imagine 2.0 is the only one in the catalog
# (surveyed 2026-09-12): its descriptor carries resolution ["1K","2K"] *and*
# quality ["low","medium"], and /endpoints prices all four pairs
# (low_1k $0.04, medium_1k / low_2k $0.06, medium_2k $0.08). The wizard has
# a single tier step, so for these models `quality_key` is the compound
# "<tier> <quality>" string and every consumer splits it with
# `_split_quality_key` below.
_TIER_QUALITY_MODELS = ("x-ai/grok-imagine-image-2.0",)


def _split_quality_key(
    quality_key: str | None, model: str | None = None
) -> tuple[str | None, str | None]:
    """Split a wizard quality_key into (tier, quality).

    ``"2K low"`` -> ``("2K", "low")`` for the two-axis models above;
    ``"high"`` -> ``(None, "high")`` for the quality-only GPT Image 2.5
    family, which has no resolution axis at all;
    ``"2K"`` / ``"Standard"`` / ``None`` -> ``(value, None)`` for everyone
    else, so single-axis callsites keep their exact previous behaviour.

    ``model`` is optional so the pure "does this string contain a compound
    key" reading still works standalone, but every in-tree callsite passes
    it: without the model there is no way to tell the bare quality ``"high"``
    apart from a tier name, and the GPT Image 2.5 family would silently
    send ``quality`` as a ``resolution`` (dropped by the payload gate) and
    no quality at all.
    """
    if not quality_key:
        return quality_key, None
    if model in _GPT_IMAGE_25_MODELS:
        # Quality-only axis: the whole key IS the quality value. Returning
        # tier=None keeps it out of `_VALID_TIER_QUALITY_KEYS` at every
        # gate, so no `resolution` field is ever emitted for this family.
        return None, quality_key
    tier, _, quality = quality_key.partition(" ")
    return tier, (quality or None)


def _ceil16(value: float) -> int:
    """Round up to a multiple of 16 (dimension granularity of /api/v1/images)."""
    return math.ceil(value / 16) * 16


def _floor16(value: float) -> int:
    """Round down to a multiple of 16 (used when scaling DOWN to honour an
    upstream pixel ceiling — see issues #10 + #23)."""
    return max(math.floor(value / 16) * 16, 16)


def _shrink_to_ceiling(width: int, height: int, ceiling: int) -> tuple[int, int]:
    """Return a uniformly-scaled (w, h) with w*h <= ceiling; edges floored to /16.

    Sets ``scale = sqrt(ceiling / (w*h))`` and floors each scaled edge
    to a /16 multiple. Holds under the precondition that
    ``min(width, height) >= 16`` — if not, ``_floor16``'s
    ``max(..., 16)`` guard can round *up* and produce a box that
    violates the ceiling. Every current caller satisfies this:
    ``_floor_size`` passes ``_ceil16``-rounded edges (always
    >= 16); ``_build_payload``'s custom-dims branch passes
    user-supplied widths/heights but never reaches a precondition
    violation in practice today.

    Pathological inputs like ``width=8, height=2_100_000`` (w*h
    above seedream's 16.78 MP ceiling with one edge under 16) are
    NOT protected here. If a future caller can produce one, add
    an explicit guard at the caller rather than papering over it
    in this helper — see @kilocode-bot follow-up review on PR #24
    (issue #23).

    If the box already fits, return it unchanged.
    """
    if width * height <= ceiling:
        return width, height
    scale = (ceiling / (width * height)) ** 0.5
    new_width = _floor16(width * scale)
    new_height = _floor16(height * scale)
    return new_width, new_height


def _compute_pixel_size(
    model: str, aspect_ratio: str, quality_key: str
) -> tuple[int, int, bool, bool] | None:
    """Pure compute (no print) of the (w, h) that satisfies BOTH the
    pixel floor and the pixel ceiling for the requested aspect-ratio
    and tier. Returns ``(width, height, floor_active, ceiling_active)``
    or None when the ratio or the absence of either bound precludes
    an explicit size.

    ``floor_active=True`` means the requested tier sits below the floor
    AND the shaped box still cleared the floor (always, since the
    floor raise is built into the shaping). ``ceiling_active=True`` means
    the post-shaping box needed the ceiling clamp to fit (i.e. would
    have overshot the cap without it).

    The two callers — :meth:`OpenRouterProvider._floor_size` (which
    prints) and :meth:`OpenRouterProvider.resolve_effective_pixels`
    (the wizard's summary, no print) — both go through this; the
    flags are the contract that decides whether each caller fires
    its diagnostic note (no more `silent=...` private-channel kwarg).
    """
    from imgprompt.presets import ASPECT_RATIO_VALUES

    floor = _MODEL_PIXEL_FLOORS.get(model)
    ceil = _MODEL_PIXEL_CEILINGS.get(model)
    if floor is None and ceil is None:
        return None
    ratio = ASPECT_RATIO_VALUES.get(aspect_ratio)
    if ratio is None:
        return None
    if floor is not None:
        target = max(_TIER_PIXELS.get(quality_key, floor), floor)
    else:
        # No floor but a ceiling (none configured today, but the code
        # path needs to exist for a future single-cap model). Default
        # to the ceiling so unknown tiers don't shoot past it.
        target = _TIER_PIXELS.get(quality_key, ceil)
    height = _ceil16((target / ratio) ** 0.5)
    width = _ceil16(height * ratio)

    floor_active = (
        floor is not None
        and target == floor
        and _TIER_PIXELS.get(quality_key, floor) < floor
    )
    ceiling_active = False
    if ceil is not None and width * height > ceil:
        width, height = _shrink_to_ceiling(width, height, ceil)
        ceiling_active = True
    return width, height, floor_active, ceiling_active


def _looks_like_svg(data: bytes) -> bool:
    """Sniff an SVG document from its head (used when media_type is absent)."""
    head = data[:512].lstrip()
    return head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head)


# Markup the wizard / logs can detect when the provider hands back a vector
# output (Recraft and similar models). Used to route SVG bytes to a real .svg
# file instead of save_image_bytes (which would mis-detect the format and
# save a .png header that nobody can open).
_SVG_MEDIA_TYPE = "image/svg+xml"

# Identifies the application on OpenRouter's side. Kept verbatim from the legacy
# header set so dashboards that already filter on these values continue to work.
_APP_HEADERS = {
    "HTTP-Referer": "https://github.com/Magneticdud/imgprompt/",
    "X-Title": "IMGPrompt",
    "X-OpenRouter-Title": "IMGPrompt",
    "X-OpenRouter-Categories": "image-gen",
}


class OpenRouterProvider(ImageProvider):
    def __init__(self) -> None:
        # Read the key eagerly so _call_api / _headers can be invoked
        # directly (e.g. by unit tests) without having to first go through
        # run(). run() still validates the key is present before kicking
        # off real network requests.
        self._api_key: str | None = os.getenv("OPENROUTER_API_KEY")
        # Initialise so callers that invoke _call_api before run() (tests,
        # ad-hoc replays) don't hit AttributeError. run() resets this anyway
        # to clear stale state from a previous run.
        self._reported_cost: float | None = None

    @classmethod
    def provider_name(cls) -> str:
        return "OpenRouter"

    # Order is mostly flat (one entry per upstream provider). Within the
    # Google trio we keep non-Lite Flash first and Lite last — mirroring the
    # Google provider — so the model *order* in pickers is consistent across
    # the two providers even though OpenRouter's overall default stays
    # `openai/gpt-5.4-image-2` (first entry).
    @classmethod
    def supported_models(cls) -> list[str]:
        return [
            "openai/gpt-5.4-image-2",
            # GPT Image 2.5 family, speed tier first. Both are pure image
            # models (no reasoning/text leg like gpt-5.4-image-2) priced on
            # a quality axis instead of a resolution one, so a `low` render
            # is the cheapest OpenAI option here by a wide margin and `max`
            # the most expensive thing in the catalog.
            "openai/gpt-image-2.5-flare",
            "openai/gpt-image-2.5-sunburst",
            # Seedream 5.0, cheaper/higher-resolution tier first: Lite is the
            # family default. It supersedes seedream-4.5 outright — $0.035 vs
            # $0.04 flat, same 4K ceiling, same ratios — so 4.5 is retired.
            "bytedance-seed/seedream-5-0-lite",
            "bytedance-seed/seedream-5-0-pro",
            "black-forest-labs/flux.2-klein-4b",
            "black-forest-labs/flux.2-flex",
            "black-forest-labs/flux.2-pro",
            "black-forest-labs/flux.2-max",
            "sourceful/riverflow-v2.5-fast",
            "sourceful/riverflow-v2.5-pro",
            "google/gemini-3.1-flash-image",
            "google/gemini-3-pro-image",
            "google/gemini-3.1-flash-lite-image",
            # MAI 2.6 family, cheaper tier first (flash ≈$0.08 → 2.6 ≈$0.15).
            "microsoft/mai-image-2.6-flash",
            "microsoft/mai-image-2.6",
            # Grok Imagine 2.0 supersedes the retired image-quality tier:
            # same ratios and 2K ceiling, but it adds the low/medium quality
            # axis, so it is both cheaper ($0.04 low 1K vs $0.05) and
            # sharper ($0.08 medium 2K vs $0.07) than the single-axis model.
            "x-ai/grok-imagine-image-2.0",
            # Qwen Image 3 family, cheaper tier first.
            "qwen/qwen-image-3",
            "qwen/qwen-image-3-pro",
            # Krea 2 family, cheapest tier first (turbo $0.015 → large $0.06).
            "krea/krea-2-medium-turbo",
            "krea/krea-2-medium",
            "krea/krea-2-large",
            # Meta Muse Image: single model, $0.01/image — the cheapest in
            # the catalog — so it sits with the other cheap entries rather
            # than with its (nonexistent) family.
            "meta/muse-image",
            # Recraft v4.1 family, grouped at the end: two axes — output
            # (raster vs. SVG vector) × tier (base/utility vs. pro).
            "recraft/recraft-v4.1",
            "recraft/recraft-v4.1-pro",
            "recraft/recraft-v4.1-utility",
            "recraft/recraft-v4.1-utility-pro",
            "recraft/recraft-v4.1-vector",
            "recraft/recraft-v4.1-pro-vector",
        ]

    def get_resolution_choices(
        self, model: str, image_path: str | None
    ) -> tuple[list[str], str]:
        from imgprompt.presets import OPENROUTER_RESOLUTIONS, OPENROUTER_STANDARD_RATIOS

        # Google's model page documents all 14 Gemini 3.x image ratios for
        # both non-Lite Flash and Lite (Nano Banana 2):
        # https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image
        # OpenRouter's /api/v1/images is a pass-through with provider-side
        # clamping ("Providers clamp to their supported subset"), so the
        # upstream allowlist is the only real gate. 3-Pro Image's model
        # page only links back to the family-level docs without enumerating
        # the ratios, so we keep it on the conservative 10+21:9 list until
        # either Google lists them per-model or we verify against real
        # upstream responses.
        if model in _GPT_IMAGE_MODELS:
            # Eight concrete ratios, no 4:5/5:4 — see _GPT_IMAGE_MODELS for
            # why the generic fallback below is wrong for this family.
            ratio_options = list(_GPT_IMAGE_RATIOS)
        elif model in (
            "google/gemini-3.1-flash-image",
            "google/gemini-3.1-flash-lite-image",
        ):
            ratio_options = list(OPENROUTER_RESOLUTIONS.keys())
        elif model in _MAI_MODELS:
            # MAI's /api/v1/images descriptor advertises exactly these seven
            # concrete ratios (plus "auto", which the wizard doesn't surface
            # for OpenRouter): no 4:5/5:4/21:9. Verified 2026-09-11 against
            # /api/v1/images/models (identical for 2.6 and 2.6 Flash) — sending
            # anything else 400s upstream.
            ratio_options = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"]
        elif model in _KREA_MODELS:
            # Krea's descriptor lists exactly these seven (see _KREA_RATIOS
            # for the 4:5-without-5:4 asymmetry). Every one has a
            # RATIO_TO_RESOLUTION entry, so the picker can preview them all.
            ratio_options = list(_KREA_RATIOS)
        elif model in _TIER_QUALITY_MODELS:
            # Grok's descriptor (verified 2026-09-12, unchanged from the
            # retired image-quality tier) lists these seven plus phone-screen
            # ratios (9:19.5, 19.5:9, 9:20, 20:9, 1:2, 2:1) and "auto". The
            # phone ratios have no RATIO_TO_RESOLUTION entry for the wizard's
            # pixel preview, so we keep them off the picker; no 4:5/5:4/21:9
            # upstream.
            ratio_options = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"]
        elif model in _QWEN_MODELS:
            # See _QWEN_RATIOS for why 1:2/2:1 are excluded from the picker.
            ratio_options = list(_QWEN_RATIOS)
        elif model in _META_MODELS:
            # Measured, not advertised: Muse's descriptor is empty, and the
            # only three shapes it returns are 1:1, 2:3 and 3:2. See the
            # _META_MODELS block for the probe results behind this list.
            ratio_options = list(_META_RATIOS)
        elif model.startswith("recraft/"):
            # Recraft's descriptor exposes NO aspect_ratio or resolution
            # parameter (verified 2026-07-07): geometry is entirely
            # model-chosen. Single "Auto" entry; _build_payload omits the
            # aspect_ratio field for it. Early return skips the
            # closest-ratio-from-image default below (nothing to match).
            return ["Auto"], "Auto"
        else:
            ratio_options = OPENROUTER_STANDARD_RATIOS + ["21:9"]

        # Descriptor-driven override (issue #11): when the live catalog
        # advertises an aspect_ratio enum, show exactly that — filtered to
        # ratios the wizard can preview (RATIO_TO_RESOLUTION keys, canonical
        # order). The hardcoded lists above remain the offline fallback.
        caps = get_capabilities(model)
        if caps and caps.aspect_ratios:
            descriptor_options = [
                r for r in OPENROUTER_RESOLUTIONS if r in caps.aspect_ratios
            ]
            if descriptor_options:
                ratio_options = descriptor_options
        default = "1:1"
        if image_path:
            from imgprompt.images import get_closest_aspect_ratio

            default = get_closest_aspect_ratio(image_path, ratio_options)
        return ratio_options, default

    def resolve_resolution(
        self, model: str, selection: str
    ) -> tuple[str, int | None, int | None]:
        from imgprompt.presets import OPENROUTER_RESOLUTIONS

        if selection == "Auto":
            # Recraft: geometry is model-chosen; "model default" keeps the
            # summary's Pixels line honest instead of a fake 1024x1024.
            return "model default", None, None
        return OPENROUTER_RESOLUTIONS.get(selection, "1024x1024"), None, None

    def resolve_effective_pixels(
        self, model: str, aspect_ratio: str | None, quality_key: str | None
    ) -> tuple[int, int] | None:
        """Return the WxH actually written into the ``size`` field, or None.

        Mirrors the aspect-ratio+resolution branch of :meth:`_build_payload`
        (the one that sends an explicit ``size`` for models with a
        pixel floor or ceiling). Lets the wizard's summary print the
        *same* shape the API will see instead of the misleading
        ``RATIO_TO_RESOLUTION`` preset (issue #23, where 4:5 /
        1K prints "896x1152" but a 4K request actually goes out as
        ─ e.g. ─ 3664x4576 after the ceiling clamp).

        Returns None when:
          - the model has no floor / ceiling configured (we delegate
            to OpenRouter's own (aspect_ratio, resolution) translation);
          - ``aspect_ratio`` is missing or "Auto" (model-chosen geometry);
          - ``quality_key``'s tier half isn't one of the canonical tiers
            (``512`` / ``1K`` / ``2K`` / ``4K``) — if it were (e.g.
            "Standard" for a clamped model) the helper would silently
            fall back to the floor as the target, which would be
            misleading rather than informational;
          - the ratio isn't in :data:`ASPECT_RATIO_VALUES` (unknown shape).

        The caller should then fall back to the ``res_key`` preset as
        before. Goes through the pure :func:`_compute_pixel_size`
        helper directly (no print) so the diagnostic notes don't fire
        twice — once during the summary preview, once on the actual
        call.
        """
        # Muse is the one model whose real output size is known without
        # being computed: it ignores every geometry input and returns one
        # of three fixed boxes (see _META_OUTPUT_SIZES). It sends no
        # explicit `size`, so it is outside this method's usual
        # floor/ceiling remit — but the *purpose* of the method is to print
        # the shape the user will get, and here we know it exactly.
        if model in _META_MODELS:
            return _META_OUTPUT_SIZES.get(aspect_ratio or "")
        if not (_MODEL_PIXEL_FLOORS.get(model) or _MODEL_PIXEL_CEILINGS.get(model)):
            return None
        if aspect_ratio is None or aspect_ratio == "Auto":
            return None
        # Wizard-summary gate: only canonical tier resolution keys
        # (`512`/`1K`/`2K`/`4K`) get a size computed here. Non-tier
        # values (`'Standard'` for MAI/Recraft, `'high'` from the
        # OpenAI legacy) fall through to the `res_key` preset —
        # without this gate they would silently fall back to the
        # floor constant as a target and produce a misleading WxH
        # in the wizard summary.
        tier, _quality = _split_quality_key(quality_key, model)
        if tier not in _VALID_TIER_QUALITY_KEYS:
            return None
        result = _compute_pixel_size(model, aspect_ratio, tier)
        if result is None:
            return None
        width, height, _floor_active, _ceiling_active = result
        return width, height

    def get_quality_choices(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        image_path: str | None,
    ) -> tuple[list[str], str]:
        from imgprompt.presets import COSTS

        # Method kept named "quality" for backwards compatibility with the
        # wizard in imgedit.py, but the user-facing label is now "Resolution"
        # (tier like "1K"/"2K"/"4K" — matching the new docs).
        if model in _GPT_IMAGE_25_MODELS:
            # No `resolution` parameter upstream at all: this family's only
            # price axis is `quality`, so the tier step offers the quality
            # enum directly (cheapest first). None of these values is in
            # {512,1K,2K,4K}, so _build_payload never emits a `resolution`
            # field for them — it emits `quality` instead.
            sizes = list(_GPT_IMAGE_25_QUALITIES)
        elif model.startswith("openai/gpt-"):
            sizes = ["1K", "2K", "4K"]
        elif model == "sourceful/riverflow-v2.5-pro":
            sizes = ["1K", "2K", "4K"]
        elif model.startswith("black-forest-labs/"):
            sizes = ["1K", "2K"]
        elif model in (
            "google/gemini-3-pro-image",
            "google/gemini-3.1-flash-image",
        ):
            sizes = ["1K", "2K", "4K"]
        elif model == "google/gemini-3.1-flash-lite-image":
            # Nano Banana 2 Lite is documented as 1K-only — cheap is the
            # whole point of choosing it; 2K/4K would silently no-op or 400
            # upstream.
            sizes = ["1K"]
        elif model in _MAI_MODELS:
            # MAI exposes NO `resolution` parameter on /api/v1/images
            # (descriptor verified 2026-09-11, both tiers): the model picks the
            # output size from the aspect ratio alone. "Standard" is deliberately
            # outside the {512,1K,2K,4K} set so _build_payload never emits
            # a resolution field for it.
            sizes = ["Standard"]
        elif model in _KREA_MODELS:
            # Krea's `resolution` enum holds the single value "1K" (verified
            # 2026-07-26, all three tiers). Unlike MAI/Recraft the parameter
            # DOES exist, so "1K" is a real canonical tier that _build_payload
            # forwards on the wire — not a synthetic "Standard" placeholder.
            sizes = ["1K"]
        elif model == "bytedance-seed/seedream-5-0-lite":
            # Lite's `resolution` enum is ["2K","4K"] — no 1K (verified
            # 2026-09-11). The retired 4.5 offered a 1K that the upstream
            # pixel floor silently raised to 3.69MP anyway; Lite just drops
            # the pretence, so there is no sub-floor tier left to flag.
            sizes = ["2K", "4K"]
        elif model == "bytedance-seed/seedream-5-0-pro":
            # Pro is the mirror image: ["1K","2K"], no 4K (verified
            # 2026-09-11). Its 2K is billed as the `high_resolution` pricing
            # variant at double the 1K rate — see _MODEL_VARIANT_TIERS in
            # capabilities.py for the variant→tier mapping.
            sizes = ["1K", "2K"]
        elif model in _TIER_QUALITY_MODELS:
            # Grok Imagine 2.0 caps at 2K (descriptor `resolution` is exactly
            # ["1K","2K"]) but adds `quality` ["low","medium"], and both axes
            # move the price — so the single tier step offers the full 2x2
            # matrix as compound keys, ordered tier-major (which is also
            # non-decreasing in price, so the default stays the cheapest
            # row at $0.04). Verified 2026-09-12
            # against /api/v1/images/models; the per-pair prices live in
            # COSTS under the same compound keys.
            sizes = [f"{t} {q}" for t in ("1K", "2K") for q in ("low", "medium")]
        elif model in _META_MODELS:
            # Muse accepts `resolution` at the gateway and then ignores it
            # (a 1K request came back at 1920x1280, verified 2026-09-12),
            # so there is no tier to offer. "Standard" is deliberately
            # outside the {512,1K,2K,4K} set, which keeps _build_payload
            # from emitting a resolution field that would do nothing.
            sizes = ["Standard"]
        elif model.startswith("recraft/"):
            # Like MAI: no resolution parameter on /api/v1/images (verified
            # 2026-07-07). One flat-priced "Standard" tier per variant; the
            # vector variants return SVG natively (routed by media_type in
            # _save_one), so they stay good for vector edits.
            sizes = ["Standard"]
        else:
            sizes = ["1K", "2K"]

        # Descriptor-driven override (issue #11): keep only tiers the live
        # catalog advertises AND presets can price. Models whose descriptor
        # has no resolution enum (gpt: quality-based; MAI/Recraft: none) and
        # offline runs keep the hardcoded branch result.
        # Two-axis models are exempt: the filter below enumerates canonical
        # tier keys, which can never match a compound "<tier> <quality>" key
        # (nor its COSTS row), so it would silently fall through to an empty
        # list and drop the quality axis from the menu. The quality-only GPT
        # Image 2.5 family is exempt for the mirror-image reason: its keys
        # are bare quality values, so a `resolution` enum appearing upstream
        # would replace a priced quality menu with unpriced tiers.
        caps = get_capabilities(model)
        if (
            caps
            and caps.resolutions
            and model not in _TIER_QUALITY_MODELS
            and model not in _GPT_IMAGE_25_MODELS
        ):
            from imgprompt.presets import COSTS as _costs

            # Descriptor-driven tier filter: enumerate canonical tier
            # keys from the live catalog, keeping only those the
            # per-model `COSTS` row can price. Pre-existing per-
            # model hardcoded `sizes = [...]` lists (e.g. Grok
            # Imagine's `["1K","2K"]`) stay separate — they are
            # intentionally narrower than the canonical list.
            descriptor_sizes = [
                s
                for s in _VALID_TIER_QUALITY_KEYS
                if s in caps.resolutions and s in _costs.get(model, {})
            ]
            if descriptor_sizes:
                sizes = descriptor_sizes
        # .3f for cents-precise Lite pricing ($0.034). Without it the wizard
        # would round $0.034 down to "$0.03" and look cheaper than 3.1's
        # $0.07, which is misleading. Trailing zero on $0.07 → $0.070 is
        # fine and matches OpenRouter's usage-report format.
        floor = _MODEL_PIXEL_FLOORS.get(model)
        choices = []
        for s in sizes:
            label = f"{s} (${self._tier_price(model, s):.3f})"
            # Pre-flight signal for tiers the upstream floor overrides: the
            # user should know before confirming that the output will be
            # larger than the tier name suggests. No shipped model triggers
            # this today (the retired seedream-4.5's 1K was the last one —
            # 5.0 Lite drops that tier), but the rule stays so a future
            # sub-floor tier is flagged rather than silently upsized.
            if floor and _TIER_PIXELS.get(s, floor) < floor:
                label += f" — raised to upstream {floor / 1e6:.1f}MP minimum"
            choices.append(label)
        return choices, choices[0]

    def _tier_price(self, model: str, tier: str) -> float:
        """Per-image price for a tier: live OpenRouter pricing when the
        model is flat-priced per image (issue #3), hardcoded COSTS estimate
        otherwise (token-billed models, network down)."""
        from imgprompt.presets import COSTS

        live = output_image_price(model, tier)
        if live is not None:
            return live
        return COSTS[model][tier]["fixed"]

    def resolve_quality(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        selection: str,
    ) -> tuple[str, float]:
        # Split on the price suffix, not on the first space: compound
        # "<tier> <quality>" keys (Grok Imagine 2.0) contain one. Every label
        # get_quality_choices builds is f"{key} (${price})", so the key is
        # everything before " (" — including for the floor annotation that
        # can follow the price.
        quality_key = selection.split(" (")[0]
        return quality_key, self._tier_price(model, quality_key)

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def supports_dual(self) -> bool:
        return True

    def preflight_warnings(
        self, model: str, aspect_ratio: str | None, quality_key: str | None
    ) -> list[str]:
        """Descriptor mismatches worth surfacing before the API call.

        Choices made through the wizard are already descriptor-driven, so
        this mainly catches stale replays and hardcoded-fallback drift.
        Empty when discovery is unavailable (nothing to check against).
        """
        caps = get_capabilities(model)
        if caps is None:
            return []
        warnings = []
        if (
            caps.aspect_ratios
            and aspect_ratio
            and aspect_ratio != "Auto"
            and aspect_ratio not in caps.aspect_ratios
        ):
            warnings.append(
                f"{model} does not advertise aspect ratio {aspect_ratio} "
                f"(supported: {', '.join(caps.aspect_ratios)})"
            )
        # Pre-flight regex: distinguish a canonical tier quality_key
        # (`'1K'`/`'2K'`/...) from a non-tier value (legacy `'high'`,
        # MAI/Recraft `'Standard'`). The model can still surface a
        # non-canonical tier internally; we only warn when the user
        # picked a CANONICAL tier that the live descriptor doesn't
        # advertise — a hard mismatch, not a stylistic pick.
        tier, _quality = _split_quality_key(quality_key, model)
        if (
            caps.resolutions
            and tier in _VALID_TIER_QUALITY_KEYS
            and tier not in caps.resolutions
        ):
            warnings.append(
                f"{model} does not advertise resolution {tier} "
                f"(supported: {', '.join(caps.resolutions)})"
            )
        return warnings

    # ------------------------------------------------------------------ entry

    def run(self, request: GenerationRequest) -> None:
        if not self._api_key:
            print(
                "Error: OPENROUTER_API_KEY not found. Please set it in your .env file."
            )
            sys.exit(1)
        # Reset the per-run cost cache so a previous run's reported cost
        # doesn't suppress this run's announcement when the value repeats.
        self._reported_cost = None

        # request.is_batch (not a raw len() check) so dual mode — two images
        # in ONE combined call — routes to _run_variants, which forwards all
        # request.images as input_references of a single POST.
        if request.is_batch:
            self._run_input_batch(request)
        else:
            self._run_variants(request)

    # ------------------------------------------------------------------ flow

    def _run_variants(self, request: GenerationRequest) -> None:
        """Single-call path: returns 1..N images depending on request.n.

        Used for text-to-image (no input image), 1-image edits, and dual
        mode (both input images as input_references of the same call).
        Server-side batching via the `n` parameter means a single HTTP call
        yields all variants in parallel instead of looping N times.
        """
        n = max(1, min(_MAX_N, getattr(request, "n", 1)))
        img_paths = request.images or None
        try:
            results = self._call_api(request, img_paths=img_paths, n=n)
        except requests.HTTPError as e:
            self._print_http_error(e)
            return
        except Exception as e:
            print(f"\nAn error occurred during OpenRouter call: {e}")
            return

        if not results:
            print("\nError: No image returned by OpenRouter API.")
            return

        save_target = request.primary_image
        if n > 1:
            self._save_variants(results, save_target, label="Variant")
        else:
            self._save_one(results[0], save_target)

    def _run_input_batch(self, request: GenerationRequest) -> None:
        """Multi-input fan-out: each input gets `request.n` variants.

        Each input is processed in a separate API call (with its own
        input_references), so the total output count is `len(images) * n`.
        """
        n = max(1, min(_MAX_N, getattr(request, "n", 1)))
        print(f"\nStarting batch processing: {len(request.images)} images...")
        success = 0
        failed = 0
        # Basenames of inputs that produced no image (HTTP/refusal/empty), so a
        # skipped PDF page can be named in the summary. The loop always moves on
        # to the next input rather than aborting the whole batch.
        failed_inputs: list[str] = []
        for idx, img_path in enumerate(request.images, 1):
            print(
                f"\n--- Processing image {idx}/{len(request.images)}: "
                f"{os.path.basename(img_path)} ---"
            )
            try:
                results = self._call_api(request, img_paths=[img_path], n=n)
            except requests.HTTPError as e:
                self._print_http_error(e)
                failed += 1
                failed_inputs.append(os.path.basename(img_path))
                continue
            except Exception as e:
                print(f"An error occurred during OpenRouter call: {e}")
                failed += 1
                failed_inputs.append(os.path.basename(img_path))
                continue

            if not results:
                failed += 1
                failed_inputs.append(os.path.basename(img_path))
                continue
            if n > 1:
                self._save_variants(results, img_path, label=f"Variant (img {idx})")
            else:
                self._save_one(results[0], img_path)
            success += 1

        print(
            f"\n=== Batch complete: {success}/{len(request.images)} inputs OK, "
            f"{success * n} images saved, {failed} failed ==="
        )
        if failed_inputs:
            print(
                f">> Skipped/failed ({len(failed_inputs)}): " + ", ".join(failed_inputs)
            )

    def _save_one(
        self,
        item: tuple[bytes, str | None],
        original_path: str | None,
    ) -> None:
        """Save one decoded image, picking .svg when needed.

        `imgprompt.images.save_image_bytes` drives extension choice off
        PIL-detectable formats (PNG/JPEG/WEBP) — vector outputs need a
        hand-written path with the canonical timestamped name so the user
        gets a file that opens in vector viewers without renaming.
        """
        img_bytes, media_type = item
        if media_type is None and _looks_like_svg(img_bytes):
            # Defensive: if the server ever omits media_type on a vector
            # item, save_image_bytes would mis-detect the SVG text and write
            # an unopenable .png. Sniffing the document head keeps the .svg
            # path working regardless.
            media_type = _SVG_MEDIA_TYPE
        if media_type == _SVG_MEDIA_TYPE:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if original_path:
                base = os.path.splitext(os.path.basename(original_path))[0]
                out_dir = os.path.dirname(original_path) or "."
                out_path = os.path.join(out_dir, f"edited_{ts}_{base}.svg")
            else:
                out_path = f"generated_{ts}.svg"
            if os.path.exists(out_path):
                root, _ = os.path.splitext(out_path)
                counter = 2
                while os.path.exists(f"{root}_{counter}.svg"):
                    counter += 1
                out_path = f"{root}_{counter}.svg"
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            print(f"File saved successfully as {out_path}")
            return
        save_image_bytes(img_bytes, original_path)

    def _save_variants(
        self,
        results: list[tuple[bytes, str | None]],
        original_path,
        label: str,
    ) -> None:
        for i, item in enumerate(results, 1):
            print(f"\n{label} {i}/{len(results)}")
            self._save_one(item, original_path)

    def _print_http_error(self, exc: requests.HTTPError) -> None:
        try:
            body = exc.response.text
        except Exception:
            body = str(exc)
        print(f"\nError: OpenRouter request failed ({len(body)} chars): {body[:500]}")

    # ------------------------------------------------------------------ HTTP

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **_APP_HEADERS,
        }

    def _build_payload(
        self, request: GenerationRequest, img_paths: list[str] | None = None
    ) -> dict:
        """Construct the /api/v1/images body.

        Rules (matches the docs):
        - `size` shorthand and `resolution+aspect_ratio` are mutually exclusive;
          pick `size` only when the user gave a custom pixel dimension.
        - `n` is always present (server defaults to 1, but being explicit avoids
          provider ambiguity on models that cap `n`).
        - `input_references` is used for image-to-image (replaces the legacy
          chat-content array).
        - Forward any opt-in advanced params from `request.extras` verbatim
          (output_format, background, seed, provider.options, etc.).
          The Recraft style picker (issue #9) relies on this merge for its
          "style" and "colors" keys — don't strip unknown extras keys when
          refactoring this path.
        """
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            **request.extras,
        }
        # `n` is owned by `_call_api`: it has the canonical clamp and the
        # per-call override that input-batch fan-out uses. Leaving it out
        # here avoids two copies of the same logic drifting apart.

        # Two-axis models carry "<tier> <quality>" in quality_key. `quality`
        # is its own wire field and is orthogonal to how the geometry is
        # expressed, so it is set here — before the size/resolution branch —
        # and rides along on the explicit-`size` path too.
        tier, quality = _split_quality_key(request.quality_key, request.model)
        if quality:
            payload["quality"] = quality

        floor = _MODEL_PIXEL_FLOORS.get(request.model)
        ceil = _MODEL_PIXEL_CEILINGS.get(request.model)
        if request.width and request.height:
            width, height = request.width, request.height
            if floor and width * height < floor:
                scale = (floor / (width * height)) ** 0.5
                width, height = _ceil16(width * scale), _ceil16(height * scale)
                print(
                    f"[OpenRouter] {request.model}: raising "
                    f"{request.width}x{request.height} to {width}x{height} to "
                    f"meet the upstream {floor:,}-pixel minimum."
                )
            elif ceil and width * height > ceil:
                # Symmetric counterpart of the floor raise: large custom
                # dimensions (e.g. 10000x2000) would 400 upstream otherwise.
                width, height = _shrink_to_ceiling(width, height, ceil)
                print(
                    f"[OpenRouter] {request.model}: lowering "
                    f"{request.width}x{request.height} to {width}x{height} to "
                    f"meet the upstream {ceil:,}-pixel maximum."
                )
            payload["size"] = f"{width}x{height}"
        else:
            explicit = (floor or ceil) and self._floor_size(
                request.model, request.aspect_ratio, tier
            )
            if explicit:
                payload["size"] = explicit
            else:
                # "Auto" (Recraft: geometry is model-chosen, the descriptor
                # has no aspect_ratio parameter at all) must not reach the
                # wire — upstream would reject the unknown field value.
                if request.aspect_ratio and request.aspect_ratio != "Auto":
                    payload["aspect_ratio"] = request.aspect_ratio
                # The new API also accepts "512" as a normalized tier — pass
                # through anything in the {512,1K,2K,4K} set (older code omitted
                # "1K" because chat-completions had no resolution field, so this
                # is also a small bugfix).
                # Payload gate: only canonical tier keys are forwarded
                # as `resolution` on the wire. Other values (`'high'`
                # legacy, `'Standard'` for MAI/Recraft, `'Auto'`) would
                # 400 upstream if sent as `resolution` — by skipping
                # them here we fall through to either a `size`
                # shorthand or the no-resolution path.
                if tier in _VALID_TIER_QUALITY_KEYS:
                    payload["resolution"] = tier

        if img_paths:
            payload["input_references"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": self._img_to_data_url(p, request.model)},
                }
                for p in img_paths
            ]
        return payload

    def _floor_size(
        self, model: str, aspect_ratio: str, quality_key: str
    ) -> str | None:
        """Compute the explicit WxH string AND emit the clamping diagnostic
        the user sees on the wire.

        The actual number crunching lives in the module-level helper
        :func:`_compute_pixel_size`: it returns the (w, h) plus
        ``floor_active`` / ``ceiling_active`` flags so this wrapper
        knows which (if any) note to print — without needing a
        ``silent=...`` private channel (issue #23 — see the
        ``resolve_effective_pixels`` callsite for the no-print
        variant).

        Returns None when the ratio is unknown — the caller then falls
        back to the plain ``aspect_ratio``+``resolution`` fields and
        lets upstream decide.
        """
        result = _compute_pixel_size(model, aspect_ratio, quality_key)
        if result is None:
            return None
        width, height, floor_active, ceiling_active = result
        floor = _MODEL_PIXEL_FLOORS.get(model)
        ceil = _MODEL_PIXEL_CEILINGS.get(model)
        if floor_active:
            print(
                f"[OpenRouter] {model}: {quality_key} on {aspect_ratio} is "
                f"below the upstream {floor:,}-pixel minimum; sending "
                f"{width}x{height} instead."
            )
        if ceiling_active:
            print(
                f"[OpenRouter] {model}: {quality_key} on {aspect_ratio} is "
                f"above the upstream {ceil:,}-pixel maximum; sending "
                f"{width}x{height} instead."
            )
        return f"{width}x{height}"

    def _call_api(
        self,
        request: GenerationRequest,
        img_paths: list[str] | None = None,
        n: int = 1,
    ) -> list[tuple[bytes, str | None]]:
        """POST /api/v1/images and decode the `data[]` array.

        Returns a list of (decoded_bytes, media_type) tuples so callers can
        pick the right file extension for vector outputs (SVG). Raises
        requests.HTTPError on non-2xx for the caller to format.

        Raises a warning visibly when the API returns fewer images than
        `n` so silent truncation from server-side caps doesn't surprise
        the user.
        """
        caps = get_capabilities(request.model)

        # Per-model input_references cap (descriptor-driven): trim instead
        # of letting upstream 400 on "too many references".
        if (
            img_paths
            and caps
            and caps.input_refs_max is not None
            and len(img_paths) > caps.input_refs_max
        ):
            print(
                f"\n[OpenRouter] {request.model} accepts at most "
                f"{caps.input_refs_max} input reference(s); using the first "
                f"{caps.input_refs_max} of {len(img_paths)}."
            )
            img_paths = img_paths[: caps.input_refs_max]

        payload = self._build_payload(request, img_paths=img_paths)
        requested_n = max(1, min(_max_n_for(request.model), n))
        # Per-model n cap: the descriptor's n.max wins over the hardcoded
        # prefix table when available (e.g. gemini/flux/sourceful cap at 1).
        # `is not None` rather than truthiness so a (nonsensical but
        # possible) n_max=0 from upstream isn't mistaken for "no cap".
        cap_n = (
            caps.n_max if caps and caps.n_max is not None else _max_n_for(request.model)
        )
        clamped_n = max(1, min(cap_n, requested_n))
        if clamped_n < requested_n:
            print(
                f"\n[OpenRouter] {request.model} caps n at {cap_n}; "
                f"requesting {clamped_n} instead of {requested_n}."
            )
        payload["n"] = clamped_n

        resp = requests.post(
            OPENROUTER_IMAGES_URL,
            json=payload,
            headers=self._headers(),
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()

        self._maybe_report_cost(body, request)

        decoded: list[tuple[bytes, str | None]] = []
        skipped_empty = 0
        for item in body.get("data") or []:
            b64 = item.get("b64_json") or ""
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            if not b64:
                # Defensive: a malformed item with no payload must not
                # crash the whole call. Log and skip; the length warning
                # below will surface the discrepancy.
                skipped_empty += 1
                continue
            decoded.append((base64.b64decode(b64), item.get("media_type")))

        if skipped_empty:
            print(
                f"\n[OpenRouter] Warning: {skipped_empty} response item(s) "
                "had no decodable payload and were skipped."
            )
        if len(decoded) != payload["n"]:
            print(
                f"\n[OpenRouter] Warning: requested {payload['n']} variant(s) "
                f"but received {len(decoded)}."
            )
        return decoded

    def _maybe_report_cost(
        self, body: dict, request: GenerationRequest | None = None
    ) -> None:
        usage = body.get("usage") or {}
        cost_usd = usage.get("cost")
        if cost_usd is None:
            return
        try:
            cost_float = float(cost_usd)
        except (TypeError, ValueError):
            return
        if self._reported_cost == cost_float:
            return
        self._reported_cost = cost_float
        print(f"\n[OpenRouter] reported cost: ${cost_float:.4f}")
        # Reconcile with the wizard's pre-call estimate (issue #3). The
        # estimate may come from live /endpoints pricing or the hardcoded
        # COSTS row, so a >10% divergence isn't automatically "COSTS
        # drifted" — it can also be un-modelled billing (input-image
        # charges, per-tier surcharges). Either way it's worth surfacing.
        estimate = getattr(request, "estimated_cost", None) if request else None
        if estimate:
            diff = abs(cost_float - estimate) / estimate
            if diff > 0.10:
                print(
                    f"[OpenRouter] Estimate vs reported cost differ by "
                    f"{diff * 100:.0f}% (estimated ${estimate:.3f}, "
                    f"reported ${cost_float:.4f})."
                )

    # ------------------------------------------------------------------ utils

    def _img_to_data_url(self, img_path: str, model: str) -> str:
        """Reads an image file, compresses if needed to stay under 4.5MB.

        The four-step fallback (untouched → JPEG/PNG resave → scale resize →
        quality reduce) is identical to the legacy adapter: it stays here
        unchanged so existing input sizes still work the same way with the
        new endpoint.
        """
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        ext = os.path.splitext(img_path)[1].lower()
        mime = mime_types.get(ext, "image/png")

        # Black Forest Labs models resize to ~4MP anyway — pre-downscale to
        # avoid sending bandwidth we know will be discarded.
        use_resized = False
        resized_data = None
        if model.startswith("black-forest-labs/"):
            with Image.open(img_path) as img:
                mp = (img.width * img.height) / 1_000_000
                if mp > _MAX_BFL_MEGAPIXELS:
                    print(
                        f"Image {os.path.basename(img_path)} is {mp:.1f}MP, "
                        f"downscaling to {_MAX_BFL_MEGAPIXELS}MP for {model}..."
                    )
                    scale = (_MAX_BFL_MEGAPIXELS / mp) ** 0.5
                    new_width = int(img.width * scale)
                    new_height = int(img.height * scale)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    output = io.BytesIO()
                    img.save(output, format="JPEG", quality=95)
                    resized_data = output.getvalue()
                    use_resized = True
                    mime = "image/jpeg"

        original_data = resized_data if use_resized else open(img_path, "rb").read()

        if len(original_data) <= _MAX_REQUEST_SIZE:
            print(
                f"Image {os.path.basename(img_path)} is "
                f"{len(original_data)/1024:.1f}KB, within 4.5MB limit."
            )
            b64 = base64.b64encode(original_data).decode()
            return f"data:{mime};base64,{b64}"

        # Compression fallback chain (kept verbatim from the legacy code).
        print(
            f"Image {os.path.basename(img_path)} is "
            f"{len(original_data)/1024:.1f}KB, exceeding 4.5MB limit. "
            f"Compressing..."
        )
        with Image.open(img_path) as img:
            out_format = img.format if img.format else "JPEG"
            if out_format == "JPEG" and img.mode != "RGB":
                img = img.convert("RGB")
            output = io.BytesIO()
            quality = 85
            img.save(output, format=out_format, quality=quality)
            compressed_data = output.getvalue()

            if len(compressed_data) > _MAX_REQUEST_SIZE:
                print("Compression not enough, resizing...")
                scale_factor = (_MAX_REQUEST_SIZE / len(compressed_data)) ** 0.5
                new_width = int(img.width * scale_factor * 0.9)
                new_height = int(img.height * scale_factor * 0.9)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                img.save(output, format=out_format, quality=quality)
                compressed_data = output.getvalue()

            if len(compressed_data) > _MAX_REQUEST_SIZE:
                print("Resizing not enough, reducing quality...")
                output = io.BytesIO()
                img.save(output, format=out_format, quality=50)
                compressed_data = output.getvalue()

            b64 = base64.b64encode(compressed_data).decode()
            print(f"Compressed image to {len(compressed_data)/1024:.1f}KB")
            return f"data:{mime};base64,{b64}"
