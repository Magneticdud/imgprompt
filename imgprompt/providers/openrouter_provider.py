import os
import sys
import io
import base64
import math
from datetime import datetime

import requests
from PIL import Image

from imgprompt.providers.base import ImageProvider, GenerationRequest
from imgprompt.providers.capabilities import get_capabilities
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
_MODEL_MAX_N_PREFIXES = {"recraft/": 6}


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

# Some upstream providers enforce a hard minimum on the OUTPUT pixel count
# and 400 anything below it (Seed: "image size must be at least 3686400
# pixels"). OpenRouter's own (aspect_ratio, resolution) → size translation
# lands below that floor for most non-square ratios, so for these models we
# resolve the size client-side and send it explicitly. See issue #10.
_MODEL_PIXEL_FLOORS = {
    "bytedance-seed/seedream-4.5": 3_686_400,
}

# Nominal pixel targets per resolution tier (the square each tier names).
# Used to derive an explicit `size` for models in _MODEL_PIXEL_FLOORS.
_TIER_PIXELS = {
    "512": 512 * 512,
    "1K": 1024 * 1024,
    "2K": 2048 * 2048,
    "4K": 4096 * 4096,
}


def _ceil16(value: float) -> int:
    """Round up to a multiple of 16 (dimension granularity of /api/v1/images)."""
    return math.ceil(value / 16) * 16


def _looks_like_svg(data: bytes) -> bool:
    """Sniff an SVG document from its head (used when media_type is absent)."""
    head = data[:512].lstrip()
    return head.startswith(b"<svg") or (
        head.startswith(b"<?xml") and b"<svg" in head
    )


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
            "bytedance-seed/seedream-4.5",
            "black-forest-labs/flux.2-klein-4b",
            "black-forest-labs/flux.2-flex",
            "black-forest-labs/flux.2-pro",
            "black-forest-labs/flux.2-max",
            "sourceful/riverflow-v2.5-fast",
            "sourceful/riverflow-v2.5-pro",
            "google/gemini-3.1-flash-image",
            "google/gemini-3-pro-image",
            "google/gemini-3.1-flash-lite-image",
            "microsoft/mai-image-2.5",
            "x-ai/grok-imagine-image-quality",
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
        if model in (
            "google/gemini-3.1-flash-image",
            "google/gemini-3.1-flash-lite-image",
        ):
            ratio_options = list(OPENROUTER_RESOLUTIONS.keys())
        elif model == "microsoft/mai-image-2.5":
            # MAI's /api/v1/images descriptor advertises exactly these seven
            # concrete ratios (plus "auto", which the wizard doesn't surface
            # for OpenRouter): no 4:5/5:4/21:9. Verified 2026-07-07 against
            # /api/v1/images/models — sending anything else 400s upstream.
            ratio_options = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"]
        elif model == "x-ai/grok-imagine-image-quality":
            # Grok's descriptor (verified 2026-07-07) lists these seven plus
            # phone-screen ratios (9:19.5, 19.5:9, 9:20, 20:9, 1:2, 2:1) and
            # "auto". The phone ratios have no RATIO_TO_RESOLUTION entry for
            # the wizard's pixel preview, so we keep them off the picker; no
            # 4:5/5:4/21:9 upstream.
            ratio_options = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"]
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
        if model.startswith("openai/gpt-"):
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
        elif model == "microsoft/mai-image-2.5":
            # MAI exposes NO `resolution` parameter on /api/v1/images
            # (descriptor verified 2026-07-07): the model picks the output
            # size from the aspect ratio alone. "Standard" is deliberately
            # outside the {512,1K,2K,4K} set so _build_payload never emits
            # a resolution field for it.
            sizes = ["Standard"]
        elif model == "x-ai/grok-imagine-image-quality":
            # Grok Imagine caps at 2K — descriptor lists exactly ["1K","2K"]
            # (verified 2026-07-07 against /api/v1/images/models). Explicit
            # branch (same values as the generic fallback) so the cap is
            # documented rather than accidental.
            sizes = ["1K", "2K"]
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
        caps = get_capabilities(model)
        if caps and caps.resolutions:
            from imgprompt.presets import COSTS as _costs

            descriptor_sizes = [
                s
                for s in ("512", "1K", "2K", "4K")
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
            label = f"{s} (${COSTS[model][s]['fixed']:.3f})"
            # Pre-flight signal for tiers the upstream floor overrides (e.g.
            # seedream 1K): the user should know before confirming that the
            # output will be larger than the tier name suggests.
            if floor and _TIER_PIXELS.get(s, floor) < floor:
                label += f" — raised to upstream {floor / 1e6:.1f}MP minimum"
            choices.append(label)
        return choices, choices[0]

    def resolve_quality(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        selection: str,
    ) -> tuple[str, float]:
        from imgprompt.presets import COSTS

        quality_key = selection.split(" ")[0]
        return quality_key, COSTS[model][quality_key]["fixed"]

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
        if (
            caps.resolutions
            and quality_key in ("512", "1K", "2K", "4K")
            and quality_key not in caps.resolutions
        ):
            warnings.append(
                f"{model} does not advertise resolution {quality_key} "
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
                continue
            except Exception as e:
                print(f"An error occurred during OpenRouter call: {e}")
                failed += 1
                continue

            if not results:
                failed += 1
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

        floor = _MODEL_PIXEL_FLOORS.get(request.model)
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
            payload["size"] = f"{width}x{height}"
        else:
            explicit = floor and self._floor_size(
                request.model, request.aspect_ratio, request.quality_key
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
                if request.quality_key in ("512", "1K", "2K", "4K"):
                    payload["resolution"] = request.quality_key

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
        """Explicit WxH for models with an upstream output-pixel floor.

        Targets the tier's nominal pixel count (1K/2K/4K square), raised to
        the model's floor when the tier sits below it, then shapes the box
        to the chosen aspect ratio with both edges rounded UP to a multiple
        of 16 so the result can never dip back under the floor. Returns
        None when the ratio is unknown — the caller then falls back to the
        plain aspect_ratio+resolution fields and lets upstream decide.
        """
        from imgprompt.presets import ASPECT_RATIO_VALUES

        floor = _MODEL_PIXEL_FLOORS.get(model)
        ratio = ASPECT_RATIO_VALUES.get(aspect_ratio)
        if floor is None or ratio is None:
            return None
        target = max(_TIER_PIXELS.get(quality_key, floor), floor)
        height = _ceil16((target / ratio) ** 0.5)
        width = _ceil16(height * ratio)
        if target == floor and _TIER_PIXELS.get(quality_key, floor) < floor:
            print(
                f"[OpenRouter] {model}: {quality_key} on {aspect_ratio} is "
                f"below the upstream {floor:,}-pixel minimum; sending "
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
            caps.n_max
            if caps and caps.n_max is not None
            else _max_n_for(request.model)
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

        self._maybe_report_cost(body)

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

    def _maybe_report_cost(self, body: dict) -> None:
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
