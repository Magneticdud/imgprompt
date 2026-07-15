# ⚠️  UNTESTED: This provider is not actively exercised in CI or in everyday
# use. Behaviour may drift from Google's API without notice — Google retiring
# `gemini-2.5-flash-image` (shutdown 2 Oct 2026) was the trigger for marking
# this whole module as untrusted-by-default, since we no longer have direct
# Google AI Studio / Vertex AI access to keep it honest. Prefer the OpenRouter
# route for Gemini models unless a feature only this path exposes is needed.
# A one-shot warning is printed the first time `run()` is invoked so anyone
# still pointing GOOGLE_API_KEY here sees it.
import os
import sys
import io
import time

from PIL import Image
from google import genai
from google.genai import types

from imgprompt.providers.base import ImageProvider, GenerationRequest
from imgprompt.images import save_image_bytes

_RETRYABLE_ERRORS = ("503", "500", "UNAVAILABLE", "INTERNAL")
_MAX_RETRIES_BATCH = 10
_MAX_RETRIES_SINGLE = 5
_BASE_DELAY = 2  # seconds
_MAX_DELAY = 60  # seconds

# Class-level latch: prints the UNTESTED warning exactly once per process,
# regardless of how many provider instances or wizard runs happen. Resetting
# is intentional — we'd rather over-warn than under-warn for an unmaintained
# code path.
_UNTESTED_WARNING_EMITTED = False

# Header shown on the first run() call to surface the UNTESTED status.
# Kept as a module-level constant so it's grep-able and easy to update.
_UNTESTED_NOTICE = (
    "\n⚠️  Google direct API provider is currently UNTESTED.\n"
    "    Gemini models are best handled via OpenRouter, which is the\n"
    "    only path verified end-to-end in this project.\n"
)


def _is_retryable(error_str: str) -> bool:
    return (
        any(e in error_str for e in _RETRYABLE_ERRORS)
        or "high demand" in error_str.lower()
    )


def _retry_delay(attempt: int) -> float:
    return min(_BASE_DELAY * (2 ** (attempt - 1)), _MAX_DELAY)


def _build_config(request: GenerationRequest) -> dict:
    config_args = {}
    if request.aspect_ratio != "Auto":
        config_args["aspect_ratio"] = request.aspect_ratio
    if request.model in (
        "gemini-3-pro-image",
        "gemini-3.1-flash-image",
    ):
        # Note: gemini-3.1-flash-lite-image is 1K-only and explicitly does
        # NOT accept `image_size`; passing it returns a 400. That model falls
        # through to the else and stays at the API's default (which is 1K).
        config_args["image_size"] = request.quality_key
    return config_args


def _build_config_kwargs(model: str, config_args: dict) -> dict:
    kwargs = {"image_config": types.ImageConfig(**config_args)}
    if "gemini-3.1" in model:
        # Applies to both gemini-3.1-flash-image and gemini-3.1-flash-lite-image
        # (matches by substring). Lite is documented as image-only and accepts
        # these kwargs; if it ever stops accepting `thinking_config` we'll
        # need to branch on the full model ID.
        kwargs["response_modalities"] = ["IMAGE"]
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="MINIMAL")
    return kwargs


class GoogleProvider(ImageProvider):
    @classmethod
    def provider_name(cls) -> str:
        return "Google"

    # Order matters: `cls.supported_models()[0]` is the wizard default. We
    # intentionally keep non-Lite Flash as the default (it covers more use
    # cases — 1K/2K/4K, multi-image editing, deeper reasoning) and put Lite
    # last as the budget option. Picking Lite by default would silently
    # change behaviour for users who never move the model selector.
    @classmethod
    def supported_models(cls) -> list[str]:
        return [
            "gemini-3.1-flash-image",
            "gemini-3-pro-image",
            "gemini-3.1-flash-lite-image",
        ]

    _STANDARD_RATIOS = [
        "1:1",
        "2:3",
        "3:2",
        "3:4",
        "4:3",
        "4:5",
        "5:4",
        "9:16",
        "16:9",
        "21:9",
    ]

    def get_resolution_choices(
        self, model: str, image_path: str | None
    ) -> tuple[list[str], str]:
        from imgprompt.presets import GEMINI_RESOLUTIONS

        if model == "gemini-3.1-flash-image":
            ratio_options = ["Auto"] + list(GEMINI_RESOLUTIONS.keys())
            default = "Auto"
        elif model == "gemini-3.1-flash-lite-image":
            # Same 14-ratio set as the non-Lite Flash: Gemini 3.x documents
            # these for the whole family, even though Google's published
            # examples only show 10. TODO: extend 3-pro to expose the same
            # 14-ratio set once we have verified OpenRouter accepts the four
            # extreme ratios (1:4, 4:1, 1:8, 8:1) for that model.
            ratio_options = ["Auto"] + list(GEMINI_RESOLUTIONS.keys())
            default = "Auto"
        elif model == "gemini-3-pro-image":
            ratio_options = ["Auto"] + self._STANDARD_RATIOS
            default = "Auto"
        else:
            ratio_options = self._STANDARD_RATIOS
            default = "1:1"
        if image_path:
            from imgprompt.images import get_closest_aspect_ratio

            default = get_closest_aspect_ratio(image_path, ratio_options)
        return ratio_options, default

    def resolve_resolution(
        self, model: str, selection: str
    ) -> tuple[str, int | None, int | None]:
        from imgprompt.presets import GEMINI_RESOLUTIONS

        return GEMINI_RESOLUTIONS.get(selection, "Auto"), None, None

    def get_quality_choices(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        image_path: str | None,
    ) -> tuple[list[str], str]:
        from imgprompt.presets import COSTS

        # Lite is documented as 1K-only — the API rejects 2K/4K. Be explicit
        # rather than letting Lite silently inherit the else branch, otherwise
        # a future addition here would have to remember the exception.
        if model in ("gemini-3-pro-image", "gemini-3.1-flash-image"):
            sizes = ["1K", "2K", "4K"]
        elif model == "gemini-3.1-flash-lite-image":
            sizes = ["1K"]
        else:
            raise NotImplementedError(
                f"Unknown Google provider model: {model!r}. Add an explicit "
                "sizes entry to GoogleProvider.get_quality_choices."
            )
        # .3f so cents-precise prices (e.g. Lite at $0.034) don't render as
        # a misleading $0.03. Trailing zero on $0.07 → $0.070 is fine and
        # matches what OpenRouter hands back on the usage report.
        choices = [f"{s} (${COSTS[model][s]['fixed']:.3f})" for s in sizes]
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

    def run(self, request: GenerationRequest) -> None:
        # One-shot UNTESTED banner. Module-level latch so a second Google
        # run in the same process doesn't repeat itself.
        global _UNTESTED_WARNING_EMITTED
        if not _UNTESTED_WARNING_EMITTED:
            print(_UNTESTED_NOTICE)
            _UNTESTED_WARNING_EMITTED = True

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
            sys.exit(1)

        self._client = genai.Client(api_key=api_key)
        config_args = _build_config(request)

        if request.is_batch:
            self._run_batch(request, config_args)
        else:
            self._run_single(request, config_args)

    def _call_api(
        self, request: GenerationRequest, config_args: dict, img_paths: list[str]
    ) -> bool:
        """Makes a single API call, saves the result, returns True on success."""
        config_kwargs = _build_config_kwargs(request.model, config_args)
        req_contents = [request.prompt]
        img_contexts = []

        if img_paths:
            if len(img_paths) == 1:
                img = Image.open(img_paths[0])
                img_contexts.append(img)
                req_contents.append(img)
            else:
                # Dual mode
                req_contents.append("IMG_1:")
                img1 = Image.open(img_paths[0])
                img_contexts.append(img1)
                req_contents.append(img1)
                req_contents.append("IMG_2:")
                img2 = Image.open(img_paths[1])
                img_contexts.append(img2)
                req_contents.append(img2)

        try:
            response = self._client.models.generate_content(
                model=request.model,
                contents=req_contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        finally:
            for img in img_contexts:
                img.close()

        original_path = img_paths[0] if img_paths else None

        if hasattr(response, "parts") and response.parts:
            for part in response.parts:
                if part.inline_data is not None:
                    buf = io.BytesIO()
                    part.as_image().save(buf, format="PNG")
                    save_image_bytes(buf.getvalue(), original_path)
                    return True
                elif part.text:
                    print(f"Response text: {part.text}")

        # No image found — report why
        print("Error: No image found in Google API response.")
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            if hasattr(response.prompt_feedback, "block_reason"):
                print(
                    f"Prompt Feedback Block Reason: {response.prompt_feedback.block_reason}"
                )
        if hasattr(response, "candidates") and response.candidates:
            for i, candidate in enumerate(response.candidates):
                if hasattr(candidate, "finish_reason"):
                    reason_str = str(candidate.finish_reason)
                    print(f"Candidate {i+1} Finish Reason: {candidate.finish_reason}")
                    if any(x in reason_str for x in ("SAFETY", "BLOCK", "OTHER")):
                        print(
                            ">> The request was likely blocked due to safety settings or policy violations."
                        )
        return False

    def _run_with_retry(
        self,
        request: GenerationRequest,
        config_args: dict,
        img_paths: list[str],
        max_retries: int,
    ) -> bool:
        for attempt in range(1, max_retries + 1):
            try:
                return self._call_api(request, config_args, img_paths)
            except Exception as e:
                if _is_retryable(str(e)):
                    if attempt < max_retries:
                        delay = _retry_delay(attempt)
                        print(
                            f"\n[Retry {attempt}/{max_retries}] Server error, waiting {delay:.0f}s..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        print(f"\nMax retries reached. Last error: {e}")
                else:
                    print(f"\nAn error occurred during Google call: {e}")
                return False
        return False

    def _run_batch(self, request: GenerationRequest, config_args: dict) -> None:
        print(f"\nStarting batch processing: {len(request.images)} images...")
        success_count = 0
        fail_count = 0
        # Basenames of inputs that produced no image, so a skipped PDF page can
        # be named in the summary. A failed input never aborts the batch.
        failed_inputs: list[str] = []

        for idx, img_path in enumerate(request.images, 1):
            print(
                f"\n--- Processing image {idx}/{len(request.images)}: {os.path.basename(img_path)} ---"
            )
            ok = self._run_with_retry(
                request, config_args, [img_path], _MAX_RETRIES_BATCH
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1
                failed_inputs.append(os.path.basename(img_path))

        print(
            f"\n=== Batch complete: {success_count} succeeded, {fail_count} failed ==="
        )
        if failed_inputs:
            print(
                f">> Skipped/failed ({len(failed_inputs)}): "
                + ", ".join(failed_inputs)
            )

    def _run_single(self, request: GenerationRequest, config_args: dict) -> None:
        print(f"\nSending request to Google ({request.model})...")
        self._run_with_retry(request, config_args, request.images, _MAX_RETRIES_SINGLE)
