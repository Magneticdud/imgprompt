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
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
    ):
        config_args["image_size"] = request.quality_key
    return config_args


def _build_config_kwargs(model: str, config_args: dict) -> dict:
    kwargs = {"image_config": types.ImageConfig(**config_args)}
    if "gemini-3.1" in model:
        kwargs["response_modalities"] = ["IMAGE"]
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="MINIMAL")
    return kwargs


class GoogleProvider(ImageProvider):
    @classmethod
    def provider_name(cls) -> str:
        return "Google"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [
            "gemini-2.5-flash-image",
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
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

        if model == "gemini-3.1-flash-image-preview":
            ratio_options = ["Auto"] + list(GEMINI_RESOLUTIONS.keys())
            default = "Auto"
        elif model == "gemini-3-pro-image-preview":
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

        if model in ("gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"):
            sizes = ["1K", "2K", "4K"]
        else:
            sizes = ["1K"]
        choices = [f"{s} (${COSTS[model][s]['fixed']:.2f})" for s in sizes]
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

        print(
            f"\n=== Batch complete: {success_count} succeeded, {fail_count} failed ==="
        )

    def _run_single(self, request: GenerationRequest, config_args: dict) -> None:
        print(f"\nSending request to Google ({request.model})...")
        self._run_with_retry(request, config_args, request.images, _MAX_RETRIES_SINGLE)
