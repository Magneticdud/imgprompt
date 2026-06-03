import os
import sys

from openai import OpenAI

from imgprompt.providers.base import ImageProvider, GenerationRequest
from imgprompt.images import process_image_for_api, save_api_image


class OpenAIProvider(ImageProvider):
    @classmethod
    def provider_name(cls) -> str:
        return "OpenAI"

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["gpt-image-2"]

    def get_resolution_choices(
        self, model: str, image_path: str | None
    ) -> tuple[list[str], str]:
        from imgprompt.presets import (
            GPT_IMAGE_2_PRESET_CHOICES,
            CUSTOM_DIMS,
            GPT_IMAGE_2_AUTO,
        )

        if model == "gpt-image-2":
            labels = [label for label, _, _, _, _ in GPT_IMAGE_2_PRESET_CHOICES]
            if image_path:
                default = GPT_IMAGE_2_AUTO
            else:
                default = labels[0]
            return [GPT_IMAGE_2_AUTO] + labels + [CUSTOM_DIMS], default
        else:
            res_options = [
                "1024x1024 (Square)",
                "1024x1536 (Vertical)",
                "1536x1024 (Horizontal)",
            ]
            if image_path:
                from imgprompt.images import get_closest_aspect_ratio

                default = get_closest_aspect_ratio(image_path, res_options)
            else:
                default = res_options[0]
            return res_options, default

    def resolve_resolution(
        self, model: str, selection: str
    ) -> tuple[str, int | None, int | None]:
        from imgprompt.presets import GPT_IMAGE_2_PRESET_CHOICES, GPT_IMAGE_2_AUTO

        if model == "gpt-image-2":
            if selection == GPT_IMAGE_2_AUTO:
                return "auto", None, None
            for label, _, _, w, h in GPT_IMAGE_2_PRESET_CHOICES:
                if label == selection:
                    return f"{w}x{h}", w, h
            return "1024x1024", 1024, 1024
        else:
            return selection.split(" ")[0], None, None

    def get_quality_choices(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        image_path: str | None,
    ) -> tuple[list[str], str]:
        from imgprompt.presets import (
            COSTS,
            calc_gpt_image2_tokens,
            GPT_IMAGE_2_PRICE_PER_MTOK,
        )

        if model == "gpt-image-2":
            choices = []
            for q in ["Low", "Medium", "High"]:
                if width is None or height is None:
                    choices.append(f"{q} (cost depends on output size)")
                else:
                    tokens = calc_gpt_image2_tokens(width, height, q)
                    cost = tokens * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000
                    choices.append(f"{q} (~{tokens:,} tokens, ${cost:.4f})")
            return choices, choices[0]
        else:
            choices = [
                f"{q} (${COSTS[model][q][res_key]:.3f})"
                for q in ["Low", "Medium", "High"]
            ]
            return choices, choices[0]

    def resolve_quality(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        selection: str,
    ) -> tuple[str, float]:
        from imgprompt.presets import (
            COSTS,
            calc_gpt_image2_tokens,
            GPT_IMAGE_2_PRICE_PER_MTOK,
        )

        quality_key = selection.split(" ")[0]
        if model == "gpt-image-2":
            if width is None or height is None:
                return quality_key, 0.0
            tokens = calc_gpt_image2_tokens(width, height, quality_key)
            return quality_key, tokens * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000
        else:
            return quality_key, COSTS[model][quality_key][res_key]

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def supports_dual(self) -> bool:
        return False

    def run(self, request: GenerationRequest) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
            sys.exit(1)

        client = OpenAI(api_key=api_key)

        if request.is_batch:
            self._run_batch(client, request)
        else:
            self._run_single(client, request)

    def _run_batch(self, client: OpenAI, request: GenerationRequest) -> None:
        print(f"\nStarting batch processing: {len(request.images)} images...")
        success_count = 0
        fail_count = 0

        for idx, img_path in enumerate(request.images, 1):
            print(
                f"\n--- Processing image {idx}/{len(request.images)}: {os.path.basename(img_path)} ---"
            )
            try:
                image_input = process_image_for_api(img_path, request.res_key)
                if request.model.startswith("gpt-image"):
                    kwargs = dict(
                        model=request.model,
                        image=image_input,
                        prompt=request.prompt,
                        n=1,
                        quality=request.quality_key.lower(),
                    )
                    if request.res_key == "auto":
                        kwargs["size"] = "auto"
                    response = client.images.edit(**kwargs)
                else:
                    # DALL-E 2 uses size parameter
                    response = client.images.edit(
                        model=request.model,
                        image=image_input,
                        prompt=request.prompt,
                        n=1,
                        size=request.res_key,
                        quality=request.quality_key.lower(),
                    )
                image_url = None
                image_b64 = None
                if hasattr(response, "data") and len(response.data) > 0:
                    image_url = getattr(response.data[0], "url", None)
                    image_b64 = getattr(response.data[0], "b64_json", None)

                if image_url or image_b64:
                    save_api_image(image_url, image_b64, img_path)
                    success_count += 1
                else:
                    print("Error: Could not retrieve image data from the API response.")
                    fail_count += 1

            except Exception as e:
                error_msg = str(e)
                if "moderation_blocked" in error_msg:
                    print(">> The request was rejected by the OpenAI safety system.")
                else:
                    print(f"An error occurred during OpenAI call: {e}")
                fail_count += 1

        print(
            f"\n=== Batch complete: {success_count} succeeded, {fail_count} failed ==="
        )

    def _run_single(self, client: OpenAI, request: GenerationRequest) -> None:
        print(f"\nSending request to OpenAI ({request.model})...")
        try:
            if len(request.images) == 1:
                image_input = process_image_for_api(request.images[0], request.res_key)
            else:
                image_input = [
                    process_image_for_api(p, request.res_key) for p in request.images
                ]

            if request.model.startswith("gpt-image"):
                kwargs = dict(
                    model=request.model,
                    image=image_input,
                    prompt=request.prompt,
                    n=1,
                    quality=request.quality_key.lower(),
                )
                if request.res_key == "auto":
                    kwargs["size"] = "auto"
                response = client.images.edit(**kwargs)
            else:
                # DALL-E 2 uses size parameter
                response = client.images.edit(
                    model=request.model,
                    image=image_input,
                    prompt=request.prompt,
                    n=1,
                    size=request.res_key,
                    quality=request.quality_key.lower(),
                )
            image_url = None
            image_b64 = None
            if hasattr(response, "data") and len(response.data) > 0:
                image_url = getattr(response.data[0], "url", None)
                image_b64 = getattr(response.data[0], "b64_json", None)

            if image_url or image_b64:
                save_api_image(image_url, image_b64, request.primary_image)
            else:
                print("\nError: Could not retrieve image data from the API response.")

        except Exception as e:
            error_msg = str(e)
            if "moderation_blocked" in error_msg:
                print("\n>> The request was rejected by the OpenAI safety system.")
                print(
                    ">> This typically happens with images of famous people, children, or NSFW content."
                )
            else:
                print(f"\nAn error occurred during OpenAI call: {e}")
