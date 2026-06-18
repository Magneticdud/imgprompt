import os
import sys
import io
import base64

from openai import OpenAI
from PIL import Image

from imgprompt.providers.base import ImageProvider, GenerationRequest
from imgprompt.images import save_image_bytes, process_image_for_api, save_api_image

_MAX_REQUEST_SIZE = 4.5 * 1024 * 1024  # 4.5MB
_MAX_BFL_MEGAPIXELS = 4


class OpenRouterProvider(ImageProvider):
    @classmethod
    def provider_name(cls) -> str:
        return "OpenRouter"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [
            "openai/gpt-5.4-image-2",
            "bytedance-seed/seedream-4.5",
            "black-forest-labs/flux.2-klein-4b",
            "black-forest-labs/flux.2-flex",
            "black-forest-labs/flux.2-pro",
            "black-forest-labs/flux.2-max",
            "sourceful/riverflow-v2-fast",
            "sourceful/riverflow-v2-pro",
            "google/gemini-2.5-flash-image",
            "google/gemini-3.1-flash-image-preview",
            "google/gemini-3-pro-image-preview",
        ]

    def get_resolution_choices(
        self, model: str, image_path: str | None
    ) -> tuple[list[str], str]:
        from imgprompt.presets import (
            GPT_IMAGE_2_PRESET_CHOICES,
            CUSTOM_DIMS,
            OPENROUTER_RESOLUTIONS,
            OPENROUTER_STANDARD_RATIOS,
        )

        if model.startswith("openai/gpt-"):
            labels = [label for label, _, _, _, _ in GPT_IMAGE_2_PRESET_CHOICES]
            if image_path:
                with Image.open(image_path) as img:
                    img_ratio = img.width / img.height
                best_idx, best_diff = 0, float("inf")
                for i, (_, ratio_str, _, w, h) in enumerate(GPT_IMAGE_2_PRESET_CHOICES):
                    p, q = map(int, ratio_str.split(":"))
                    diff = abs(img_ratio - p / q)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = i
                default = labels[best_idx]
            else:
                default = labels[0]
            return labels + [CUSTOM_DIMS], default
        if model == "google/gemini-3.1-flash-image-preview":
            ratio_options = list(OPENROUTER_RESOLUTIONS.keys())
        else:
            ratio_options = OPENROUTER_STANDARD_RATIOS + ["21:9"]
        default = "1:1"
        if image_path:
            from imgprompt.images import get_closest_aspect_ratio

            default = get_closest_aspect_ratio(image_path, ratio_options)
        return ratio_options, default

    def resolve_resolution(
        self, model: str, selection: str
    ) -> tuple[str, int | None, int | None]:
        from imgprompt.presets import GPT_IMAGE_2_PRESET_CHOICES, OPENROUTER_RESOLUTIONS

        if model.startswith("openai/gpt-"):
            for label, _, _, w, h in GPT_IMAGE_2_PRESET_CHOICES:
                if label == selection:
                    return f"{w}x{h}", w, h
            return "1024x1024", 1024, 1024
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

        if model.startswith("openai/gpt-"):
            choices = [
                f"{q} (${COSTS[model][q]['fixed']:.2f})"
                for q in ["Low", "Medium", "High"]
            ]
            return choices, choices[1]
        if model == "sourceful/riverflow-v2-pro":
            sizes = ["1K", "2K", "4K"]
        elif model.startswith("black-forest-labs/"):
            sizes = ["1K", "2K"]
        elif model in (
            "google/gemini-3-pro-image-preview",
            "google/gemini-3.1-flash-image-preview",
        ):
            sizes = ["1K", "2K", "4K"]
        elif model == "google/gemini-2.5-flash-image":
            sizes = ["1K"]
        else:
            sizes = ["1K", "2K"]
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
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print(
                "Error: OPENROUTER_API_KEY not found. Please set it in your .env file."
            )
            sys.exit(1)

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Check if this is an OpenAI model that uses the images endpoint
        if request.model.startswith("openai/gpt-"):
            if request.is_batch:
                self._run_batch_openai(request)
            else:
                self._run_single_openai(request)
        else:
            if request.is_batch:
                self._run_batch(request)
            else:
                self._run_single(request)

    def _img_to_data_url(self, img_path: str, model: str) -> str:
        """Reads an image file, compresses if needed to stay under 4.5MB, returns a base64 data URL."""
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        ext = os.path.splitext(img_path)[1].lower()
        mime = mime_types.get(ext, "image/png")

        # Downscale large images for black-forest-labs models (they resize to 4MP anyway)
        use_resized = False
        resized_data = None
        if model.startswith("black-forest-labs/"):
            with Image.open(img_path) as img:
                mp = (img.width * img.height) / 1_000_000
                if mp > _MAX_BFL_MEGAPIXELS:
                    print(
                        f"Image {os.path.basename(img_path)} is {mp:.1f}MP, downscaling to {_MAX_BFL_MEGAPIXELS}MP for {model}..."
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

        if use_resized:
            original_data = resized_data
        else:
            with open(img_path, "rb") as f:
                original_data = f.read()

        if len(original_data) <= _MAX_REQUEST_SIZE:
            print(
                f"Image {os.path.basename(img_path)} is {len(original_data)/1024:.1f}KB, within 4.5MB limit."
            )
            b64 = base64.b64encode(original_data).decode()
            return f"data:{mime};base64,{b64}"

        # If too large, compress/resize
        print(
            f"Image {os.path.basename(img_path)} is {len(original_data)/1024:.1f}KB, exceeding 4.5MB limit. Compressing..."
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

    def _call_api(
        self, request: GenerationRequest, img_paths: list[str] | None = None
    ) -> str | None:
        """Calls OpenRouter and returns a base64 data URL or None."""
        image_config = {"aspect_ratio": request.aspect_ratio}
        if request.quality_key in ["2K", "4K"]:
            image_config["image_size"] = request.quality_key

        if img_paths:
            content = []
            for p in img_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._img_to_data_url(p, request.model)},
                    }
                )
            content.append({"type": "text", "text": request.prompt})
        else:
            content = request.prompt

        resp = self._client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": content}],
            extra_body={
                "modalities": ["image"],
                "image_config": image_config,
            },
        )

        if not resp.choices:
            print("\nError: No choices returned in API response")
            print(f"Debug - Full response: {resp}")
            return None

        msg = resp.choices[0].message
        images = getattr(msg, "images", None)
        if images:
            return images[0]["image_url"]["url"]
        return None

    def _run_batch(self, request: GenerationRequest) -> None:
        print(f"\nStarting batch processing: {len(request.images)} images...")
        success_count = 0
        fail_count = 0

        for idx, img_path in enumerate(request.images, 1):
            print(
                f"\n--- Processing image {idx}/{len(request.images)}: {os.path.basename(img_path)} ---"
            )
            try:
                data_url = self._call_api(request, img_paths=[img_path])
                if data_url:
                    b64_data = (
                        data_url.split(",", 1)[1] if "," in data_url else data_url
                    )
                    save_image_bytes(base64.b64decode(b64_data), img_path)
                    success_count += 1
                else:
                    print("Error: No image returned by OpenRouter API.")
                    fail_count += 1
            except Exception as e:
                print(f"An error occurred during OpenRouter call: {e}")
                fail_count += 1

        print(
            f"\n=== Batch complete: {success_count} succeeded, {fail_count} failed ==="
        )

    def _run_single(self, request: GenerationRequest) -> None:
        print(f"\nSending request to OpenRouter ({request.model})...")
        try:
            img_paths = request.images if request.images else None
            data_url = self._call_api(request, img_paths=img_paths)
            if data_url:
                b64_data = data_url.split(",", 1)[1] if "," in data_url else data_url
                save_image_bytes(base64.b64decode(b64_data), request.primary_image)
            else:
                print("\nError: No image returned by OpenRouter API.")
        except Exception as e:
            print(f"\nAn error occurred during OpenRouter call: {e}")

    def _run_batch_openai(self, request: GenerationRequest) -> None:
        """Batch processing for OpenAI models using images.edit endpoint."""
        print(f"\nStarting batch processing: {len(request.images)} images...")
        success_count = 0
        fail_count = 0

        for idx, img_path in enumerate(request.images, 1):
            print(
                f"\n--- Processing image {idx}/{len(request.images)}: {os.path.basename(img_path)} ---"
            )
            try:
                image_input = process_image_for_api(img_path, request.res_key)

                # OpenRouter doesn't support width/height parameters
                response = self._client.images.edit(
                    model=request.model,
                    image=image_input,
                    prompt=request.prompt,
                    n=1,
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
                print(f"An error occurred during OpenRouter call: {e}")
                fail_count += 1

        print(
            f"\n=== Batch complete: {success_count} succeeded, {fail_count} failed ==="
        )

    def _run_single_openai(self, request: GenerationRequest) -> None:
        """Single image processing for OpenAI models using images.edit endpoint."""
        print(f"\nSending request to OpenRouter ({request.model})...")
        try:
            if len(request.images) == 1:
                image_input = process_image_for_api(request.images[0], request.res_key)
            else:
                # For text-to-image, we need to use images.generate instead
                if not request.images:
                    response = self._client.images.generate(
                        model=request.model,
                        prompt=request.prompt,
                        n=1,
                        quality=request.quality_key.lower(),
                        size=request.res_key,
                    )

                    image_url = None
                    image_b64 = None
                    if hasattr(response, "data") and len(response.data) > 0:
                        image_url = getattr(response.data[0], "url", None)
                        image_b64 = getattr(response.data[0], "b64_json", None)

                    if image_url or image_b64:
                        save_api_image(image_url, image_b64, None)
                    else:
                        print(
                            "\nError: Could not retrieve image data from the API response."
                        )
                    return

                image_input = [
                    process_image_for_api(p, request.res_key) for p in request.images
                ]

            # OpenRouter doesn't support width/height parameters
            response = self._client.images.edit(
                model=request.model,
                image=image_input,
                prompt=request.prompt,
                n=1,
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
            print(f"\nAn error occurred during OpenRouter call: {e}")
