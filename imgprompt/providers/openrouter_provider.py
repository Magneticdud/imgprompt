import os
import sys
import io
import base64
from datetime import datetime

from openai import OpenAI
from PIL import Image

from imgprompt.providers.base import ImageProvider, GenerationRequest
from imgprompt.images import get_image_extension

_MAX_REQUEST_SIZE = 4.5 * 1024 * 1024  # 4.5MB
_MAX_BFL_MEGAPIXELS = 4


class OpenRouterProvider(ImageProvider):
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
                    output = io.BytesIO()
                    img.save(output, format="JPEG", quality=95)
                    resized_data = output.getvalue()
                    use_resized = True
                    mime = "image/jpeg"

        original_data = resized_data if use_resized else open(img_path, "rb").read()

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
            output = io.BytesIO()
            quality = 85
            img.save(
                output, format=img.format if img.format else "JPEG", quality=quality
            )
            compressed_data = output.getvalue()

            if len(compressed_data) > _MAX_REQUEST_SIZE:
                print("Compression not enough, resizing...")
                scale_factor = (_MAX_REQUEST_SIZE / len(compressed_data)) ** 0.5
                new_width = int(img.width * scale_factor * 0.9)
                new_height = int(img.height * scale_factor * 0.9)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                img.save(
                    output, format=img.format if img.format else "JPEG", quality=quality
                )
                compressed_data = output.getvalue()

            if len(compressed_data) > _MAX_REQUEST_SIZE:
                print("Resizing not enough, reducing quality...")
                output = io.BytesIO()
                img.save(
                    output, format=img.format if img.format else "JPEG", quality=50
                )
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

    def _save_image(self, data_url: str, original_path: str | None) -> None:
        """Saves a base64 data URL image to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if "," in data_url:
            _, b64_data = data_url.split(",", 1)
        else:
            b64_data = data_url

        img_bytes = base64.b64decode(b64_data)
        ext = get_image_extension(img_bytes)

        if original_path:
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            output_dir = os.path.dirname(original_path) or "."
            output_path = os.path.join(
                output_dir, f"edited_{timestamp}_{base_name}{ext}"
            )
        else:
            output_path = f"generated_{timestamp}{ext}"

        with open(output_path, "wb") as f:
            f.write(img_bytes)
        print(f"\nSuccess! File saved successfully as {output_path}")

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
                    self._save_image(data_url, img_path)
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
                self._save_image(data_url, request.primary_image)
            else:
                print("\nError: No image returned by OpenRouter API.")
        except Exception as e:
            print(f"\nAn error occurred during OpenRouter call: {e}")
