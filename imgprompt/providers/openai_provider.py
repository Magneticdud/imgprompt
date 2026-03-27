import os
import sys

from openai import OpenAI

from imgprompt.providers.base import ImageProvider, GenerationRequest
from imgprompt.images import process_image_for_api, save_api_image


class OpenAIProvider(ImageProvider):
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
