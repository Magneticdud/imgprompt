import os

import requests as http_requests

from imgprompt.providers.base import ImageProvider, GenerationRequest
from imgprompt.images import save_api_image

OVH_ENDPOINT = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/images/generations"


class OVHProvider(ImageProvider):
    @classmethod
    def provider_name(cls) -> str:
        return "OVH"

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["stabilityai/stable-diffusion-xl-base-1.0"]

    def get_resolution_choices(
        self, model: str, image_path: str | None
    ) -> tuple[list[str], str]:
        return ["1:1"], "1:1"

    def resolve_resolution(
        self, model: str, selection: str
    ) -> tuple[str, int | None, int | None]:
        return "1024x1024", None, None

    def get_quality_choices(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        image_path: str | None,
    ) -> tuple[list[str], str]:
        return ["1K (Free)"], "1K (Free)"

    def resolve_quality(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        selection: str,
    ) -> tuple[str, float]:
        return "1K", 0.0

    @property
    def supports_batch(self) -> bool:
        return False

    @property
    def supports_dual(self) -> bool:
        return False

    def run(self, request: GenerationRequest) -> None:
        api_key = os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        is_placeholder = api_key and api_key.startswith("your_")
        if api_key and not is_placeholder:
            headers["Authorization"] = f"Bearer {api_key}"
            print("Using authenticated OVH access (400 rpm).")
        else:
            print("Using anonymous OVH access (2 rpm).")

        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "size": request.res_key,
            "response_format": "b64_json",
        }

        print(f"\nSending request to OVH ({request.model})...")
        try:
            response = http_requests.post(OVH_ENDPOINT, headers=headers, json=payload)
            if response.status_code == 200:
                resp_json = response.json()
                image_b64 = None
                if "data" in resp_json and len(resp_json["data"]) > 0:
                    image_b64 = resp_json["data"][0].get("b64_json")
                if image_b64:
                    save_api_image(None, image_b64, None)
                else:
                    print(
                        "\nError: Could not retrieve image data from the OVH API response."
                    )
            else:
                print(f"Error: {response.status_code} {response.text}")
        except Exception as e:
            print(f"\nAn error occurred during OVH call: {e}")
