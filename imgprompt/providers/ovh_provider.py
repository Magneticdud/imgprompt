import os

import requests as http_requests

from imgprompt.providers.base import ImageProvider, GenerationRequest
from imgprompt.images import save_api_image

OVH_ENDPOINT = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/images/generations"


class OVHProvider(ImageProvider):
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
