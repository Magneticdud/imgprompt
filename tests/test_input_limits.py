"""Per-provider input-image limits for combined mode.

OpenAI and Google direct expose no capability endpoint, so their bounds mirror
the equivalent models' OpenRouter descriptors (surveyed 2026-10-04). OVH is
text-to-image only and accepts no input images.
"""

from imgprompt.providers.google_provider import GoogleProvider
from imgprompt.providers.openai_provider import OpenAIProvider
from imgprompt.providers.ovh_provider import OVHProvider


def test_openai_matches_the_openrouter_gpt_image_limit():
    provider = OpenAIProvider()
    for model in OpenAIProvider.supported_models():
        assert provider.max_input_images(model) == 16


def test_google_matches_the_openrouter_gemini_limit():
    provider = GoogleProvider()
    for model in GoogleProvider.supported_models():
        assert provider.max_input_images(model) == 14


def test_ovh_accepts_no_input_images():
    provider = OVHProvider()
    for model in OVHProvider.supported_models():
        assert provider.max_input_images(model) == 0
