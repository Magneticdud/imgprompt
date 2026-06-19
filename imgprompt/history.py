"""Persist the last generation request so it can be replayed verbatim.

The wizard in ``imgedit.py`` builds a :class:`GenerationRequest` together with a
provider name. We dump both to a gitignored JSON file so a later ``--replay``
run can reconstruct the exact same parameters and prompt without walking the
wizard again.
"""

import dataclasses
import json
import os

from imgprompt.providers.base import GenerationRequest

LAST_GENERATION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".last_generation.json",
)


def save_last_generation(provider: str, request: GenerationRequest) -> None:
    """Write the provider name and request fields to the gitignored JSON file."""
    data = {"provider": provider, "request": dataclasses.asdict(request)}
    try:
        with open(LAST_GENERATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"Warning: could not save last generation: {e}")


def load_last_generation() -> tuple[str, GenerationRequest] | None:
    """Load the last saved generation. Returns (provider, request) or None."""
    if not os.path.exists(LAST_GENERATION_FILE):
        return None
    try:
        with open(LAST_GENERATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        provider = data["provider"]
        request = GenerationRequest(**data["request"])
        return provider, request
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"Warning: could not load last generation: {e}")
        return None
