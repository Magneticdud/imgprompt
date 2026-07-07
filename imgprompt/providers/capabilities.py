"""Typed view over OpenRouter's /api/v1/images/models capability descriptors.

The endpoint returns, per model, a `supported_parameters` block with typed
descriptors (enum / range / boolean) for aspect_ratio, resolution, n,
input_references, seed, etc. This module fetches the catalog once per
process, parses it into ModelCapabilities, and hands it to the OpenRouter
provider so wizard choices and payload clamps track upstream truth instead
of hardcoded branches (issue #11).

Failure policy: any network or parse problem downgrades to "no descriptor"
— callers keep their hardcoded fallback behaviour and the CLI prints a
one-line note. A stale in-session cache is preferred over re-fetching.
"""

import os
from dataclasses import dataclass

import requests

OPENROUTER_IMAGE_MODELS_URL = "https://openrouter.ai/api/v1/images/models"

# Tight budget: discovery must never make the wizard feel slow. On a miss
# we fall back to the hardcoded tables, which is exactly what the CLI did
# before this module existed.
_FETCH_TIMEOUT = (2.0, 2.0)


@dataclass(frozen=True)
class ModelCapabilities:
    model: str
    aspect_ratios: tuple[str, ...] = ()
    resolutions: tuple[str, ...] = ()
    n_min: int = 1
    n_max: int | None = None
    input_refs_max: int | None = None
    extra_flags: frozenset = frozenset()


def _parse_item(item: dict) -> ModelCapabilities | None:
    model_id = item.get("id")
    if not model_id:
        return None
    sp = item.get("supported_parameters") or {}

    def _enum(name: str) -> tuple[str, ...]:
        d = sp.get(name) or {}
        values = d.get("values") if d.get("type") == "enum" else None
        return tuple(values) if isinstance(values, list) else ()

    def _safe_int(value) -> int | None:
        # External API data: a stringy "10" must not crash the min()/max()
        # clamps downstream, and garbage must degrade to "no bound".
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _range(name: str) -> tuple[int | None, int | None]:
        d = sp.get(name) or {}
        if d.get("type") != "range":
            return None, None
        return _safe_int(d.get("min")), _safe_int(d.get("max"))

    n_min, n_max = _range("n")
    _, refs_max = _range("input_references")
    flags = frozenset(
        k for k, v in sp.items() if isinstance(v, dict) and v.get("type") == "boolean"
    )
    return ModelCapabilities(
        model=model_id,
        aspect_ratios=_enum("aspect_ratio"),
        resolutions=_enum("resolution"),
        n_min=1 if n_min is None else n_min,
        n_max=n_max,
        input_refs_max=refs_max,
        extra_flags=flags,
    )


class CapabilityCatalog:
    """One fetch per process; failures cache as an empty catalog so a dead
    network costs a single 2s attempt for the whole session, not one per
    wizard step."""

    def __init__(self) -> None:
        self._models: dict[str, ModelCapabilities] | None = None

    def get(self, model: str) -> ModelCapabilities | None:
        if self._models is None:
            self._models = self._fetch()
        return self._models.get(model)

    def _fetch(self) -> dict[str, ModelCapabilities]:
        headers = {}
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            resp = requests.get(
                OPENROUTER_IMAGE_MODELS_URL,
                headers=headers,
                timeout=_FETCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json().get("data") or []
        except Exception:
            print(
                "[OpenRouter] Capability discovery unavailable, "
                "using built-in defaults."
            )
            return {}
        models: dict[str, ModelCapabilities] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            caps = _parse_item(item)
            if caps:
                models[caps.model] = caps
        return models


_catalog = CapabilityCatalog()


def get_capabilities(model: str) -> ModelCapabilities | None:
    """Descriptor for a model, or None (unknown model / discovery failed)."""
    return _catalog.get(model)


def reset_catalog() -> None:
    """Drop the session cache (tests / long-lived processes)."""
    global _catalog
    _catalog = CapabilityCatalog()
