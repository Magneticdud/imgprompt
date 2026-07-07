"""Live discovery for OpenRouter's /api/v1/images catalog (issues #11, #3).

Two things live here:

1. **Capability descriptors** — `/api/v1/images/models` returns, per model,
   a `supported_parameters` block with typed descriptors (enum / range /
   boolean) for aspect_ratio, resolution, n, input_references, seed, etc.
   Parsed into ModelCapabilities and handed to the OpenRouter provider so
   wizard choices and payload clamps track upstream truth instead of
   hardcoded branches.

2. **Live pricing** — `/api/v1/images/models/<id>/endpoints` carries a
   `pricing[]` array per endpoint. Flat per-image prices override the
   hardcoded `presets.COSTS` estimates in the wizard; token-billed models
   (unit != "image") keep the hardcoded estimate.

Caching: the models catalog is cached to `.cache/openrouter_models.json`
with a 24h TTL. Fresh cache → no network at all. Stale cache → served
immediately, refreshed by a background daemon thread for the *next* run
(a stale catalog beats blocking the wizard). No cache + network down →
hardcoded fallback with a one-line note. Pricing is cached in-memory per
session only (one request per model actually selected).

Failure policy everywhere: degrade to "no live data" — callers keep their
hardcoded behaviour — and never print more than one note per session.

NOTE: only OpenRouter gets live discovery. OpenAI, Google direct and OVH
expose no comparable pricing/capability API, so their tables in
presets.py stay hardcoded on purpose.
"""

import json
import os
import threading
import time
from dataclasses import dataclass

import requests

OPENROUTER_IMAGE_MODELS_URL = "https://openrouter.ai/api/v1/images/models"

# Tight budget: discovery must never make the wizard feel slow. On a miss
# we fall back to the hardcoded tables, which is exactly what the CLI did
# before this module existed.
_FETCH_TIMEOUT = (2.0, 2.0)

# File cache for the models catalog (issue #3): repo root /.cache, 24h TTL.
CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".cache",
    "openrouter_models.json",
)
CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class ModelCapabilities:
    model: str
    aspect_ratios: tuple[str, ...] = ()
    resolutions: tuple[str, ...] = ()
    n_min: int = 1
    n_max: int | None = None
    input_refs_max: int | None = None
    extra_flags: frozenset = frozenset()


@dataclass(frozen=True)
class PriceEntry:
    billable: str  # e.g. "output_image", "input_image", "input_text"
    unit: str  # "image" (flat per image) or "token"
    cost_usd: float
    variant: str | None = None  # tier variant like "1k"/"2k" when priced per tier


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


def _parse_data(data: list) -> dict[str, ModelCapabilities]:
    models: dict[str, ModelCapabilities] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        caps = _parse_item(item)
        if caps:
            models[caps.model] = caps
    return models


def _auth_headers() -> dict:
    key = os.getenv("OPENROUTER_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


class CapabilityCatalog:
    """Catalog with file cache + TTL. One network attempt per process at
    most; failures cache as an empty catalog so a dead network costs a
    single 2s attempt for the whole session, not one per wizard step."""

    def __init__(self) -> None:
        self._models: dict[str, ModelCapabilities] | None = None

    def get(self, model: str) -> ModelCapabilities | None:
        if self._models is None:
            self._models = self._load()
        return self._models.get(model)

    # ------------------------------------------------------------- loading

    def _load(self) -> dict[str, ModelCapabilities]:
        cached = self._read_cache()
        # Structural validity, not content, decides whether the cache
        # counts: a fresh-but-empty catalog is a valid answer from
        # upstream and must not be mistaken for "no cache" (which would
        # re-fetch on every session).
        if (
            cached is not None
            and isinstance(cached.get("data"), list)
            and "fetched_at" in cached
        ):
            age = time.time() - cached["fetched_at"]
            models = _parse_data(cached["data"])
            if age <= CACHE_TTL_SECONDS:
                return models
            # Stale: serve immediately (a stale catalog beats blocking
            # the wizard) and refresh in the background for next run.
            threading.Thread(target=self._refresh_cache, daemon=True).start()
            return models
        # Cold start: one synchronous, tightly-bounded fetch.
        data = self._fetch_data()
        if data is None:
            print(
                "[OpenRouter] Capability discovery unavailable, "
                "using built-in defaults."
            )
            return {}
        self._write_cache(data)
        return _parse_data(data)

    def _refresh_cache(self) -> None:
        data = self._fetch_data()
        if data is not None:
            self._write_cache(data)

    def _fetch_data(self) -> list | None:
        try:
            resp = requests.get(
                OPENROUTER_IMAGE_MODELS_URL,
                headers=_auth_headers(),
                timeout=_FETCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json().get("data")
        except Exception:
            return None
        return data if isinstance(data, list) else None

    # ------------------------------------------------------------- cache IO

    def _read_cache(self) -> dict | None:
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return cached if isinstance(cached, dict) else None

    def _write_cache(self, data: list) -> None:
        payload = {"fetched_at": time.time(), "data": data}
        tmp_path = f"{CACHE_FILE}.tmp"
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            # Atomic swap so a concurrent reader (or the background refresh
            # racing a second process) never sees a half-written file.
            os.replace(tmp_path, CACHE_FILE)
        except OSError:
            pass  # cache is an optimisation; never fail the wizard over it


_catalog = CapabilityCatalog()

# Per-session pricing cache: model id -> tuple[PriceEntry, ...] | None.
_pricing_cache: dict[str, tuple[PriceEntry, ...] | None] = {}
_pricing_note_printed = False


def get_capabilities(model: str) -> ModelCapabilities | None:
    """Descriptor for a model, or None (unknown model / discovery failed)."""
    return _catalog.get(model)


def get_live_pricing(model: str) -> tuple[PriceEntry, ...] | None:
    """pricing[] of the model's first endpoint, or None when unavailable."""
    global _pricing_note_printed
    if model in _pricing_cache:
        return _pricing_cache[model]
    entries: tuple[PriceEntry, ...] | None = None
    try:
        resp = requests.get(
            f"{OPENROUTER_IMAGE_MODELS_URL}/{model}/endpoints",
            headers=_auth_headers(),
            timeout=_FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
        endpoints = (body.get("data") or {}).get("endpoints") or []
        pricing = endpoints[0].get("pricing") if endpoints else None
        if isinstance(pricing, list):
            parsed = []
            for p in pricing:
                if not isinstance(p, dict) or "cost_usd" not in p:
                    continue
                parsed.append(
                    PriceEntry(
                        billable=p.get("billable", ""),
                        unit=p.get("unit", ""),
                        cost_usd=float(p["cost_usd"]),
                        variant=p.get("variant"),
                    )
                )
            entries = tuple(parsed) if parsed else None
    except Exception:
        entries = None
    if entries is None and not _pricing_note_printed:
        _pricing_note_printed = True
        print("[OpenRouter] Live pricing unavailable, using cached estimates.")
    _pricing_cache[model] = entries
    return entries


def output_image_price(model: str, tier: str | None) -> float | None:
    """Flat per-image output price for a tier, from live pricing.

    Variant-priced entries (e.g. Grok: 1k/2k) are matched case-insensitively
    against the tier; a variant-less output_image entry is the flat price
    for every tier (Recraft, seedream). Token-billed models return None so
    callers keep the hardcoded per-image estimate.
    """
    entries = get_live_pricing(model)
    if not entries:
        return None
    tier_l = tier.lower() if tier else None
    flat = None
    for e in entries:
        if e.billable != "output_image" or e.unit != "image":
            continue
        if e.variant is not None:
            if tier_l and e.variant.lower() == tier_l:
                return e.cost_usd
        else:
            flat = e.cost_usd
    return flat


def reset_catalog() -> None:
    """Drop the session caches (tests / long-lived processes)."""
    global _catalog, _pricing_cache, _pricing_note_printed
    _catalog = CapabilityCatalog()
    _pricing_cache = {}
    _pricing_note_printed = False
