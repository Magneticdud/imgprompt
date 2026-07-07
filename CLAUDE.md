# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-entry-point terminal wizard (`imgedit.py`) that edits or generates images through several hosted model APIs (OpenAI, Google, OVH, and many models via OpenRouter). The wizard walks the user through provider → model → resolution → quality → prompt, estimates the USD cost before spending, then dispatches to a provider. It also supports batch mode, dual-image mode, replay, and PDF rasterization.

## Commands

```bash
pip install -r requirements-dev.txt   # runtime deps + pytest (use requirements.txt for runtime only)
pytest                                 # full suite
pytest -v                              # one line per test
pytest tests/test_presets.py           # single file
pytest -k physical_to_pixels           # tests matching a name expression
python imgedit.py [options] [paths...] # run the tool (see README for the full flag set)
```

Only pure logic is tested — dimension/DPI math, pricing-token estimates, image helpers, capability parsing, and the replay-history round-trip. **Network calls to providers are intentionally not tested**; do not add tests that hit real APIs. `tests/test_presets.py` is the home for gpt-image-2 dimension math, including a grid test asserting `auto_adjust_gpt_image2_dims` always returns valid dims — extend it when touching that code.

## Architecture

- **`imgedit.py`** — the CLI/wizard. Owns argument parsing (`main`), the interactive `step_*` functions, replay flow (`run_replay`, `run_replay_on_different_model`), input resolution (globs, directories, PDFs, `.txt` prompt files), and `PROVIDER_MAP` (the `provider_name() -> class` registry, ~line 56). This file talks to providers **only** through the abstract interface below.
- **`imgprompt/providers/base.py`** — the contract. `GenerationRequest` is the single dataclass passed end-to-end (wizard → provider → history JSON). `ImageProvider` is the ABC every provider implements: `run`, `supported_models`, `provider_name`, and the four wizard-driving hooks (`get_resolution_choices`/`resolve_resolution`, `get_quality_choices`/`resolve_quality`). The wizard asks the provider what choices to offer and how to resolve a selection, so provider-specific capability logic never leaks into `imgedit.py`.
- **`imgprompt/providers/{openai,google,openrouter,ovh}_provider.py`** — one class each. OpenRouter is by far the largest (multiple model families, per-model resolution/`n` clamping, SVG handling for Recraft vector models). Google direct is marked untested/unverified — the OpenRouter path to Gemini is the trusted one.
- **`imgprompt/providers/capabilities.py`** — **OpenRouter-only** live discovery. Fetches `/api/v1/images/models` (capability descriptors) and per-model `/endpoints` (live pricing), cached to `.cache/openrouter_models.json` (24h TTL, gitignored). Degrade-gracefully by design: no cache + no network → hardcoded fallback in `presets.py`, one note per session max. OpenAI/Google/OVH expose no such API, so their tables stay hardcoded on purpose.
- **`imgprompt/presets.py`** — hardcoded pricing/dimension tables and the math: gpt-image-2 token/cost estimation, dimension validation and auto-adjustment, physical-unit (cm/inch) → pixel conversion, Recraft defaults. This is the fallback source of truth when live discovery is unavailable.
- **`imgprompt/images.py`** — image I/O helpers: cwd listing, PDF detection/rasterization (PyMuPDF), pre-upload processing/downscaling, closest-aspect-ratio matching, and saving API results (including detecting SVG bytes to write a real `.svg`).
- **`imgprompt/history.py`** — dumps `(provider, GenerationRequest)` to gitignored `.last_generation.json` after every completed run and reloads it for `--replay`.

## Conventions to preserve

- **`GenerationRequest` is a persisted schema.** It is serialized to `.last_generation.json`, so new fields **must** have defaults — old replay files unpack without them (see the comments on `n`, `is_dual`, `extras`, `estimated_cost` in `base.py`). Never make a field required.
- **Adding a provider**: implement `ImageProvider`, register it in `PROVIDER_MAP`. The wizard picks it up automatically; capability/pricing hooks let it drive its own resolution/quality menus without editing `imgedit.py`.
- **Wizard step functions** return either a value, `None` (user cancelled with Ctrl+C), or the `BACK_OPTION` sentinel (from `presets.py`) to step backward. Preserve this three-way convention when editing steps.
- **Cost reconciliation**: `estimated_cost` on the request is compared to the provider-reported `usage.cost` after each call; a >10% divergence prints a warning. If you change pricing tables in `presets.py` or the live-pricing path, keep that reconciliation meaningful.
- Prices, model lists, and behavior notes are documented in `README.md` — keep it in sync when adding/removing models or changing cost math.
