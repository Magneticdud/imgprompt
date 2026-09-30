# Dedicated OpenAI image key (`OPENAI_IMAGE_API_KEY`)

## Goal
Allow an optional dedicated OpenAI key for image generation so image spend
bills to a separate key/project, while keeping zero-config back-compat for
users who only have `OPENAI_API_KEY` set.

## Decisions (agreed with user)
- New var name: `OPENAI_IMAGE_API_KEY`, optional.
- Precedence: `OPENAI_IMAGE_API_KEY` → fallback `OPENAI_API_KEY` →
  clean error mentioning both vars.
- Env-only, no CLI flag (consistent with all other providers; no
  `--api-key` flag exists today).
- Scope: direct `OpenAI` provider only. OpenRouter `openai/*` models keep
  using `OPENROUTER_API_KEY` (different billing route by design).

## Context (verified in code)
- `imgprompt/providers/openai_provider.py:141` reads only
  `os.getenv("OPENAI_API_KEY")` in `run()` and `sys.exit(1)` if missing.
- Same single-env pattern in `google_provider.py:203`
  (`GOOGLE_API_KEY`), `openrouter_provider.py:509`
  (`OPENROUTER_API_KEY`), `ovh_provider.py:59`.
- `load_dotenv()` in `imgedit.py:28` already loads `.env`, so no wiring
  change needed there.
- Key is resolved at call time, never persisted: no change needed in
  `GenerationRequest` (`base.py`) or replay/history flow.

## Implementation tasks
1. **`imgprompt/providers/openai_provider.py`**
   - Add a module-level (or staticmethod) helper, e.g.
     `resolve_openai_api_key() -> tuple[str | None, str | None]`
     returning `(key, source_var_name)`.
     - Read both env vars, `strip()` whitespace.
     - Treat unset/empty/blank dedicated key as absent → fallback.
     - Do NOT add `your_`-placeholder filtering (existing OpenAI path
       doesn't filter either; keep scope minimal).
   - `run()` uses the helper instead of direct `os.getenv`.
   - Error when neither is set mentions both vars, e.g.
     `"OPENAI_IMAGE_API_KEY nor OPENAI_API_KEY found…"`.
   - Print which source was used (var name only, never the value), so the
     user can confirm billing separation at a glance. Recommended: always
     print one line, e.g. `[OpenAI] using OPENAI_IMAGE_API_KEY` /
     `[OpenAI] using OPENAI_API_KEY`.
2. **`.env.example`** — add commented optional entry:
   `# OPENAI_IMAGE_API_KEY=...  # optional: dedicated image key, falls back to OPENAI_API_KEY`
3. **`README.md`** — Setup env block (lines ~22-28): document the optional
   var + fallback in one or two lines. Optionally one line in the OpenAI
   bullet (§ Supported Models) clarifying OpenRouter `openai/*` still
   bills via `OPENROUTER_API_KEY`.
4. **Tests (`tests/test_openai_provider.py`)** — new `TestApiKeyResolution`
   class covering the helper (no network):
   - dedicated set → dedicated wins (both set).
   - only default set → default used.
   - neither set → `(None, None)` / run-path exits mentioning both vars.
   - blank dedicated (`""` / whitespace) → falls back to default.
   - For the `run()` exit path, mock the `OpenAI` client constructor so no
     network/SDK call happens.

## Edge cases
- `OPENAI_IMAGE_API_KEY=""` or whitespace-only → same as unset (fallback).
- Both set → dedicated wins silently (plus the source line in output).
- Neither set → `sys.exit(1)` as today, message names both vars.
- Replay/batch/dual flows: unaffected (key resolved fresh in `run()`).

## Validation
- `pytest tests/test_openai_provider.py` (new + existing tests green).
- Full `pytest` suite.
- Manual: `OPENAI_IMAGE_API_KEY=sk-test python imgedit.py --free --replay`
  or wizard run confirms the source line; unset dedicated key still works
  via `OPENAI_API_KEY`.

## Out of scope
- Placeholder (`your_…`) filtering, CLI `--openai-key` flag, per-model
  keys, key rotation, other providers' keys, persisting key choice in
  `.last_generation.json`.
