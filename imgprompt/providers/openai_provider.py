import os
import sys

from openai import OpenAI

from imgprompt.providers.base import ImageProvider, GenerationRequest
from imgprompt.images import process_image_for_api, save_api_image


class OpenAIProvider(ImageProvider):
    @classmethod
    def provider_name(cls) -> str:
        return "OpenAI"

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["gpt-image-2"]

    def get_resolution_choices(
        self, model: str, image_path: str | None
    ) -> tuple[list[str], str]:
        from imgprompt.presets import (
            GPT_IMAGE_2_PRESET_CHOICES,
            CUSTOM_DIMS,
            GPT_IMAGE_2_AUTO,
        )

        if model == "gpt-image-2":
            labels = [label for label, _, _, _, _ in GPT_IMAGE_2_PRESET_CHOICES]
            if image_path:
                default = GPT_IMAGE_2_AUTO
            else:
                default = labels[0]
            return [GPT_IMAGE_2_AUTO] + labels + [CUSTOM_DIMS], default
        else:
            res_options = [
                "1024x1024 (Square)",
                "1024x1536 (Vertical)",
                "1536x1024 (Horizontal)",
            ]
            if image_path:
                from imgprompt.images import get_closest_aspect_ratio

                default = get_closest_aspect_ratio(image_path, res_options)
            else:
                default = res_options[0]
            return res_options, default

    def resolve_resolution(
        self, model: str, selection: str
    ) -> tuple[str, int | None, int | None]:
        from imgprompt.presets import GPT_IMAGE_2_PRESET_CHOICES, GPT_IMAGE_2_AUTO

        if model == "gpt-image-2":
            if selection == GPT_IMAGE_2_AUTO:
                return "auto", None, None
            for label, _, _, w, h in GPT_IMAGE_2_PRESET_CHOICES:
                if label == selection:
                    return f"{w}x{h}", w, h
            return "1024x1024", 1024, 1024
        else:
            return selection.split(" ")[0], None, None

    def get_quality_choices(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        image_path: str | None,
    ) -> tuple[list[str], str]:
        from imgprompt.presets import (
            COSTS,
            calc_gpt_image2_tokens,
            GPT_IMAGE_2_PRICE_PER_MTOK,
        )

        if model == "gpt-image-2":
            choices = []
            for q in ["Low", "Medium", "High"]:
                if width is None or height is None:
                    choices.append(f"{q} (cost depends on output size)")
                else:
                    tokens = calc_gpt_image2_tokens(width, height, q)
                    cost = tokens * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000
                    choices.append(f"{q} (~{tokens:,} tokens, ${cost:.4f})")
            return choices, choices[0]
        else:
            choices = [
                f"{q} (${COSTS[model][q][res_key]:.3f})"
                for q in ["Low", "Medium", "High"]
            ]
            return choices, choices[0]

    def resolve_quality(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        selection: str,
    ) -> tuple[str, float]:
        from imgprompt.presets import (
            COSTS,
            calc_gpt_image2_tokens,
            GPT_IMAGE_2_PRICE_PER_MTOK,
        )

        quality_key = selection.split(" ")[0]
        if model == "gpt-image-2":
            if width is None or height is None:
                return quality_key, 0.0
            tokens = calc_gpt_image2_tokens(width, height, quality_key)
            return quality_key, tokens * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000
        else:
            return quality_key, COSTS[model][quality_key][res_key]

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
        # Aggregators for the issue #4 real-cost report. Each batch call is an
        # independent API request, so token counts and costs add linearly
        # across images. Skipped (kept at the empty initial value) whenever
        # `_extract_usage` returns None — we still end up printing nothing
        # if *no* image reported usage, but a partial batch (some calls did,
        # some didn't) still gets a meaningful aggregate.
        batch_tokens = 0
        batch_cost = 0.0

        for idx, img_path in enumerate(request.images, 1):
            print(
                f"\n--- Processing image {idx}/{len(request.images)}: {os.path.basename(img_path)} ---"
            )
            try:
                image_input = process_image_for_api(img_path, request.res_key)
                kwargs = dict(
                    model=request.model,
                    image=image_input,
                    prompt=request.prompt,
                    n=1,
                    quality=request.quality_key.lower(),
                )
                if request.res_key:
                    kwargs["size"] = request.res_key
                response = client.images.edit(**kwargs)
                image_url = None
                image_b64 = None
                if hasattr(response, "data") and len(response.data) > 0:
                    image_url = getattr(response.data[0], "url", None)
                    image_b64 = getattr(response.data[0], "b64_json", None)

                if image_url or image_b64:
                    save_api_image(image_url, image_b64, img_path)
                    success_count += 1
                    # Print per-image real cost immediately after the save
                    # line so the user sees estimated vs. actual side by side
                    # in the live run. Aggregated total lands in the summary
                    # block below.
                    tokens, cost = self._report_usage(response)
                    if tokens is not None:
                        batch_tokens += tokens
                        batch_cost += cost
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
        if batch_tokens > 0:
            print(
                f"[OpenAI] batch usage total: {batch_tokens:,} tokens "
                f"(${batch_cost:.4f})"
            )

    def _generate(self, client: OpenAI, request: GenerationRequest):
        """Text-to-Image: no input image, so use the generate endpoint."""
        kwargs = dict(
            model=request.model,
            prompt=request.prompt,
            n=1,
            size=request.res_key,
            quality=request.quality_key.lower(),
        )
        return client.images.generate(**kwargs)

    def _save_response(self, response, original_path: str | None) -> None:
        image_url = None
        image_b64 = None
        if hasattr(response, "data") and len(response.data) > 0:
            image_url = getattr(response.data[0], "url", None)
            image_b64 = getattr(response.data[0], "b64_json", None)

        if image_url or image_b64:
            save_api_image(image_url, image_b64, original_path)
        else:
            print("\nError: Could not retrieve image data from the API response.")

    def _run_single(self, client: OpenAI, request: GenerationRequest) -> None:
        print(f"\nSending request to OpenAI ({request.model})...")
        try:
            if request.is_text_to_image:
                response = self._generate(client, request)
                self._save_response(response, None)
                self._report_usage(response)
                return

            if len(request.images) == 1:
                image_input = process_image_for_api(request.images[0], request.res_key)
            else:
                image_input = [
                    process_image_for_api(p, request.res_key) for p in request.images
                ]

            kwargs = dict(
                model=request.model,
                image=image_input,
                prompt=request.prompt,
                n=1,
                quality=request.quality_key.lower(),
            )
            if request.res_key:
                kwargs["size"] = request.res_key
            response = client.images.edit(**kwargs)
            self._save_response(response, request.primary_image)
            # Report the real token count + cost the API actually charged
            # for this call, so it sits next to the pre-call wizard estimate
            # in the terminal scrollback. No-op when the response lacks a
            # `usage` block (defensive: older models / network errors
            # before full parse must not surface as a hard crash).
            self._report_usage(response)

        except Exception as e:
            error_msg = str(e)
            if "moderation_blocked" in error_msg:
                print("\n>> The request was rejected by the OpenAI safety system.")
                print(
                    ">> This typically happens with images of famous people, children, or NSFW content."
                )
            else:
                print(f"\nAn error occurred during OpenAI call: {e}")

    def _report_usage(self, response) -> tuple[int | None, float | None]:
        """Print and return the real token count + USD cost from an OpenAI
        ImagesResponse. Defensive across all the shapes the SDK / API have
        shipped:

        - ``response.usage`` missing entirely (older models): no print, no
          return value, falls back to the wizard's pre-call estimate that
          the user already saw.
        - ``response.usage`` set with both ``input_tokens`` and
          ``output_tokens``: used directly, cost computed with split
          per-modality pricing (input is cheaper than output on
          gpt-image-2 — they're billed at different rates).
        - ``response.usage`` set with only ``total_tokens``: total is
          assumed to be output-only (the higher rate), so we over-estimate
          the bill rather than under-charge. gpt-image-2 always returns
          both fields in practice; this branch is the defensive fallback.
        - ``response.usage`` as a plain ``dict`` instead of an SDK object:
          tolerated because some wrappers / mock servers return it that
          way; we read it the same way either way.
        - Floating-point or stringy token counts: coerced via ``int(...)``
          on a successful parse, never silently accepted on parse error.

        Cost is computed with split per-million-token pricing from
        ``GPT_IMAGE_2_INPUT_PRICE_PER_MTOK`` and
        ``GPT_IMAGE_2_PRICE_PER_MTOK`` (the latter is the output rate).
        The legacy OpenAI image models that this provider still supports
        (the non-gpt-image-2 "dall-e" branch with fixed per-image
        pricing) don't return a ``usage`` block at all, which is the
        documented fallback path.

        Returns ``(tokens, cost)`` when usage was readable, otherwise
        ``(None, None)``. ``tokens`` is the sum of input + output for the
        convenience of the batch aggregator; the split is shown in the
        printed line. ``_run_batch`` sums these across calls; callers
        that just want the print can ignore the return value.
        """
        # If the API gave us no image data, there is nothing useful to
        # charge for either. Skipping here keeps the wizard output clean
        # in the rare case where the response carries an accounting block
        # but ``data`` is empty (the only remaining "error + usage" pair).
        if not getattr(response, "data", None):
            return None, None

        from imgprompt.presets import (
            GPT_IMAGE_2_INPUT_PRICE_PER_MTOK,
            GPT_IMAGE_2_PRICE_PER_MTOK,
        )

        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None

        usage_dict = usage if isinstance(usage, dict) else None

        def _read(field: str):
            if usage_dict is not None:
                return usage_dict.get(field)
            return getattr(usage, field, None)

        raw_input = _read("input_tokens")
        raw_output = _read("output_tokens")
        try:
            input_tokens = int(raw_input) if raw_input is not None else 0
            output_tokens = int(raw_output) if raw_output is not None else 0
        except (TypeError, ValueError):
            return None, None

        # Fallback when only ``total_tokens`` is present (no per-modality
        # split): bill everything at the output rate. The output rate is the
        # more expensive tier on gpt-image-2, so this over-estimates rather
        # than under-charges if the SDK ever flips the API's response shape.
        if input_tokens == 0 and output_tokens == 0:
            total_raw = _read("total_tokens")
            if total_raw is None:
                return None, None
            try:
                output_tokens = int(total_raw)
            except (TypeError, ValueError):
                return None, None

        cost = (
            input_tokens * GPT_IMAGE_2_INPUT_PRICE_PER_MTOK
            + output_tokens * GPT_IMAGE_2_PRICE_PER_MTOK
        ) / 1_000_000
        total_tokens = input_tokens + output_tokens
        print(
            f"\n[OpenAI] actual usage: {total_tokens:,} tokens "
            f"(in={input_tokens:,}, out={output_tokens:,}, ${cost:.4f})"
        )
        return total_tokens, cost
