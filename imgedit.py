#!/usr/bin/env python3
import glob
import os
import re
import sys

# Auto-activate venv if it exists and we aren't using it
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.join(script_dir, ".venv")
venv_python = os.path.join(venv_dir, "bin", "python")

if os.path.exists(venv_python):
    # Check if we are running from this venv by comparing sys.prefix
    if os.path.abspath(sys.prefix) != os.path.abspath(venv_dir):
        # Re-execute the script with the venv python
        try:
            os.execv(venv_python, [venv_python] + sys.argv)
        except OSError:
            # Fallback if exec fails
            print("Warning: Could not auto-activate .venv. Running with system python.")

import argparse
import questionary
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from imgprompt.presets import (
    BACK_OPTION,
    COSTS,
    CUSTOM_DIMS,
    PRESET_PROMPTS_EDIT,
    PRESET_PROMPTS_GENERATE,
    PRESET_PROMPTS_DUAL,
    GPT_IMAGE_2_PRICE_PER_MTOK,
    calc_gpt_image2_tokens,
    validate_gpt_image2_dims,
    auto_adjust_gpt_image2_dims,
    physical_to_pixels,
)
from imgprompt.images import (
    get_images_in_cwd,
    is_pdf,
    pdf_page_count,
    rasterize_pdf,
)
from imgprompt.history import save_last_generation, load_last_generation
from imgprompt.providers.base import GenerationRequest
from imgprompt.providers.openai_provider import OpenAIProvider
from imgprompt.providers.google_provider import GoogleProvider
from imgprompt.providers.openrouter_provider import OpenRouterProvider
from imgprompt.providers.ovh_provider import OVHProvider

PROVIDER_MAP = {
    OpenAIProvider.provider_name(): OpenAIProvider,
    GoogleProvider.provider_name(): GoogleProvider,
    OpenRouterProvider.provider_name(): OpenRouterProvider,
    OVHProvider.provider_name(): OVHProvider,
}


def multiline_prompt(message: str, default: str = "") -> str | None:
    """Multi-line text input. Enter submits, Ctrl+J inserts a newline.

    Built on prompt_toolkit (already a questionary dependency) so prompts can
    span several lines -- useful for typographic prompts where the literal line
    breaks signal how text should be laid out in the image.

    Returns the entered text, or None if cancelled (Ctrl+C / Ctrl+D).
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings

    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    session = PromptSession(multiline=True, key_bindings=bindings)
    try:
        return session.prompt(f"? {message} (Ctrl+J = newline) ", default=default)
    except (KeyboardInterrupt, EOFError):
        return None


def normalize_path(p: str) -> str:
    """Return a usable filesystem path, tolerating shell-escaped paths.

    Paths copied from file managers (e.g. GVFS/SMB mounts) often arrive with
    backslash escapes like ``server\\=server2.local,share\\=2_personale``. When
    such a path is pasted inside quotes the backslashes become literal and the
    lookup fails. If the path as given doesn't exist but its un-escaped form
    does, use that instead. A valid path is never modified.
    """
    if os.path.exists(p):
        return p
    unescaped = re.sub(r"\\(.)", r"\1", p)
    if unescaped != p and os.path.exists(unescaped):
        return unescaped
    return p


def resolve_pdf_inputs(paths: list[str]) -> list[str]:
    """Rasterize any PDF inputs to PNG so providers receive a bitmap.

    Image APIs accept raster formats only, never PDF, so each PDF is rendered to
    a PNG up front -- before any provider sees the path -- leaving all downstream
    loading/resizing code unchanged. For a multi-page PDF the user picks which
    page to use. Returns the input list with PDF paths swapped for PNG paths.
    """
    resolved = []
    for p in paths:
        if not is_pdf(p):
            resolved.append(p)
            continue
        try:
            n_pages = pdf_page_count(p)
        except Exception as e:
            print(f"Error: could not open PDF '{p}': {e}")
            sys.exit(1)

        page_index = 0
        if n_pages > 1:
            choice = questionary.select(
                f"'{os.path.basename(p)}' has {n_pages} pages. Select a page to use:",
                choices=[f"Page {i + 1}" for i in range(n_pages)],
            ).ask()
            if not choice:
                sys.exit(0)
            page_index = int(choice.split()[1]) - 1

        try:
            png_path = rasterize_pdf(p, page_index=page_index)
        except Exception as e:
            print(f"Error: could not rasterize PDF '{p}': {e}")
            sys.exit(1)
        print(f"Rasterized {os.path.basename(p)} (page {page_index + 1}) -> {png_path}")
        resolved.append(png_path)
    return resolved


def select_inputs(provided_path: str | None) -> tuple[list[str], bool, str | None]:
    """Selects images either from arguments or from a list of files.

    Returns (paths, is_dual, replay_choice) where replay_choice is None or
    one of REPLAY_OPTION / REPLAY_DIFFERENT_OPTION."""
    if provided_path:
        if os.path.isfile(provided_path):
            return [provided_path], False, None
        else:
            print(f"Error: {provided_path} is not a valid file.")
            sys.exit(1)

    images = get_images_in_cwd()

    # Add option for Text-to-Image
    t2i_option = "Text-to-Image (No input image)"
    dual_option = "Two Images (Dual Input)"

    choices = []
    # Offer replay as the first choices only if a previous run was saved.
    if load_last_generation() is not None:
        choices.append(REPLAY_OPTION)
        choices.append(REPLAY_DIFFERENT_OPTION)
    choices.append(t2i_option)
    if len(images) >= 2:
        choices.append(dual_option)
    choices.extend(images)

    selected = questionary.select(
        "Select an image to edit or mode:", choices=choices
    ).ask()

    if not selected:
        sys.exit(0)

    if selected in (REPLAY_OPTION, REPLAY_DIFFERENT_OPTION):
        return [], False, selected

    if selected == t2i_option:
        return [], False, None

    if selected == dual_option:
        img1 = questionary.select("Select first image (IMG_1):", choices=images).ask()
        if not img1:
            sys.exit(0)

        remaining_images = [img for img in images if img != img1]
        img2 = questionary.select(
            "Select second image (IMG_2):", choices=remaining_images
        ).ask()
        if not img2:
            sys.exit(0)

        return [img1, img2], True, None

    return [selected], False, None


REPLAY_OPTION = "🔁 Replay last generation"
REPLAY_DIFFERENT_OPTION = "🔁 Replay on a different model"


def run_replay(
    iterations_arg: int | None,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> None:
    """Re-run the last saved generation, optionally on a different
    provider/model (issue #12: retry a refused prompt elsewhere without
    re-walking the wizard). Prompt, images, ratio, resolution tier, n and
    extras are reused verbatim. Exits on error."""
    loaded = load_last_generation()
    if not loaded:
        print("Error: no saved generation found to replay.")
        sys.exit(1)
    provider, request = loaded

    if provider_override:
        matched = next(
            (p for p in PROVIDER_MAP if p.lower() == provider_override.lower()),
            None,
        )
        if matched is None:
            print(
                f"Error: unknown provider '{provider_override}'. "
                f"Choices: {', '.join(PROVIDER_MAP)}"
            )
            sys.exit(1)
        provider = matched

    provider_cls = PROVIDER_MAP.get(provider)
    if provider_cls is None:
        print(f"Error: unknown provider in saved generation: {provider}")
        sys.exit(1)

    if model_override:
        request.model = model_override

    # Catch retired-model references (e.g. .last_generation.json saved when
    # gemini-2.5-flash-image was still shipped) AND bad --model overrides.
    # Without this we'd POST to the upstream API and surface a cryptic 404
    # from OpenRouter / Google. Better to bail early with an actionable
    # message, before any network call. We check *before* constructing a
    # provider instance: a side-effecting constructor on a stale-model
    # OVHProvider could otherwise noisily fail.
    if request.model not in provider_cls.supported_models():
        if model_override:
            print(
                f"Error: model '{request.model}' is not supported by "
                f"'{provider}'. Supported models: "
                f"{', '.join(provider_cls.supported_models())}"
            )
        else:
            print(
                f"Error: model '{request.model}' is no longer supported by "
                f"'{provider}'."
            )
            print(
                "The saved generation references a model that has been retired; "
                "please start a new one (drop --replay and run the wizard again)."
            )
        sys.exit(1)

    if provider == "OVH" and request.images:
        print("Error: OVH provider only supports Text-to-Image mode; the saved")
        print("generation has input images and cannot be replayed there.")
        sys.exit(1)

    provider_obj = provider_cls()

    print("\n--- Replaying last generation ---")
    print(f"Provider:   {provider}")
    print(f"Model:      {request.model}")
    if request.images:
        print(f"Images:     {', '.join(request.images)}")
    else:
        print("Image:      None (Text-to-Image)")
    print(f"Prompt:     {request.prompt}")

    # Persist the (possibly overridden) request so a further bare --replay
    # repeats THIS attempt — enabling quick A → B → B retry chains.
    save_last_generation(provider, request)

    iterations = max(1, iterations_arg or 1)
    for i in range(1, iterations + 1):
        if iterations > 1:
            print(f"\n===== Iteration {i}/{iterations} =====")
        provider_obj.run(request)


def run_replay_on_different_model(iterations_arg: int | None) -> None:
    """Interactive flavour of the replay override: a one-shot provider →
    model picker seeded with the saved values, then the same replay path."""
    loaded = load_last_generation()
    if not loaded:
        print("Error: no saved generation found to replay.")
        sys.exit(1)
    saved_provider, request = loaded

    provider_names = list(PROVIDER_MAP.keys())
    provider = questionary.select(
        "Select Provider:",
        choices=provider_names,
        default=saved_provider if saved_provider in provider_names else None,
    ).ask()
    if not provider:
        sys.exit(0)

    provider_cls = PROVIDER_MAP[provider]
    models = provider_cls.supported_models()
    model = questionary.select(
        f"Select {provider} model:",
        choices=models,
        default=request.model if request.model in models else models[0],
    ).ask()
    if not model:
        sys.exit(0)

    run_replay(iterations_arg, model_override=model, provider_override=provider)


def step_provider() -> str | None:
    """Step 1: Select provider. Returns provider name, REPLAY_OPTION,
    REPLAY_DIFFERENT_OPTION or BACK_OPTION."""
    choices = [BACK_OPTION]
    # Offer replay here too so it's reachable even when an image was passed as an
    # argument (which skips the image-selection menu).
    if load_last_generation() is not None:
        choices.append(REPLAY_OPTION)
        choices.append(REPLAY_DIFFERENT_OPTION)
    choices.extend(PROVIDER_MAP.keys())
    provider = questionary.select(
        "Select Provider:",
        choices=choices,
        default="OpenAI",
    ).ask()
    if provider == BACK_OPTION or not provider:
        return BACK_OPTION
    return provider


def step_model(provider_obj) -> str | None:
    """Step 2: Select model. Returns model name or BACK_OPTION."""
    models = provider_obj.supported_models()
    default = provider_obj.default_model()
    model = questionary.select(
        f"Select {provider_obj.provider_name()} model:",
        choices=[BACK_OPTION] + models,
        default=default,
    ).ask()
    if model == BACK_OPTION or not model:
        return BACK_OPTION
    return model


def _prompt_physical_dims() -> tuple[int | None, int | None]:
    """Prompt for physical dimensions (cm/mm/in) and DPI. Returns (width_px, height_px) or (None, None) on back."""
    while True:
        width_str = questionary.text(
            "Width (mm):",
            default="",
        ).ask()
        if not width_str:
            return None, None
        if width_str.strip().lower() == "back":
            return None, None
        try:
            width = float(width_str)
        except ValueError:
            print("Error: Width must be a number.")
            continue

        while True:
            height_str = questionary.text(
                "Height (mm):",
                default="",
            ).ask()
            if not height_str:
                return None, None
            if height_str.strip().lower() == "back":
                return None, None
            try:
                height = float(height_str)
            except ValueError:
                print("Error: Height must be a number.")
                continue

            while True:
                unit = questionary.text(
                    "Unit (cm / mm / in):",
                    default="mm",
                ).ask()
                if not unit:
                    return None, None
                if unit.strip().lower() == "back":
                    return None, None
                unit = unit.strip().lower()
                if unit not in ("cm", "mm", "in"):
                    print("Error: Unit must be 'cm', 'mm', or 'in'.")
                    continue

                while True:
                    dpi_str = questionary.text(
                        "DPI:",
                        default="300",
                    ).ask()
                    if not dpi_str:
                        return None, None
                    if dpi_str.strip().lower() == "back":
                        return None, None
                    try:
                        dpi = int(dpi_str)
                    except ValueError:
                        print("Error: DPI must be a whole number.")
                        continue

                    # Convert physical to pixels
                    px_width = physical_to_pixels(width, unit, dpi)
                    px_height = physical_to_pixels(height, unit, dpi)

                    print(
                        f"→ {width}{unit} × {height}{unit} @ {dpi} DPI = {px_width}×{px_height} px"
                    )

                    # Auto-adjust dimensions to meet requirements
                    adj_width, adj_height = auto_adjust_gpt_image2_dims(
                        px_width, px_height
                    )

                    # Show adjustment if dimensions changed
                    if adj_width != px_width or adj_height != px_height:
                        print(
                            f"Adjusted: {px_width}x{px_height} → {adj_width}x{adj_height} (rounded to multiple of 16)"
                        )

                    # Validate adjusted dimensions
                    errors = validate_gpt_image2_dims(adj_width, adj_height)
                    if errors:
                        print("Invalid dimensions after adjustment:")
                        for err in errors:
                            print(f"  - {err}")
                        break

                    print(
                        f"Valid: {adj_width}x{adj_height} = {adj_width * adj_height:,} px"
                    )
                    return adj_width, adj_height


def _prompt_custom_dims() -> tuple[int | None, int | None]:
    """Prompt for custom width/height with validation. Returns (width, height) or (None, None) on back."""
    mode = questionary.select(
        "How do you want to specify dimensions?",
        choices=["Pixels", "Physical units (cm/mm/in + DPI)"],
    ).ask()
    if not mode:
        return None, None

    if mode == "Physical units (cm/mm/in + DPI)":
        return _prompt_physical_dims()

    # Pixel mode (original flow)
    while True:
        width_str = questionary.text(
            "Width (will be rounded to multiple of 16):",
            default="1024",
        ).ask()
        if not width_str:
            return None, None
        if width_str.strip().lower() == "back":
            return None, None
        try:
            width = int(width_str)
        except ValueError:
            print("Error: Width must be a whole number.")
            continue

        while True:
            height_str = questionary.text(
                "Height (will be rounded to multiple of 16):",
                default="1024",
            ).ask()
            if not height_str:
                return None, None
            if height_str.strip().lower() == "back":
                return None, None
            try:
                height = int(height_str)
            except ValueError:
                print("Error: Height must be a whole number.")
                continue

            # Auto-adjust dimensions to meet requirements
            adj_width, adj_height = auto_adjust_gpt_image2_dims(width, height)

            # Show adjustment if dimensions changed
            if adj_width != width or adj_height != height:
                print(
                    f"Adjusted: {width}x{height} → {adj_width}x{adj_height} (rounded to multiple of 16)"
                )

            # Validate adjusted dimensions
            errors = validate_gpt_image2_dims(adj_width, adj_height)
            if errors:
                print("Invalid dimensions after adjustment:")
                for err in errors:
                    print(f"  - {err}")
                break

            print(f"Valid: {adj_width}x{adj_height} = {adj_width * adj_height:,} px")
            return adj_width, adj_height


def step_resolution(
    provider_obj, model: str, image_path: str | None
) -> tuple[str | None, str | None, int | None, int | None]:
    """Step 3: Select resolution. Returns (selection, res_key, width, height) or (BACK_OPTION, None, None, None)."""
    choices, default = provider_obj.get_resolution_choices(model, image_path)
    selection = questionary.select(
        "Select resolution:",
        choices=[BACK_OPTION] + choices,
        default=default,
    ).ask()
    if selection == BACK_OPTION or not selection:
        return BACK_OPTION, None, None, None
    if selection == CUSTOM_DIMS:
        width, height = _prompt_custom_dims()
        if width is None:
            return BACK_OPTION, None, None, None
        return f"Custom ({width}x{height})", f"{width}x{height}", width, height
    res_key, width, height = provider_obj.resolve_resolution(model, selection)
    return selection, res_key, width, height


def step_quality(
    provider_obj,
    model: str,
    res_key: str,
    width: int | None,
    height: int | None,
    image_path: str | None,
) -> tuple[str | None, float]:
    """Step 4: Select quality/size. Returns (quality_key, cost) or (BACK_OPTION, 0.0)."""
    choices, default = provider_obj.get_quality_choices(
        model, res_key, width, height, image_path
    )
    selection = questionary.select(
        "Select resolution:",
        choices=[BACK_OPTION] + choices,
        default=default,
    ).ask()
    if selection == BACK_OPTION or not selection:
        return BACK_OPTION, 0.0
    quality_key, cost = provider_obj.resolve_quality(
        model, res_key, width, height, selection
    )
    return quality_key, cost


RECRAFT_CUSTOM_STYLE = "Custom (type slug)"
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def is_recraft_model(provider: str | None, model: str | None) -> bool:
    """Gate for the Recraft style step: only OpenRouter Recraft models have
    the style/colors payload axis; every other model must not see it."""
    return provider == "OpenRouter" and bool(model) and model.startswith("recraft/")


def step_recraft_style(model: str) -> tuple[str, list[str]] | str | None:
    """Recraft-only step: pick a style slug and optional brand colors.

    Runs between resolution and quality, only for recraft/* models. The
    values land verbatim in GenerationRequest.extras ("style", "colors"),
    which the OpenRouter provider merges into the JSON body untouched.

    Returns:
      - (style_slug, colors) on success; colors is possibly empty.
      - BACK_OPTION if the user backed out (returns to the resolution step).
      - None if cancelled (Ctrl+C / EOF); the wizard should exit.
    """
    from imgprompt.presets import (
        RECRAFT_STYLE_SLUGS,
        RECRAFT_STYLE_LABELS,
        RECRAFT_BRAND_COLOR_HELP,
        recraft_default_style,
    )

    default_slug = recraft_default_style(model)
    # Default first so accepting the highlighted entry picks the documented
    # default for this variant (vector models default to vector styles).
    ordered = [default_slug] + [s for s in RECRAFT_STYLE_SLUGS if s != default_slug]
    style_choices = [
        questionary.Choice(title=RECRAFT_STYLE_LABELS.get(s, s), value=s)
        for s in ordered
    ]
    selection = questionary.select(
        "Select Recraft style:",
        choices=[BACK_OPTION] + style_choices + [RECRAFT_CUSTOM_STYLE],
    ).ask()
    if selection is None:
        return None
    if selection == BACK_OPTION:
        return BACK_OPTION

    if selection == RECRAFT_CUSTOM_STYLE:
        while True:
            custom = questionary.text("Recraft style slug:").ask()
            if custom is None or not custom.strip():
                return BACK_OPTION
            custom = custom.strip()
            if re.fullmatch(r"[a-z0-9_]+", custom):
                selection = custom
                break
            print("Error: style slugs are lowercase letters, digits and underscores.")

    while True:
        raw = questionary.text(f"Brand colors ({RECRAFT_BRAND_COLOR_HELP}):").ask()
        if raw is None:
            return None
        raw = raw.strip()
        if not raw:
            return selection, []
        colors = [p.strip() for p in raw.split(",") if p.strip()]
        if colors and all(_HEX_COLOR_RE.match(c) for c in colors):
            return selection, colors
        print(f"Error: expected {RECRAFT_BRAND_COLOR_HELP}.")


def step_variants() -> int | str | None:
    """Step 6: number of variants to generate server-side via the OpenRouter
    `n` parameter (1..10). Other providers currently ignore this value, so
    the wizard only invokes this step for OpenRouter.

    Returns:
      - int in [1, 10] on success.
      - BACK_OPTION (str sentinel) if the user picked back.
      - None if the user cancelled (Ctrl+C / EOF); the wizard should exit.
    """
    choices = [str(i) for i in range(1, 11)]
    selection = questionary.select(
        "How many variants? (sent in a single API call via `n`):",
        choices=[BACK_OPTION] + choices,
        default="1",
    ).ask()
    if selection is None:
        return None
    if selection == BACK_OPTION or not selection:
        return BACK_OPTION
    try:
        return max(1, min(10, int(selection)))
    except (TypeError, ValueError):
        return 1


def step_prompt(input_images: list, is_dual: bool) -> tuple[str | None, str]:
    """Step 5: Select prompt. Returns (final_prompt or BACK_OPTION, original_selection)."""
    if is_dual:
        prompt_list = PRESET_PROMPTS_DUAL
    elif input_images:
        prompt_list = PRESET_PROMPTS_EDIT
    else:
        prompt_list = PRESET_PROMPTS_GENERATE

    prompt_choices = []
    for p in prompt_list:
        title = p.replace("\n", " ").strip()
        if len(title) > 100:
            title = title[:97] + "..."
        prompt_choices.append(questionary.Choice(title=title, value=p))

    prompt_selection = questionary.select(
        "Select a prompt or enter a custom one:", choices=prompt_choices
    ).ask()
    if prompt_selection is None:
        return BACK_OPTION, ""

    # For prompts that need text input, ask confirmation first (with Go back option)
    prompts_needing_input = [
        "Custom Prompt",
        "Object Removal (High Quality)",
        "A retro-style BW lettering with thick outline",
        "1990s Memphis Style Logo",
        "Business Card",
        "APPROVED Stamp",
        "Generic Logotype",
        "Comic Book Style Text",
    ]

    if prompt_selection in prompts_needing_input:
        confirm_prompt = questionary.select(
            f"Use '{prompt_selection}'?",
            choices=[BACK_OPTION, "Yes", "No"],
            default="Yes",
        ).ask()
        if confirm_prompt == BACK_OPTION or confirm_prompt == "No":
            return BACK_OPTION, prompt_selection

    final_prompt = prompt_selection
    if prompt_selection == "Custom Prompt":
        final_prompt = multiline_prompt("Enter your custom prompt:")
        if not final_prompt:
            return BACK_OPTION, prompt_selection
    elif prompt_selection == "Object Removal (High Quality)":
        base_prompt = "Preserve the exact composition and identity. Remove JPEG artifacts and noise, enhance real details only. Do not change facial features. Do not hallucinate text or logos; if unreadable, keep it unreadable. High-resolution output."
        remove_input = questionary.text("What to remove?").ask()
        if not remove_input:
            final_prompt = base_prompt
        else:
            final_prompt = f"{base_prompt} Remove {remove_input}."
    elif prompt_selection == "A retro-style BW lettering with thick outline":
        text_input = questionary.text("What text to write?").ask()
        if not text_input:
            print("Error: Text input is required for this preset.")
            return BACK_OPTION, prompt_selection
        final_prompt = (
            f"Create a clean vector-style black and white typographic logo. "
            f'Text: "{text_input}" on two lines, centered and slightly slanted upward to the right. '
            f"Use bold retro script lettering (smooth connected cursive, thick strokes), white fill with a thick black outline. "
            f"Add a large black drop shadow offset down-right to create a strong 3D sticker effect. "
            f"Add a swoosh underline under the second word, also white with black outline and black shadow. "
            f"High contrast, crisp edges, no textures, no gradients, bright green background (it will be keyed out), no extra elements. "
            f"Export as a logo/wordmark."
        )
    elif prompt_selection == "1990s Memphis Style Logo":
        text_input = questionary.text("What text to write?").ask()
        if not text_input:
            print("Error: Text input is required for this preset.")
            return BACK_OPTION, prompt_selection
        final_prompt = (
            f'Create a 1990s Memphis-inspired typographic logo that reads: "{text_input}". '
            f"Style: 90s TV commercial / snack packaging lettering, playful and energetic, slightly italic script with uneven hand-drawn feel, two-tone gradients and airbrushed shading, subtle halftone/print vibe, thin white highlight, no shadow at all, not a thick sticker outline). "
            f"Add a simple zig-zag / squiggle underline in Memphis style. "
            f"Background: solid flat chroma key green (#00FF00), perfectly uniform. "
            f"Strictly avoid: 3D chrome, rainbow neon, glossy metallic look, thick black outline sticker effect, modern esports logo style, glow effects, bevel/emboss, photorealism, extra objects, patterns, textures, collage, food images, shadow. "
            f"Centered composition, clean edges, high resolution."
        )
    elif prompt_selection == "Business Card":
        print("Enter ONLY the main title (e.g. 'DJ Set'):")
        main_title = questionary.text("Main Title:").ask()
        if not main_title:
            print("Error: Main title is required.")
            return BACK_OPTION, prompt_selection

        print(
            "Enter the rest of the details (multiline). Press Alt+Enter or Esc+Enter to submit:"
        )
        details = questionary.text("Details:", multiline=True).ask()
        if not details:
            print("Error: Details are required.")
            return BACK_OPTION, prompt_selection

        final_prompt = (
            f"Create a 2D graphic design for a business card without mockup: no 3D rendering, no scene, no photography, no perspective.\n"
            f"Canvas: 85x55 mm (aspect ratio 1.545:1), horizontal, equivalent to 300 dpi, 3 mm safety margins, text strictly within the safe area.\n"
            f"Text Layout: Centered in a single column. Visual hierarchy: '{main_title}' must be very large and bold. The rest smaller but clearly legible.\n"
            f"Write EXACTLY the following text:\n\n"
            f"{main_title}\n"
            f"{details}\n\n"
            f"Negative constraints: No other text. No QR code. No social media icons. No watermark. No invented logo. Output: flat 2D graphic only."
        )
    elif prompt_selection == "APPROVED Stamp":
        custom_string = questionary.text(
            "Enter the custom text (will be followed by 'APPROVED'):"
        ).ask()
        if not custom_string:
            print("Error: Custom text is required for this preset.")
            return BACK_OPTION, prompt_selection

        final_prompt = (
            f"Create an APPROVED-style rubber stamp graphic, rectangular with slightly rounded corners. "
            f"Text: '{custom_string} APPROVED' on two lines, red, bold, all caps, with a light distressed texture. "
            f"White background. No extra elements, no gradients, no shadows. Vector/flat style, high resolution."
        )
    elif prompt_selection == "Generic Logotype":
        print(
            "Enter the text for the logotype (multiline). Press Alt+Enter or Esc+Enter to submit:"
        )
        logo_text = questionary.text("Logotype Text:", multiline=True).ask()
        if not logo_text:
            print("Error: Text is required for this preset.")
            return BACK_OPTION, prompt_selection

        final_prompt = f"A typographic logo, with centered text, with the following text: {logo_text}"
    elif prompt_selection == "Comic Book Style Text":
        text_input = questionary.text("What text to write?").ask()
        if not text_input:
            print("Error: Text input is required for this preset.")
            return BACK_OPTION, prompt_selection
        final_prompt = (
            f'Comic book style bold text that reads "{text_input}", '
            f"retro superhero font, thick black outline, yellow to red gradient fill, "
            f"halftone dot pattern texture, 3D extruded shadow effect, pop art style, "
            f"white background, high contrast, vintage Marvel/DC comics typography."
        )

    return final_prompt, prompt_selection


def step_confirm() -> str | None:
    """Step 6: Confirm. Returns 'Yes', 'No', or BACK_OPTION."""
    confirm = questionary.select(
        "Proceed with API call?", choices=[BACK_OPTION, "Yes", "No"], default="No"
    ).ask()
    if confirm == BACK_OPTION:
        return BACK_OPTION
    return confirm


def main():
    parser = argparse.ArgumentParser(description="GPT-Image & Gemini Image Editor")
    parser.add_argument(
        "images",
        nargs="*",
        help="Path to one or more images to edit (space-separated, or use glob pattern)",
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="Start in Text-to-Image mode (no base image)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=None,
        help="Number of times to run the same request (generates N variations)",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Re-run the last generation with identical parameters and prompt "
        "(any image arguments are ignored)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="With --replay: retry the saved generation on a different model "
        "(e.g. --replay --model bytedance-seed/seedream-4.5)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="With --replay: retry the saved generation on a different "
        f"provider ({', '.join(PROVIDER_MAP)}); case-insensitive",
    )
    parser.add_argument(
        "-p",
        "--prompt-file",
        default=None,
        help="Read the prompt from a text file (preserves newlines). A .txt "
        "passed as a positional argument is auto-detected as the prompt file.",
    )
    args = parser.parse_args()

    # --model/--provider only make sense as replay modifiers: outside replay
    # the wizard owns those choices and a silent no-op would be confusing.
    if (args.model or args.provider) and not args.replay:
        parser.error("--model and --provider require --replay")

    # Replay mode: skip the wizard and reuse the last saved request verbatim
    # (or retry it on a different provider/model via --provider/--model).
    # Any image arguments are ignored so users can just append --replay to
    # their previous command (e.g. arrow-up then add the flag).
    if args.replay:
        if args.images:
            print("Note: image arguments are ignored in --replay mode.")
        run_replay(args.iterations, args.model, args.provider)
        return

    # Resolve a CLI-supplied prompt: an explicit --prompt-file, or a .txt found
    # among the positional arguments (so `imgedit.py photo.jpg prompt.txt` just
    # works, in any order, without a flag). Non-.txt arguments stay as images.
    prompt_file = args.prompt_file
    remaining_images = []
    for pattern in args.images:
        if (
            prompt_file is None
            and pattern.lower().endswith(".txt")
            and os.path.isfile(normalize_path(pattern))
        ):
            prompt_file = pattern
        else:
            remaining_images.append(pattern)
    args.images = remaining_images

    cli_prompt = None
    if prompt_file:
        pf = normalize_path(prompt_file)
        try:
            with open(pf, "r", encoding="utf-8") as f:
                cli_prompt = f.read().strip()
        except OSError as e:
            print(f"Error: could not read prompt file '{prompt_file}': {e}")
            sys.exit(1)
        if not cli_prompt:
            print(f"Error: prompt file '{prompt_file}' is empty.")
            sys.exit(1)
        print(f"\nPrompt loaded from: {prompt_file}")

    # Determine input images
    is_dual = False
    input_images = []
    if args.free:
        input_images = []
    elif args.images:
        all_images = []
        for pattern in args.images:
            if "*" in pattern or "?" in pattern:
                matched = glob.glob(pattern)
                all_images.extend(matched)
                continue
            pattern = normalize_path(pattern)
            if os.path.isfile(pattern):
                all_images.append(pattern)
            elif os.path.isdir(pattern):
                for f in os.listdir(pattern):
                    fpath = os.path.join(pattern, f)
                    if os.path.isfile(fpath) and f.lower().endswith(
                        (
                            ".pcx",
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".bmp",
                            ".tiff",
                            ".webp",
                            ".pdf",
                        )
                    ):
                        all_images.append(fpath)
            else:
                print(f"Warning: {pattern} is not a valid file or directory.")

        input_images = sorted(list(set(all_images)))

        if not input_images:
            print("Error: No valid images found from provided arguments.")
            sys.exit(1)

        if len(input_images) > 1:
            print(f"\nBatch mode: {len(input_images)} images selected")
    else:
        input_images, is_dual, replay_choice = select_inputs(None)
        if replay_choice == REPLAY_DIFFERENT_OPTION:
            run_replay_on_different_model(args.iterations)
            return
        if replay_choice == REPLAY_OPTION:
            run_replay(args.iterations)
            return

    # Rasterize any PDF inputs to PNG before providers ever see the paths, so all
    # downstream loading/resizing stays image-only.
    if input_images:
        input_images = resolve_pdf_inputs(input_images)

    # Legacy support variable
    image_path = input_images[0] if input_images else None

    if input_images:
        if is_dual:
            print(f"\nSelected Images (dual): {', '.join(input_images)}")
        elif len(input_images) == 1:
            print(f"\nSelected Image: {input_images[0]}")
        else:
            print(f"\nSelected Images: {', '.join(input_images)}")
    else:
        print(f"\nMode: Text-to-Image (No input image)")

    # State machine for navigation
    # step 0=provider, 1=model, 2=resolution, 3=quality, 4=prompt, 5=variants, 6=confirm
    current_step = 0

    # Initialize all variables
    provider = None
    provider_obj = None
    model_choice = None
    aspect_ratio = None
    res_key = None
    quality_key = None
    final_cost = 0.0
    final_prompt = None
    dim_width = None
    dim_height = None
    n_variants = 1  # populated by step_variants (only triggered for OpenRouter)
    recraft_style = None  # populated by step_recraft_style (Recraft only)
    recraft_colors: list[str] = []

    while current_step >= 0:
        if current_step == 0:
            # Step 1: Select Provider
            result = step_provider()
            if result == BACK_OPTION:
                print("Cancelled.")
                return
            if result in (REPLAY_OPTION, REPLAY_DIFFERENT_OPTION):
                if input_images:
                    print("Note: selected image is ignored in replay mode.")
                if result == REPLAY_DIFFERENT_OPTION:
                    run_replay_on_different_model(args.iterations)
                else:
                    run_replay(args.iterations)
                return
            provider = result
            provider_obj = PROVIDER_MAP[provider]()
            if provider == "OVH" and input_images:
                print("\nError: OVH provider only supports Text-to-Image mode.")
                print(
                    "Please run the script again and select 'Text-to-Image' or use --free."
                )
                sys.exit(1)
            current_step = 1

        elif current_step == 1:
            # Step 2: Select Model
            result = step_model(provider_obj)
            if result == BACK_OPTION:
                current_step = 0
                continue
            model_choice = result
            current_step = 2

        elif current_step == 2:
            # Step 3: Select Resolution
            result_res, result_key, result_width, result_height = step_resolution(
                provider_obj, model_choice, image_path
            )
            if result_res == BACK_OPTION:
                current_step = 1
                continue
            aspect_ratio = result_res
            res_key = result_key
            dim_width = result_width
            dim_height = result_height
            # Recraft-only sub-step (issue #9): the style picker rides with
            # the resolution step so the state-machine numbering stays
            # untouched for every other model. Go back re-runs resolution.
            if is_recraft_model(provider, model_choice):
                style_result = step_recraft_style(model_choice)
                if style_result is None:
                    print("Cancelled.")
                    return
                if style_result == BACK_OPTION:
                    continue
                recraft_style, recraft_colors = style_result
            current_step = 3

        elif current_step == 3:
            # Step 4: Select Quality
            result_quality, final_cost = step_quality(
                provider_obj, model_choice, res_key, dim_width, dim_height, image_path
            )
            if result_quality == BACK_OPTION:
                current_step = 2
                continue
            quality_key = result_quality
            current_step = 4

        elif current_step == 4:
            # Step 5: Select Prompt (skipped when supplied via --prompt-file / .txt)
            if cli_prompt is not None:
                final_prompt = cli_prompt
                current_step = 5
                continue
            result_prompt, prompt_selection = step_prompt(input_images, is_dual)
            if result_prompt == BACK_OPTION:
                current_step = 3
                continue
            final_prompt = result_prompt
            current_step = 5

        elif current_step == 5:
            # Step 6: Select variants (server-side `n`, 1..10).
            if provider == "OpenRouter":
                result_n = step_variants()
                if result_n is None:
                    # Ctrl+C / EOF — let the user out instead of silently
                    # defaulting to 1 variant.
                    print("Cancelled.")
                    return
                if result_n == BACK_OPTION:
                    # With cli_prompt the prompt step is auto-skipped, so
                    # BACK from variants jumps straight to quality.
                    current_step = 3 if cli_prompt is not None else 4
                    continue
                n_variants = result_n
            current_step = 6

        elif current_step == 6:
            # Step 7: Summary & Confirm (was Step 6)
            is_batch_mode = len(input_images) > 1 and not is_dual

            print("\n--- Summary ---")
            if input_images:
                if is_dual:
                    print(f"Images:     {', '.join(input_images)} (dual mode)")
                elif is_batch_mode:
                    print(f"Images:     {len(input_images)} images (batch mode)")
                else:
                    print(f"Image:      {input_images[0]}")
            else:
                print(f"Image:      None (Text-to-Image)")
            print(f"Provider:   {provider}")
            print(f"Model:      {model_choice}")
            if provider == "OpenAI" and model_choice == "gpt-image-2":
                print(f"Ratio:      {aspect_ratio}")
                if dim_width and dim_height:
                    print(f"Dimensions: {dim_width}x{dim_height}")
                else:
                    print(f"Dimensions: auto")
                print(f"Quality:    {quality_key}")
                if dim_width and dim_height:
                    tokens = calc_gpt_image2_tokens(dim_width, dim_height, quality_key)
                    cost_display = tokens * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000
                    print(f"Tokens:     ~{tokens:,} (${cost_display:.4f})")
                else:
                    print(f"Tokens:     depends on output size")
            elif provider == "OpenAI":
                print(f"Resolution: {res_key}")
                print(f"Quality:    {quality_key}")
            elif provider == "OpenRouter":
                print(f"Ratio:      {aspect_ratio}")
                print(f"Resolution: {quality_key}")
                print(f"Pixels:     {res_key}")
                if recraft_style:
                    print(f"Style:      {recraft_style}")
                    if recraft_colors:
                        print(f"Colors:     {', '.join(recraft_colors)}")
                if n_variants > 1:
                    print(f"Variants:   {n_variants}")
                # Pre-flight capability check (issue #11): mirrors the >4MP
                # BFL downscale warning style — surface descriptor
                # mismatches before money is spent on a doomed call.
                for warning in provider_obj.preflight_warnings(
                    model_choice, aspect_ratio, quality_key
                ):
                    print(f"⚠ Warning:  {warning}")
            else:
                print(f"Ratio:      {aspect_ratio}")
                print(f"Size:       {quality_key}")
            print(f"Prompt:     {final_prompt}")

            # Calculate total cost for batch mode
            input_cost = 0.0
            if (
                provider == "OpenRouter"
                and model_choice
                in [
                    "black-forest-labs/flux.2-flex",
                    "black-forest-labs/flux.2-pro",
                    "black-forest-labs/flux.2-max",
                ]
                and "input_mp_rate" in COSTS[model_choice]
            ):
                input_mp_rate = COSTS[model_choice]["input_mp_rate"]
                if image_path:
                    with Image.open(image_path) as img:
                        mp = (img.width * img.height) / 1_000_000
                    input_cost = mp * input_mp_rate
                elif is_batch_mode and input_images:
                    for inp_img in input_images:
                        with Image.open(inp_img) as img:
                            mp = (img.width * img.height) / 1_000_000
                        input_cost += mp * input_mp_rate

            # Flat per-input-image billing (e.g. Grok Imagine: $0.01 per
            # reference image regardless of size), as opposed to the
            # per-megapixel Flux rate handled above.
            if (
                provider == "OpenRouter"
                and input_images
                and "input_flat" in COSTS.get(model_choice, {})
            ):
                input_cost += COSTS[model_choice]["input_flat"] * len(input_images)

            total_cost = final_cost + input_cost

            if is_batch_mode:
                if provider == "OpenAI" and model_choice == "gpt-image-2":
                    print(f"Per Image:  varies by input size")
                    print(
                        f"Total Cost: varies by input size ({len(input_images)} images)"
                    )
                elif provider == "OpenAI":
                    total_cost = final_cost * len(input_images)
                    print(f"Per Image:  ${final_cost:.3f}")
                    print(f"Total Cost: ${total_cost:.3f} ({len(input_images)} images)")
                else:
                    # Only OpenRouter delivers `n` variants in one HTTP call.
                    # For Google/OVH the wizard forces n_variants=1 anyway;
                    # here we keep the cost math honest if someone calls
                    # directly with a GenerationRequest having n>1.
                    if provider == "OpenRouter":
                        per_call_cost = final_cost * n_variants + input_cost
                    else:
                        per_call_cost = final_cost + input_cost
                    total_cost = per_call_cost * len(input_images)
                    print(f"Per Input:  ${per_call_cost:.3f}")
                    print(
                        f"Total Cost: ${total_cost:.3f} "
                        f"({len(input_images)} inputs × {n_variants} variants)"
                    )
            else:
                # Single-input mode: only show the Output/Input breakdown
                # when there's an input cost (Flux models). A pure output
                # line next to Total would be a tautology.
                if provider == "OpenRouter":
                    output_cost = final_cost * n_variants
                else:
                    output_cost = final_cost
                if input_cost > 0:
                    print(f"Output Cost: ${output_cost:.3f}")
                    print(f"Input Cost:  ${input_cost:.3f}")
                print(f"Total Cost: ${output_cost + input_cost:.3f}")

            action = questionary.select(
                "What would you like to do?",
                choices=["Proceed with API call", "Edit Prompt", BACK_OPTION, "Cancel"],
                default="Proceed with API call",
            ).ask()

            if action == "Proceed with API call":
                break
            elif action == "Edit Prompt":
                new_prompt = multiline_prompt(
                    "Edit your prompt:", default=final_prompt or ""
                )
                if new_prompt:
                    final_prompt = new_prompt
                continue  # Will show summary again
            elif action == BACK_OPTION:
                # Confirm is now step 6; BACK goes to step 5 (variants), which
                # itself decides whether to skip the prompt step when
                # cli_prompt was provided.
                current_step = 5
                continue
            else:
                print("Cancelled.")
                return

    # CLI --iterations is the explicit override for n_variants. The wizard's
    # step_variants already populated n_variants during the interactive flow;
    # when --iterations is passed on the command line we honour it instead so
    # scripts retain their batch semantics.
    if args.iterations is not None:
        n_variants = max(1, args.iterations)

    extras = {}
    if is_recraft_model(provider, model_choice):
        from imgprompt.presets import recraft_default_style

        # Never silently empty: a skipped picker still pins the documented
        # default for the chosen variant.
        extras["style"] = recraft_style or recraft_default_style(model_choice)
        if recraft_colors:
            extras["colors"] = recraft_colors

    request = GenerationRequest(
        prompt=final_prompt,
        model=model_choice,
        aspect_ratio=aspect_ratio,
        res_key=res_key,
        quality_key=quality_key,
        images=input_images,
        width=dim_width,
        height=dim_height,
        n=n_variants,
        is_dual=is_dual,
        extras=extras,
    )

    # Persist the request so it can be replayed verbatim with --replay.
    save_last_generation(provider, request)

    # The OpenRouter provider now delivers all n_variants in a single API
    # call, and fan-out across input images is handled inside the provider
    # (one HTTP request per input). No client-side loop is needed; calling
    # provider.run() once executes the full plan.
    provider_obj.run(request)


if __name__ == "__main__":
    main()
