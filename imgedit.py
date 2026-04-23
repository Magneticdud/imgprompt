#!/usr/bin/env python3
import os
import sys
import shutil
import time
import math

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
from typing import Optional, List
import base64
import questionary
import requests
from datetime import datetime
import io
from PIL import Image
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from imgprompt.presets import (
    BACK_OPTION,
    COSTS,
    GEMINI_RESOLUTIONS,
    OPENROUTER_RESOLUTIONS,
    OPENROUTER_STANDARD_RATIOS,
    ASPECT_RATIO_VALUES,
    PRESET_PROMPTS_EDIT,
    PRESET_PROMPTS_GENERATE,
    PRESET_PROMPTS_DUAL,
    GPT_IMAGE_2_PRESET_CHOICES,
    GPT_IMAGE_2_PRICE_PER_MTOK,
    calc_gpt_image2_tokens,
    validate_gpt_image2_dims,
    auto_adjust_gpt_image2_dims,
)
from imgprompt.images import (
    get_images_in_cwd,
    process_image_for_api,
    get_closest_aspect_ratio,
    get_image_extension,
    save_api_image,
)
from imgprompt.session import SessionState
from imgprompt.providers.base import GenerationRequest
from imgprompt.providers.openai_provider import OpenAIProvider
from imgprompt.providers.google_provider import GoogleProvider
from imgprompt.providers.openrouter_provider import OpenRouterProvider
from imgprompt.providers.ovh_provider import OVHProvider


def select_inputs(provided_path: Optional[str]) -> List[str]:
    """Selects images either from arguments or from a list of files."""
    if provided_path:
        if os.path.isfile(provided_path):
            return [provided_path]
        else:
            print(f"Error: {provided_path} is not a valid file.")
            sys.exit(1)

    images = get_images_in_cwd()

    # Add option for Text-to-Image
    t2i_option = "Text-to-Image (No input image)"
    dual_option = "Two Images (Dual Input)"

    choices = [t2i_option]
    if len(images) >= 2:
        choices.append(dual_option)
    choices.extend(images)

    selected = questionary.select(
        "Select an image to edit or mode:", choices=choices
    ).ask()

    if not selected:
        sys.exit(0)

    if selected == t2i_option:
        return []

    if selected == dual_option:
        img1 = questionary.select("Select first image (IMG_1):", choices=images).ask()
        if not img1:
            sys.exit(0)

        # Remove selected image from options for the second one
        remaining_images = [img for img in images if img != img1]
        img2 = questionary.select(
            "Select second image (IMG_2):", choices=remaining_images
        ).ask()
        if not img2:
            sys.exit(0)

        return [img1, img2]

    return [selected]


def step_provider() -> str | None:
    """Step 1: Select provider. Returns provider name or BACK_OPTION."""
    provider = questionary.select(
        "Select Provider:",
        choices=[BACK_OPTION, "OpenAI", "Google", "OpenRouter", "OVH"],
        default="OpenAI",
    ).ask()
    if provider == BACK_OPTION or not provider:
        return BACK_OPTION
    return provider


def step_model(provider: str, current_model: str | None = None) -> str | None:
    """Step 2: Select model based on provider. Returns model name or BACK_OPTION."""
    if provider == "OpenAI":
        model_choices = ["gpt-image-2", "gpt-image-1.5", "gpt-image-1-mini"]
        default_model = "gpt-image-2"
    elif provider == "OpenRouter":
        model_choices = [
            "bytedance-seed/seedream-4.5",
            "black-forest-labs/flux.2-klein-4b",
            "black-forest-labs/flux.2-flex",
            "black-forest-labs/flux.2-pro",
            "black-forest-labs/flux.2-max",
            "sourceful/riverflow-v2-fast",
            "sourceful/riverflow-v2-pro",
            "google/gemini-2.5-flash-image",
            "google/gemini-3.1-flash-image-preview",
            "google/gemini-3-pro-image-preview",
        ]
        default_model = "bytedance-seed/seedream-4.5"
    elif provider == "OVH":
        model_choices = ["stabilityai/stable-diffusion-xl-base-1.0"]
        default_model = "stabilityai/stable-diffusion-xl-base-1.0"
    else:  # Google
        model_choices = [
            "gemini-2.5-flash-image",
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
        ]
        default_model = "gemini-2.5-flash-image"

    default_idx = model_choices.index(default_model)
    choices_with_back = [BACK_OPTION] + model_choices
    default_idx += 1  # Account for BACK_OPTION at index 0

    model = questionary.select(
        f"Select {provider} model:",
        choices=choices_with_back,
        default=choices_with_back[default_idx],
    ).ask()
    if model == BACK_OPTION or not model:
        return BACK_OPTION
    return model


def _prompt_custom_dims() -> tuple[int | None, int | None]:
    """Prompt for custom width/height with validation. Returns (width, height) or (None, None) on back."""
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
    provider: str, model: str, image_path: str | None
) -> tuple[str | None, str | None, int | None, int | None]:
    """Step 3: Select resolution/aspect ratio. Returns (selection, res_key, width, height) or (BACK_OPTION, None, None, None)."""
    choices_with_back = [BACK_OPTION]

    if provider == "OpenAI" and model == "gpt-image-2":
        preset_labels = [label for label, _, _, _, _ in GPT_IMAGE_2_PRESET_CHOICES]
        custom_label = "Custom dimensions"

        if image_path:
            with Image.open(image_path) as img:
                img_ratio = img.width / img.height
            best_idx = 0
            best_diff = float("inf")
            for i, (label, ratio_str, _, w, h) in enumerate(GPT_IMAGE_2_PRESET_CHOICES):
                parts = ratio_str.split(":")
                preset_ratio = int(parts[0]) / int(parts[1])
                diff = abs(img_ratio - preset_ratio)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            default_choice = preset_labels[best_idx]
        else:
            default_choice = preset_labels[0]

        resolution = questionary.select(
            "Select resolution:",
            choices=choices_with_back + preset_labels + [custom_label],
            default=default_choice,
        ).ask()
        if resolution == BACK_OPTION or not resolution:
            return BACK_OPTION, None, None, None

        if resolution == custom_label:
            width, height = _prompt_custom_dims()
            if width is None:
                return BACK_OPTION, None, None, None
            res_key = f"{width}x{height}"
            return f"Custom ({width}x{height})", res_key, width, height

        idx = preset_labels.index(resolution)
        _, ratio_str, size_key, width, height = GPT_IMAGE_2_PRESET_CHOICES[idx]
        res_key = f"{width}x{height}"
        return resolution, res_key, width, height

    elif provider == "OpenAI":
        res_options = [
            "1024x1024 (Square)",
            "1024x1536 (Vertical)",
            "1536x1024 (Horizontal)",
        ]
        if image_path:
            closest = get_closest_aspect_ratio(image_path, res_options)
        else:
            closest = res_options[0]

        resolution = questionary.select(
            "Select resolution:",
            choices=choices_with_back + res_options,
            default=closest,
        ).ask()
        if resolution == BACK_OPTION or not resolution:
            return BACK_OPTION, None, None, None
        res_key = resolution.split(" ")[0]
        return resolution, res_key, None, None

    elif provider == "OpenRouter":
        if model == "google/gemini-3.1-flash-image-preview":
            ratio_options = list(OPENROUTER_RESOLUTIONS.keys())
        else:
            ratio_options = OPENROUTER_STANDARD_RATIOS + ["21:9"]
        default_ratio = "1:1"

        if image_path:
            closest = get_closest_aspect_ratio(image_path, ratio_options)
        else:
            closest = default_ratio

        aspect_ratio = questionary.select(
            "Select aspect ratio:",
            choices=choices_with_back + ratio_options,
            default=closest,
        ).ask()
        if aspect_ratio == BACK_OPTION or not aspect_ratio:
            return BACK_OPTION, None, None, None
        res_key = OPENROUTER_RESOLUTIONS.get(aspect_ratio, "1024x1024")
        return aspect_ratio, res_key, None, None

    elif provider == "OVH":
        return "1:1", "1024x1024", None, None

    else:  # Google
        STANDARD_RATIOS = [
            "1:1",
            "2:3",
            "3:2",
            "3:4",
            "4:3",
            "4:5",
            "5:4",
            "9:16",
            "16:9",
            "21:9",
        ]

        if model == "gemini-3.1-flash-image-preview":
            ratio_options = ["Auto"] + list(GEMINI_RESOLUTIONS.keys())
            default_ratio = "Auto"
        elif model == "gemini-3-pro-image-preview":
            ratio_options = ["Auto"] + STANDARD_RATIOS
            default_ratio = "Auto"
        else:
            ratio_options = STANDARD_RATIOS
            default_ratio = "1:1"

        if image_path:
            closest = get_closest_aspect_ratio(image_path, ratio_options)
        else:
            closest = default_ratio

        aspect_ratio = questionary.select(
            "Select aspect ratio:",
            choices=choices_with_back + ratio_options,
            default=closest,
        ).ask()
        if aspect_ratio == BACK_OPTION or not aspect_ratio:
            return BACK_OPTION, None, None, None
        res_key = GEMINI_RESOLUTIONS.get(aspect_ratio, "Auto")
        return aspect_ratio, res_key, None, None


def step_quality(
    provider: str,
    model: str,
    res_key: str,
    width: int | None,
    height: int | None,
    image_path: str | None,
) -> tuple[str | None, float, int]:
    """Step 4: Select quality/size. Returns (quality_key, cost, input_pixels) or (BACK_OPTION, 0, 0)."""
    choices_with_back = [BACK_OPTION]

    if provider == "OpenAI" and model == "gpt-image-2":
        quality_choices = []
        for q in ["Low", "Medium", "High"]:
            tokens = calc_gpt_image2_tokens(width, height, q)
            cost = tokens * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000
            quality_choices.append(
                (q, tokens, cost, f"{q} (~{tokens:,} tokens, ${cost:.4f})")
            )

        quality_selected = questionary.select(
            "Select quality:",
            choices=choices_with_back + [c[3] for c in quality_choices],
            default=quality_choices[1][3],
        ).ask()
        if quality_selected == BACK_OPTION or not quality_selected:
            return BACK_OPTION, 0, 0

        quality_key = quality_selected.split(" ")[0]
        selected_entry = next(
            entry for entry in quality_choices if entry[3] == quality_selected
        )
        final_cost = selected_entry[2]
        return quality_key, final_cost, 0

    elif provider == "OpenAI":
        quality_choices = []
        for q in ["Low", "Medium", "High"]:
            cost = COSTS[model][q][res_key]
            quality_choices.append(f"{q} (${cost:.3f})")

        quality_selected = questionary.select(
            "Select quality:",
            choices=choices_with_back + quality_choices,
            default=quality_choices[1],
        ).ask()
        if quality_selected == BACK_OPTION or not quality_selected:
            return BACK_OPTION, 0, 0
        quality_key = quality_selected.split(" ")[0]
        final_cost = COSTS[model][quality_key][res_key]
        return quality_key, final_cost, 0

    elif provider == "OpenRouter":
        if model == "sourceful/riverflow-v2-pro":
            size_choices = []
            for s in ["1K", "2K", "4K"]:
                cost = COSTS[model][s]["fixed"]
                size_choices.append(f"{s} (${cost:.2f})")
        elif model.startswith("black-forest-labs/"):
            size_choices = []
            for s in ["1K", "2K"]:
                cost = COSTS[model][s]["fixed"]
                size_choices.append(f"{s} (${cost:.2f})")
        elif model in [
            "google/gemini-3-pro-image-preview",
            "google/gemini-3.1-flash-image-preview",
        ]:
            size_choices = []
            for s in ["1K", "2K", "4K"]:
                cost = COSTS[model][s]["fixed"]
                size_choices.append(f"{s} (${cost:.2f})")
        elif model in [
            "google/gemini-2.5-flash-image",
            "gemini-2.5-flash-image",
        ]:
            size_choices = []
            for s in ["1K"]:
                cost = COSTS[model][s]["fixed"]
                size_choices.append(f"{s} (${cost:.2f})")
        else:
            size_choices = []
            for s in ["1K", "2K"]:
                cost = COSTS[model][s]["fixed"]
                size_choices.append(f"{s} (${cost:.2f})")

        size_selected = questionary.select(
            "Select image size:",
            choices=choices_with_back + size_choices,
            default=size_choices[0],
        ).ask()
        if size_selected == BACK_OPTION or not size_selected:
            return BACK_OPTION, 0, 0
        quality_key = size_selected.split(" ")[0]
        final_cost = COSTS[model][quality_key]["fixed"]
        return quality_key, final_cost, 0

    elif provider == "OVH":
        return "1K", 0.0, 0

    else:  # Google
        if model in [
            "gemini-3-pro-image-preview",
            "gemini-3.1-flash-image-preview",
        ]:
            size_choices = []
            for s in ["1K", "2K", "4K"]:
                cost = COSTS[model][s]["fixed"]
                size_choices.append(f"{s} (${cost:.2f})")

            size_selected = questionary.select(
                "Select image size:",
                choices=choices_with_back + size_choices,
                default=size_choices[0],
            ).ask()
            if size_selected == BACK_OPTION or not size_selected:
                return BACK_OPTION, 0, 0
            quality_key = size_selected.split(" ")[0]
        else:
            quality_key = "1K"

        final_cost = COSTS[model][quality_key]["fixed"]
        return quality_key, final_cost, 0


def step_prompt(input_images: list, image_path: str | None) -> tuple[str | None, str]:
    """Step 5: Select prompt. Returns (final_prompt or BACK_OPTION, original_selection)."""
    is_batch_mode = len(input_images) > 1
    if is_batch_mode:
        prompt_list = PRESET_PROMPTS_EDIT
    elif len(input_images) > 1:
        prompt_list = PRESET_PROMPTS_DUAL
    elif image_path:
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
        final_prompt = questionary.text("Enter your custom prompt:").ask()
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
    args = parser.parse_args()

    # Determine if we have multiple images (batch mode)
    input_images = []
    if args.free:
        input_images = []
    elif args.images:
        # Expand glob patterns if any
        import glob

        all_images = []
        for pattern in args.images:
            # Check if it's a glob pattern
            if "*" in pattern or "?" in pattern:
                matched = glob.glob(pattern)
                all_images.extend(matched)
            elif os.path.isfile(pattern):
                all_images.append(pattern)
            elif os.path.isdir(pattern):
                # If it's a directory, get all images in it
                for f in os.listdir(pattern):
                    fpath = os.path.join(pattern, f)
                    if os.path.isfile(fpath) and f.lower().endswith(
                        (".pcx", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
                    ):
                        all_images.append(fpath)
            else:
                print(f"Warning: {pattern} is not a valid file or directory.")

        # Remove duplicates and sort
        input_images = sorted(list(set(all_images)))

        if not input_images:
            print("Error: No valid images found from provided arguments.")
            sys.exit(1)

        if len(input_images) > 1:
            print(f"\nBatch mode: {len(input_images)} images selected")
    else:
        input_images = select_inputs(None)

    # Legacy support variable
    image_path = input_images[0] if input_images else None

    if input_images:
        if len(input_images) == 1:
            print(f"\nSelected Image: {input_images[0]}")
        else:
            print(f"\nSelected Images: {', '.join(input_images)}")
    else:
        print(f"\nMode: Text-to-Image (No input image)")

    # State machine for navigation
    # step 0=provider, 1=model, 2=resolution, 3=quality, 4=prompt, 5=confirm
    current_step = 0

    # Initialize all variables
    provider = None
    model_choice = None
    aspect_ratio = None
    res_key = None
    quality_key = None
    final_cost = 0.0
    final_prompt = None
    dim_width = None
    dim_height = None

    while current_step >= 0:
        if current_step == 0:
            # Step 1: Select Provider
            result = step_provider()
            if result == BACK_OPTION:
                print("Cancelled.")
                return
            provider = result
            if provider == "OVH" and input_images:
                print("\nError: OVH provider only supports Text-to-Image mode.")
                print(
                    "Please run the script again and select 'Text-to-Image' or use --free."
                )
                sys.exit(1)
            current_step = 1

        elif current_step == 1:
            # Step 2: Select Model
            result = step_model(provider)
            if result == BACK_OPTION:
                current_step = 0
                continue
            model_choice = result
            current_step = 2

        elif current_step == 2:
            # Step 3: Select Resolution
            result_res, result_key, result_width, result_height = step_resolution(
                provider, model_choice, image_path
            )
            if result_res == BACK_OPTION:
                current_step = 1
                continue
            aspect_ratio = result_res
            res_key = result_key
            dim_width = result_width
            dim_height = result_height
            current_step = 3

        elif current_step == 3:
            # Step 4: Select Quality
            result_quality, final_cost, input_pixels = step_quality(
                provider, model_choice, res_key, dim_width, dim_height, image_path
            )
            if result_quality == BACK_OPTION:
                current_step = 2
                continue
            quality_key = result_quality
            current_step = 4

        elif current_step == 4:
            # Step 5: Select Prompt
            result_prompt, prompt_selection = step_prompt(input_images, image_path)
            if result_prompt == BACK_OPTION:
                current_step = 3
                continue
            final_prompt = result_prompt
            current_step = 5

        elif current_step == 5:
            # Step 6: Summary & Confirm
            is_batch_mode = len(input_images) > 1

            print("\n--- Summary ---")
            if input_images:
                if is_batch_mode:
                    print(f"Images:     {len(input_images)} images (batch mode)")
                else:
                    print(f"Images:     {', '.join(input_images)}")
            else:
                print(f"Image:      None (Text-to-Image)")
            print(f"Provider:   {provider}")
            print(f"Model:      {model_choice}")
            if provider == "OpenAI" and model_choice == "gpt-image-2":
                print(f"Ratio:      {aspect_ratio}")
                print(f"Dimensions: {dim_width}x{dim_height}")
                print(f"Quality:    {quality_key}")
                tokens = calc_gpt_image2_tokens(dim_width, dim_height, quality_key)
                cost_display = tokens * GPT_IMAGE_2_PRICE_PER_MTOK / 1_000_000
                print(f"Tokens:     ~{tokens:,} (${cost_display:.4f})")
            elif provider == "OpenAI":
                print(f"Resolution: {res_key}")
                print(f"Quality:    {quality_key}")
            elif provider == "OpenRouter":
                print(f"Ratio:      {aspect_ratio}")
                print(f"Resolution: {res_key}")
                print(f"Size:       {quality_key}")
            else:
                print(f"Ratio:      {aspect_ratio}")
                print(f"Size:       {quality_key}")
            print(f"Prompt:     {final_prompt}")

            # Calculate total cost for batch mode
            input_cost = 0
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
                    total_cost = (final_cost + input_cost) * len(input_images)
                    print(f"Per Image:  ${final_cost + input_cost:.3f}")
                    print(f"Total Cost: ${total_cost:.3f} ({len(input_images)} images)")
            else:
                if input_cost > 0:
                    print(f"Output Cost: ${final_cost:.3f}")
                    print(f"Input Cost:  ${input_cost:.3f}")
                print(f"Total Cost: ${final_cost + input_cost:.3f}")

            action = questionary.select(
                "What would you like to do?",
                choices=["Proceed with API call", "Edit Prompt", BACK_OPTION, "Cancel"],
                default="Proceed with API call",
            ).ask()

            if action == "Proceed with API call":
                break
            elif action == "Edit Prompt":
                new_prompt = questionary.text(
                    "Edit your prompt:", default=final_prompt
                ).ask()
                if new_prompt:
                    final_prompt = new_prompt
                continue  # Will show summary again
            elif action == BACK_OPTION:
                current_step = 4
                continue
            else:
                print("Cancelled.")
                return

    request = GenerationRequest(
        prompt=final_prompt,
        model=model_choice,
        aspect_ratio=aspect_ratio,
        res_key=res_key,
        quality_key=quality_key,
        images=input_images,
        width=dim_width,
        height=dim_height,
        input_pixels=input_pixels,
    )

    provider_map = {
        "OpenAI": OpenAIProvider,
        "Google": GoogleProvider,
        "OpenRouter": OpenRouterProvider,
        "OVH": OVHProvider,
    }
    provider_map[provider]().run(request)


if __name__ == "__main__":
    main()
