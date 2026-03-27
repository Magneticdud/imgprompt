#!/usr/bin/env python3
import os
import sys
import shutil
import time

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
)
from imgprompt.images import (
    get_images_in_cwd,
    process_image_for_api,
    get_closest_aspect_ratio,
    get_image_extension,
    save_api_image,
)
from imgprompt.session import SessionState


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
        model_choices = ["gpt-image-1.5", "gpt-image-1-mini"]
        default_model = "gpt-image-1.5"
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


def step_resolution(
    provider: str, model: str, image_path: str | None
) -> tuple[str | None, str | None]:
    """Step 3: Select resolution/aspect ratio. Returns (selection, res_key) or (BACK_OPTION, None)."""
    choices_with_back = [BACK_OPTION]

    if provider == "OpenAI":
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
            return BACK_OPTION, None
        res_key = resolution.split(" ")[0]
        return resolution, res_key

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
            return BACK_OPTION, None
        res_key = OPENROUTER_RESOLUTIONS.get(aspect_ratio, "1024x1024")
        return aspect_ratio, res_key

    elif provider == "OVH":
        return "1:1", "1024x1024"

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
            return BACK_OPTION, None
        res_key = GEMINI_RESOLUTIONS.get(aspect_ratio, "Auto")
        return aspect_ratio, res_key


def step_quality(provider: str, model: str, res_key: str) -> tuple[str | None, float]:
    """Step 4: Select quality/size. Returns (quality_key, cost) or (BACK_OPTION, 0)."""
    choices_with_back = [BACK_OPTION]

    if provider == "OpenAI":
        quality_choices = []
        for q in ["Low", "Medium", "High"]:
            cost = COSTS[model][q][res_key]
            quality_choices.append(f"{q} (${cost:.3f})")

        quality_selected = questionary.select(
            "Select quality:",
            choices=choices_with_back + quality_choices,
            default=quality_choices[1],  # Default to Medium
        ).ask()
        if quality_selected == BACK_OPTION or not quality_selected:
            return BACK_OPTION, 0
        quality_key = quality_selected.split(" ")[0]
        final_cost = COSTS[model][quality_key][res_key]
        return quality_key, final_cost

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
            return BACK_OPTION, 0
        quality_key = size_selected.split(" ")[0]
        final_cost = COSTS[model][quality_key]["fixed"]
        return quality_key, final_cost

    elif provider == "OVH":
        return "1K", 0.0

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
                return BACK_OPTION, 0
            quality_key = size_selected.split(" ")[0]
        else:
            quality_key = "1K"

        final_cost = COSTS[model][quality_key]["fixed"]
        return quality_key, final_cost


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
            result_res, result_key = step_resolution(provider, model_choice, image_path)
            if result_res == BACK_OPTION:
                current_step = 1
                continue
            aspect_ratio = result_res
            res_key = result_key
            current_step = 3

        elif current_step == 3:
            # Step 4: Select Quality
            result_quality, final_cost = step_quality(provider, model_choice, res_key)
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
            if provider == "OpenAI":
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
                if provider == "OpenAI":
                    total_cost = final_cost * len(input_images)
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

    is_batch_mode = len(input_images) > 1

    # 7. API Call
    if provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
            sys.exit(1)

        client_openai = OpenAI(api_key=api_key)

        if is_batch_mode:
            # BATCH MODE: Process each image separately
            print(f"\nStarting batch processing: {len(input_images)} images...")
            success_count = 0
            fail_count = 0

            for idx, img_path in enumerate(input_images, 1):
                print(
                    f"\n--- Processing image {idx}/{len(input_images)}: {os.path.basename(img_path)} ---"
                )
                try:
                    image_input = process_image_for_api(img_path, res_key)

                    response = client_openai.images.edit(
                        model=model_choice,
                        image=image_input,
                        prompt=final_prompt,
                        n=1,
                        size=res_key,
                        quality=quality_key.lower(),
                    )

                    # Handle Response
                    image_url = None
                    image_b64 = None
                    if hasattr(response, "data") and len(response.data) > 0:
                        image_url = getattr(response.data[0], "url", None)
                        image_b64 = getattr(response.data[0], "b64_json", None)

                    if image_url or image_b64:
                        save_api_image(image_url, image_b64, img_path)
                        success_count += 1
                    else:
                        print(
                            f"Error: Could not retrieve image data from the API response."
                        )
                        fail_count += 1

                except Exception as e:
                    error_msg = str(e)
                    if "moderation_blocked" in error_msg:
                        print(
                            f">> The request was rejected by the OpenAI safety system."
                        )
                    else:
                        print(f"An error occurred during OpenAI call: {e}")
                    fail_count += 1

            print(
                f"\n=== Batch complete: {success_count} succeeded, {fail_count} failed ==="
            )

        elif input_images:
            # Single image or dual mode (original logic)
            print(f"\nSending request to OpenAI ({model_choice})...")
            try:
                # EDIT MODE
                if len(input_images) == 1:
                    image_input = process_image_for_api(input_images[0], res_key)
                else:
                    image_input = [
                        process_image_for_api(p, res_key) for p in input_images
                    ]

                response = client_openai.images.edit(
                    model=model_choice,
                    image=image_input,
                    prompt=final_prompt,
                    n=1,
                    size=res_key,
                    quality=quality_key.lower(),
                )
                # Handle Response (same as before)
                image_url = None
                image_b64 = None
                if hasattr(response, "data") and len(response.data) > 0:
                    image_url = getattr(response.data[0], "url", None)
                    image_b64 = getattr(response.data[0], "b64_json", None)

                if image_url or image_b64:
                    save_api_image(image_url, image_b64, image_path)
                else:
                    print(
                        "\nError: Could not retrieve image data from the API response."
                    )
                    # print(f"Debug Response: {response}")

            except Exception as e:
                # Check for OpenAI moderation blocked error
                error_msg = str(e)
                if "moderation_blocked" in error_msg:
                    print("\n>> The request was rejected by the OpenAI safety system.")
                    print(
                        ">> This typically happens with images of famous people, children, or NSFW content."
                    )
                else:
                    print(f"\nAn error occurred during OpenAI call: {e}")

    elif provider == "OpenRouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print(
                "Error: OPENROUTER_API_KEY not found. Please set it in your .env file."
            )
            sys.exit(1)

        client_openrouter = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        def _img_to_data_url(img_path: str) -> str:
            """Reads an image file, compresses if needed to stay under 4.5MB, and returns a base64 data URL."""
            MAX_REQUEST_SIZE = 4.5 * 1024 * 1024  # 4.5MB in bytes
            MAX_MP = 4  # black-forest-labs models resize images >4MP anyway
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            ext = os.path.splitext(img_path)[1].lower()
            mime = mime_types.get(ext, "image/png")

            # Downscale large images for black-forest-labs models (they resize to 4MP anyway)
            use_resized = False
            resized_data = None
            if model_choice.startswith("black-forest-labs/"):
                with Image.open(img_path) as img:
                    mp = (img.width * img.height) / 1_000_000
                    if mp > MAX_MP:
                        print(
                            f"Image {os.path.basename(img_path)} is {mp:.1f}MP, downscaling to 4MP for {model_choice}..."
                        )
                        scale = (MAX_MP / mp) ** 0.5
                        new_width = int(img.width * scale)
                        new_height = int(img.height * scale)
                        img = img.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )
                        output = io.BytesIO()
                        img.save(output, format="JPEG", quality=95)
                        resized_data = output.getvalue()
                        use_resized = True
                        mime = "image/jpeg"

            # First, check original file size
            if use_resized and resized_data:
                original_data = resized_data
            else:
                with open(img_path, "rb") as f:
                    original_data = f.read()

            if len(original_data) <= MAX_REQUEST_SIZE:
                print(
                    f"Image {os.path.basename(img_path)} is {len(original_data)/1024:.1f}KB, within 4.5MB limit."
                )
                b64 = base64.b64encode(original_data).decode()
                return f"data:{mime};base64,{b64}"

            # If too large, compress/resize
            print(
                f"Image {os.path.basename(img_path)} is {len(original_data)/1024:.1f}KB, exceeding 4.5MB limit. Compressing..."
            )

            with Image.open(img_path) as img:
                # Try compressing first
                output = io.BytesIO()
                quality = 85
                img.save(
                    output, format=img.format if img.format else "JPEG", quality=quality
                )
                compressed_data = output.getvalue()

                # If still too large, resize
                if len(compressed_data) > MAX_REQUEST_SIZE:
                    print("Compression not enough, resizing...")
                    # Calculate target size to get under limit (estimate)
                    scale_factor = (MAX_REQUEST_SIZE / len(compressed_data)) ** 0.5
                    new_width = int(img.width * scale_factor * 0.9)  # Add 10% buffer
                    new_height = int(img.height * scale_factor * 0.9)

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    output = io.BytesIO()
                    img.save(
                        output,
                        format=img.format if img.format else "JPEG",
                        quality=quality,
                    )
                    compressed_data = output.getvalue()

                # If still too large, try lower quality
                if len(compressed_data) > MAX_REQUEST_SIZE:
                    print("Resizing not enough, reducing quality...")
                    output = io.BytesIO()
                    img.save(
                        output, format=img.format if img.format else "JPEG", quality=50
                    )
                    compressed_data = output.getvalue()

                b64 = base64.b64encode(compressed_data).decode()
                print(f"Compressed image to {len(compressed_data)/1024:.1f}KB")

                return f"data:{mime};base64,{b64}"

        def _call_openrouter(
            prompt_text: str, img_paths: list[str] | None = None
        ) -> str | None:
            """
            Calls OpenRouter image generation and returns a base64 data URL or None.
            If img_paths is provided, images are included as multimodal content.
            """
            image_config = {"aspect_ratio": aspect_ratio}
            if quality_key in ["2K", "4K"]:
                image_config["image_size"] = quality_key

            # Build message content
            if img_paths:
                content = []
                for p in img_paths:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": _img_to_data_url(p)},
                        }
                    )
                content.append({"type": "text", "text": prompt_text})
            else:
                content = prompt_text

            resp = client_openrouter.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": content}],
                extra_body={
                    "modalities": ["image"],
                    "image_config": image_config,
                },
            )

            if not resp.choices:
                print("\nError: No choices returned in API response")
                print(f"Debug - Full response: {resp}")
                return None

            msg = resp.choices[0].message
            images = getattr(msg, "images", None)
            if images:
                return images[0]["image_url"]["url"]
            return None

        def _save_openrouter_image(data_url: str, original_path: str | None):
            """Saves a base64 data URL image to disk."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if "," in data_url:
                header, b64_data = data_url.split(",", 1)
            else:
                header = ""
                b64_data = data_url

            img_bytes = base64.b64decode(b64_data)
            ext = get_image_extension(img_bytes)

            if original_path:
                base_name = os.path.splitext(os.path.basename(original_path))[0]
                output_dir = os.path.dirname(original_path)
                if not output_dir:
                    output_dir = "."
                filename = f"edited_{timestamp}_{base_name}{ext}"
                output_path = os.path.join(output_dir, filename)
            else:
                filename = f"generated_{timestamp}{ext}"
                output_path = filename

            with open(output_path, "wb") as f:
                f.write(img_bytes)
            print(f"\nSuccess! File saved successfully as {output_path}")

        if is_batch_mode:
            print(f"\nStarting batch processing: {len(input_images)} images...")
            success_count = 0
            fail_count = 0
            for idx, img_path in enumerate(input_images, 1):
                print(
                    f"\n--- Processing image {idx}/{len(input_images)}: {os.path.basename(img_path)} ---"
                )
                try:
                    data_url = _call_openrouter(final_prompt, img_paths=[img_path])
                    if data_url:
                        _save_openrouter_image(data_url, img_path)
                        success_count += 1
                    else:
                        print("Error: No image returned by OpenRouter API.")
                        fail_count += 1
                except Exception as e:
                    print(f"An error occurred during OpenRouter call: {e}")
                    fail_count += 1
            print(
                f"\n=== Batch complete: {success_count} succeeded, {fail_count} failed ==="
            )
        else:
            print(f"\nSending request to OpenRouter ({model_choice})...")
            try:
                img_paths = input_images if input_images else None
                data_url = _call_openrouter(final_prompt, img_paths=img_paths)
                if data_url:
                    _save_openrouter_image(data_url, image_path)
                else:
                    print("\nError: No image returned by OpenRouter API.")
            except Exception as e:
                print(f"\nAn error occurred during OpenRouter call: {e}")

    elif provider == "OVH":
        api_key = os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")

        # Determine if we should send an authorization header
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Check if the token is valid (not empty and not a placeholder)
        is_placeholder = api_key and api_key.startswith("your_")
        if api_key and not is_placeholder:
            headers["Authorization"] = f"Bearer {api_key}"
            print("Using authenticated OVH access (400 rpm).")
        else:
            print("Using anonymous OVH access (2 rpm).")

        url = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/images/generations"
        payload = {
            "model": model_choice,
            "prompt": final_prompt,
            "size": res_key,
            "response_format": "b64_json",
        }

        print(f"\nSending request to OVH ({model_choice})...")
        try:
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                resp_json = response.json()
                image_b64 = None
                if "data" in resp_json and len(resp_json["data"]) > 0:
                    image_b64 = resp_json["data"][0].get("b64_json")

                if image_b64:
                    save_api_image(None, image_b64, None)
                else:
                    print(
                        "\\nError: Could not retrieve image data from the OVH API response."
                    )
            else:
                print(f"Error: {response.status_code} {response.text}")

        except Exception as e:
            print(f"\\nAn error occurred during OVH call: {e}")

    else:
        # Google Provider
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
            sys.exit(1)

        client_google = genai.Client(api_key=api_key)

        # Prepare Google Config (same for all images in batch mode)
        config_args = {}
        if aspect_ratio != "Auto":
            config_args["aspect_ratio"] = aspect_ratio

        if model_choice in [
            "gemini-3-pro-image-preview",
            "gemini-3.1-flash-image-preview",
        ]:
            config_args["image_size"] = quality_key

        if is_batch_mode:
            # BATCH MODE: Process each image separately
            print(f"\nStarting batch processing: {len(input_images)} images...")
            success_count = 0
            fail_count = 0

            MAX_RETRIES = 10
            BASE_DELAY = 5

            for idx, img_path in enumerate(input_images, 1):
                print(
                    f"\n--- Processing image {idx}/{len(input_images)}: {os.path.basename(img_path)} ---"
                )
                saved = False
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        req_contents = [final_prompt]
                        img_contexts = []

                        # Single image for this iteration
                        img = Image.open(img_path)
                        img_contexts.append(img)
                        req_contents.append(img)

                        config_kwargs = {
                            "image_config": types.ImageConfig(**config_args),
                        }

                        # Only gemini-3 models support response_modalities and thinking_config
                        if "gemini-3.1" in model_choice:
                            config_kwargs["response_modalities"] = ["IMAGE"]
                            config_kwargs["thinking_config"] = types.ThinkingConfig(
                                thinking_level="MINIMAL"
                            )

                        response = client_google.models.generate_content(
                            model=model_choice,
                            contents=req_contents,
                            config=types.GenerateContentConfig(**config_kwargs),
                        )

                        # Close images
                        for img in img_contexts:
                            img.close()

                        # Handle Response
                        if hasattr(response, "parts") and response.parts:
                            for part in response.parts:
                                if part.inline_data is not None:
                                    generated_image = part.as_image()
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    base_name = os.path.splitext(
                                        os.path.basename(img_path)
                                    )[0]
                                    output_dir = os.path.dirname(img_path)
                                    if not output_dir:
                                        output_dir = "."

                                    temp_filename = os.path.join(
                                        output_dir,
                                        f"edited_{timestamp}_{base_name}_temp.png",
                                    )
                                    generated_image.save(temp_filename)

                                    with open(temp_filename, "rb") as f:
                                        img_data = f.read()
                                    ext = get_image_extension(img_data)
                                    filename = f"edited_{timestamp}_{base_name}{ext}"
                                    output_path = os.path.join(output_dir, filename)

                                    os.replace(temp_filename, output_path)
                                    print(
                                        f"Success! File saved successfully as {output_path}"
                                    )
                                    saved = True
                                    success_count += 1
                                    break

                        if saved:
                            break  # exit retry loop on success

                        if not saved:
                            print("Error: No image found in Google API response.")
                            # Check for errors
                            if (
                                hasattr(response, "prompt_feedback")
                                and response.prompt_feedback
                            ):
                                if hasattr(response.prompt_feedback, "block_reason"):
                                    print(
                                        f"Prompt Feedback Block Reason: {response.prompt_feedback.block_reason}"
                                    )
                            fail_count += 1

                        break  # non-retryable: API succeeded but returned no image

                    except Exception as e:
                        error_str = str(e)
                        if (
                            "503" in error_str
                            or "500" in error_str
                            or "UNAVAILABLE" in error_str
                            or "INTERNAL" in error_str
                            or "high demand" in error_str.lower()
                        ):
                            if attempt < MAX_RETRIES:
                                delay = min(BASE_DELAY * (2 ** (attempt - 1)), 60)
                                print(
                                    f"\n[Retry {attempt}/{MAX_RETRIES}] Server error, waiting {delay}s..."
                                )
                                time.sleep(delay)
                                continue
                            else:
                                print(f"\nMax retries reached. Last error: {e}")
                                fail_count += 1
                                break
                        else:
                            print(f"An error occurred during Google call: {e}")
                            fail_count += 1
                            break

            print(
                f"\n=== Batch complete: {success_count} succeeded, {fail_count} failed ==="
            )

        else:
            # Single image or dual mode (original logic)
            print(f"\nSending request to Google ({model_choice})...")

            MAX_RETRIES = 5
            BASE_DELAY = 2

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    req_contents = [final_prompt]
                    img_contexts = []

                    if input_images:
                        if len(input_images) == 1:
                            img = Image.open(input_images[0])
                            img_contexts.append(img)
                            req_contents.append(img)
                        else:
                            # Dual mode
                            req_contents.append("IMG_1:")
                            img1 = Image.open(input_images[0])
                            img_contexts.append(img1)
                            req_contents.append(img1)

                            req_contents.append("IMG_2:")
                            img2 = Image.open(input_images[1])
                            img_contexts.append(img2)
                            req_contents.append(img2)

                    config_kwargs = {
                        "image_config": types.ImageConfig(**config_args),
                    }

                    # Only gemini-3 models support response_modalities and thinking_config
                    if "gemini-3.1" in model_choice:
                        config_kwargs["response_modalities"] = ["IMAGE"]
                        config_kwargs["thinking_config"] = types.ThinkingConfig(
                            thinking_level="MINIMAL"
                        )

                    response = client_google.models.generate_content(
                        model=model_choice,
                        contents=req_contents,
                        config=types.GenerateContentConfig(**config_kwargs),
                    )

                    # Close images
                    for img in img_contexts:
                        img.close()

                    # Handle Response
                    saved = False
                    if hasattr(response, "parts") and response.parts:
                        for part in response.parts:
                            if part.inline_data is not None:
                                generated_image = part.as_image()
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                                if image_path:
                                    base_name = os.path.splitext(
                                        os.path.basename(image_path)
                                    )[0]
                                    output_dir = os.path.dirname(image_path)
                                    if not output_dir:
                                        output_dir = "."
                                    temp_filename = os.path.join(
                                        output_dir,
                                        f"edited_{timestamp}_{base_name}_temp.png",
                                    )
                                else:
                                    output_dir = "."
                                    temp_filename = f"generated_{timestamp}_temp.png"

                                generated_image.save(temp_filename)

                                with open(temp_filename, "rb") as f:
                                    img_data = f.read()
                                ext = get_image_extension(img_data)

                                if image_path:
                                    filename = f"edited_{timestamp}_{base_name}{ext}"
                                else:
                                    filename = f"generated_{timestamp}{ext}"
                                output_path = os.path.join(output_dir, filename)

                                os.replace(temp_filename, output_path)
                                print(
                                    f"\nSuccess! File saved successfully as {output_path}"
                                )
                                saved = True
                            elif part.text:
                                print(f"Response text: {part.text}")

                    if not saved:
                        print("\nError: No image found in Google API response.")

                        # Check for prompt feedback blocks
                        if (
                            hasattr(response, "prompt_feedback")
                            and response.prompt_feedback
                        ):
                            if hasattr(response.prompt_feedback, "block_reason"):
                                print(
                                    f"Prompt Feedback Block Reason: {response.prompt_feedback.block_reason}"
                                )
                                reason_str = str(response.prompt_feedback.block_reason)
                                if any(
                                    x in reason_str
                                    for x in ["SAFETY", "BLOCK", "OTHER"]
                                ):
                                    print(
                                        ">> The request was likely blocked due to safety settings or policy violations."
                                    )

                        # Check for safety blocks or other finish reasons
                        if hasattr(response, "candidates") and response.candidates:
                            for i, candidate in enumerate(response.candidates):
                                if hasattr(candidate, "finish_reason"):
                                    print(
                                        f"Candidate {i+1} Finish Reason: {candidate.finish_reason}"
                                    )
                                    reason_str = str(candidate.finish_reason)
                                    if any(
                                        x in reason_str
                                        for x in ["SAFETY", "BLOCK", "OTHER"]
                                    ):
                                        print(
                                            ">> The request was likely blocked due to safety settings or policy violations."
                                        )

                        if hasattr(response, "parts") and response.parts:
                            for part in response.parts:
                                if part.text:
                                    print(f"Response text: {part.text}")

                    break

                except Exception as e:
                    error_str = str(e)
                    if (
                        "503" in error_str
                        or "500" in error_str
                        or "UNAVAILABLE" in error_str
                        or "INTERNAL" in error_str
                        or "high demand" in error_str.lower()
                    ):
                        if attempt < MAX_RETRIES:
                            delay = min(BASE_DELAY * (2 ** (attempt - 1)), 60)
                            print(
                                f"\n[Retry {attempt}/{MAX_RETRIES}] Server error, waiting {delay}s..."
                            )
                            import time

                            time.sleep(delay)
                            continue
                        else:
                            print(f"\nMax retries reached. Last error: {e}")
                            break
                    else:
                        print(f"\nAn error occurred during Google call: {e}")
                        break


if __name__ == "__main__":
    main()
