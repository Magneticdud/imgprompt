import os
import sys
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

# Constants for pricing
# costs are in USD and are not accurate, they are just for reference
COSTS = {
    "gpt-image-1.5": {
        "Low": {"1024x1024": 0.06, "1024x1536": 0.07, "1536x1024": 0.07},
        "Medium": {"1024x1024": 0.07, "1024x1536": 0.11, "1536x1024": 0.11},
        "High": {"1024x1024": 0.133, "1024x1536": 0.26, "1536x1024": 0.26},
    },
    "gpt-image-1-mini": {
        "Low": {"1024x1024": 0.01, "1024x1536": 0.007, "1536x1024": 0.007},
        "Medium": {"1024x1024": 0.011, "1024x1536": 0.016, "1536x1024": 0.016},
        "High": {"1024x1024": 0.036, "1024x1536": 0.054, "1536x1024": 0.054},
    },
    "gemini-2.5-flash-image": {
        "1K": {"fixed": 0.04},  # Fixed price per image
    },
    "gemini-3-pro-image-preview": {
        "1K": {"fixed": 0.14},
        "2K": {"fixed": 0.14},
        "4K": {"fixed": 0.25},
    },
}

GEMINI_RESOLUTIONS = {
    "1:1": "1024x1024",
    "2:3": "832x1248",
    "3:2": "1248x832",
    "3:4": "864x1184",
    "4:3": "1184x864",
    "4:5": "896x1152",
    "5:4": "1152x896",
    "9:16": "768x1344",
    "16:9": "1344x768",
    "21:9": "1536x672",
}

# Mapping of aspect ratio strings to their float values for comparison
ASPECT_RATIO_VALUES = {
    "1:1": 1.0,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "3:4": 3 / 4,
    "4:3": 4 / 3,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
    "1024x1024 (Square)": 1.0,
    "1024x1536 (Vertical)": 1024 / 1536,
    "1536x1024 (Horizontal)": 1536 / 1024,
}

PRESET_PROMPTS_EDIT = [
    "Outpaint the provided image, maintain all existing details. Preserve the exact composition and identity.",
    "The quality of this logo is poor, recreate it faithfully as if it were vector-based, with sharp edges and limited colors.",
    "Upscale this photo 4x. Preserve the exact composition and identity. Remove JPEG artifacts and noise, enhance real details only. Do not add or remove objects. Do not change facial features. Do not hallucinate text or logos; if unreadable, keep it unreadable. High-resolution output.",
    "Object Removal (High Quality)",
    "Convert this photo into a classic oil painting style.",
    "Add a realistic sunset lighting to this landscape.",
    "Remove the background and replace it with a clean minimalist studio setting.",
    "Photorealistic restoration. Strictly preserve geometry and identity. No creative reinterpretation. No new details beyond what is implied by the pixels.",
    "Change the season of this photo to winter, adding snow and frost.",
    "Give this portrait a 1950s vintage film look.",
    "Modify the colors to follow a warm autumnal palette.",
    "Custom Prompt",
]

PRESET_PROMPTS_GENERATE = [
    "A futuristic cyberpunk cityscape at night, neon lights, rain, high detail.",
    "A cute minimalist vector logo of a fox.",
    "A photorealistic portrait of an astronaut on Mars, cinematic lighting.",
    "Abstract geometric patterns, vibrant colors, 3d render style.",
    "A serene japanese garden with cherry blossoms, watercolor style.",
    "Isometric view of a cozy coffee shop interior.",
    "A retro-style BW lettering with thick outline",
    "1990s Memphis Style Logo",
    "Custom Prompt",
]


def get_images_in_cwd() -> List[str]:
    """Returns a list of image files in the current working directory."""
    extensions = (".pcx", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    return [f for f in os.listdir(".") if f.lower().endswith(extensions)]


def select_image(provided_path: Optional[str]) -> Optional[str]:
    """Selects an image either from arguments or from a list of files."""
    if provided_path:
        if os.path.isfile(provided_path):
            return provided_path
        else:
            print(f"Error: {provided_path} is not a valid file.")
            sys.exit(1)

    images = get_images_in_cwd()

    # Add option for Text-to-Image
    t2i_option = "Text-to-Image (No input image)"
    choices = [t2i_option] + images

    selected = questionary.select(
        "Select an image to edit or mode:", choices=choices
    ).ask()

    if not selected:
        sys.exit(0)

    if selected == t2i_option:
        return None

    return selected


def process_image_for_api(image_path: str, target_res: str) -> tuple:
    """
    Checks if the image needs resizing and returns a tuple (filename, data, mime_type).
    If the image is larger than the target resolution in any dimension, it is resized.
    """
    # Parse target resolution
    target_width, target_height = map(int, target_res.split("x"))
    filename = os.path.basename(image_path)

    # Map extensions to MIME types
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = mime_types.get(ext, "image/png")  # Default to png if unknown

    with Image.open(image_path) as img:
        original_width, original_height = img.size

        # Check if resizing is needed
        if original_width > target_width or original_height > target_height:
            print(
                f"Resizing input image from {original_width}x{original_height} to fit within {target_width}x{target_height}..."
            )
            # Use thumbnail to maintain aspect ratio while fitting within bounds
            img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

            output = io.BytesIO()
            # Determine format from original file extension
            fmt = img.format if img.format else "PNG"
            if ext in (".jpg", ".jpeg"):
                fmt = "JPEG"
                mime_type = "image/jpeg"

            img.save(output, format=fmt)
            output.seek(0)
            return (filename, output, mime_type)
        else:
            # Return original file content as BytesIO
            print(
                f"Input image {original_width}x{original_height} is within limits. Sending untouched."
            )
            with open(image_path, "rb") as f:
                return (filename, io.BytesIO(f.read()), mime_type)


def get_closest_aspect_ratio(image_path: str, supported_ratios: List[str]) -> str:
    """Calculates the aspect ratio of the image and returns the closest supported ratio."""
    with Image.open(image_path) as img:
        w, h = img.size
        img_ratio = w / h

    closest_ratio = supported_ratios[0]
    min_diff = float("inf")

    for ratio in supported_ratios:
        ratio_val = ASPECT_RATIO_VALUES.get(ratio)
        if ratio_val is not None:
            diff = abs(img_ratio - ratio_val)
            if diff < min_diff:
                min_diff = diff
                closest_ratio = ratio

    return closest_ratio


def main():
    parser = argparse.ArgumentParser(description="GPT-Image-1.5 POC Image Editor")
    parser.add_argument("image", nargs="?", help="Path to the image to edit")
    parser.add_argument(
        "--free",
        action="store_true",
        help="Start in Text-to-Image mode (no base image)",
    )
    args = parser.parse_args()

    # 1. Select Image
    if args.free:
        image_path = None
    else:
        image_path = select_image(args.image)

    if image_path:
        print(f"\nSelected Image: {image_path}")
    else:
        print(f"\nMode: Text-to-Image (No input image)")

    # 2. Select Provider
    provider = questionary.select(
        "Select Provider:", choices=["OpenAI", "Google"], default="OpenAI"
    ).ask()
    if not provider:
        sys.exit(0)

    # 3. Select Model
    if provider == "OpenAI":
        model_choices = ["gpt-image-1.5", "gpt-image-1-mini"]
        default_model = "gpt-image-1.5"
    else:
        model_choices = ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]
        default_model = "gemini-2.5-flash-image"

    model_choice = questionary.select(
        f"Select {provider} model:",
        choices=model_choices,
        default=default_model,
    ).ask()
    if not model_choice:
        sys.exit(0)

    # 4. Select Resolution / Aspect Ratio
    if provider == "OpenAI":
        res_options = [
            "1024x1024 (Square)",
            "1024x1536 (Vertical)",
            "1536x1024 (Horizontal)",
        ]
        if image_path:
            closest = get_closest_aspect_ratio(image_path, res_options)
        else:
            closest = res_options[0]  # Default Square

        resolution = questionary.select(
            "Select resolution:", choices=res_options, default=closest
        ).ask()
        if not resolution:
            sys.exit(0)
        res_key = resolution.split(" ")[0]
    else:
        # Google uses aspect ratio
        # Note: only gemini-3-pro-image-preview supports "Auto" aspect ratio.
        if model_choice == "gemini-3-pro-image-preview":
            ratio_options = ["Auto"] + list(GEMINI_RESOLUTIONS.keys())
            default_ratio = "Auto"
        else:
            ratio_options = list(GEMINI_RESOLUTIONS.keys())
            default_ratio = "1:1"

        if image_path:
            closest = get_closest_aspect_ratio(image_path, list(GEMINI_RESOLUTIONS.keys()))
        else:
            closest = default_ratio

        aspect_ratio = questionary.select(
            "Select aspect ratio:", choices=ratio_options, default=closest
        ).ask()
        if not aspect_ratio:
            sys.exit(0)
        res_key = GEMINI_RESOLUTIONS.get(aspect_ratio, "Auto")

    # 5. Select Quality / Size and show costs
    quality_key = "1K"  # Default for Google
    if provider == "OpenAI":
        quality_choices = []
        for q in ["Low", "Medium", "High"]:
            cost = COSTS[model_choice][q][res_key]
            quality_choices.append(f"{q} (${cost:.3f})")

        quality_selected = questionary.select(
            "Select quality:", choices=quality_choices
        ).ask()
        if not quality_selected:
            sys.exit(0)
        quality_key = quality_selected.split(" ")[0]
        final_cost = COSTS[model_choice][quality_key][res_key]
    else:
        # Google quality/size selection
        if model_choice == "gemini-3-pro-image-preview":
            size_choices = []
            for s in ["1K", "2K", "4K"]:
                cost = COSTS[model_choice][s]["fixed"]
                size_choices.append(f"{s} (${cost:.2f})")

            size_selected = questionary.select(
                "Select image size:", choices=size_choices, default=size_choices[0]
            ).ask()
            if not size_selected:
                sys.exit(0)
            quality_key = size_selected.split(" ")[0]
        else:
            quality_key = "1K"

        final_cost = COSTS[model_choice][quality_key]["fixed"]

    # 6. Select Prompt
    if image_path:
        prompt_choices = PRESET_PROMPTS_EDIT
    else:
        prompt_choices = PRESET_PROMPTS_GENERATE

    prompt_selection = questionary.select(
        "Select a prompt or enter a custom one:", choices=prompt_choices
    ).ask()
    if not prompt_selection:
        sys.exit(0)

    final_prompt = prompt_selection
    if prompt_selection == "Custom Prompt":
        final_prompt = questionary.text("Enter your custom prompt:").ask()
        if not final_prompt:
            sys.exit(0)
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
            sys.exit(0)
        final_prompt = (
            f'Create a clean vector-style black and white typographic logo. '
            f'Text: "{text_input}" on two lines, centered and slightly slanted upward to the right. '
            f'Use bold retro script lettering (smooth connected cursive, thick strokes), white fill with a thick black outline. '
            f'Add a large black drop shadow offset down-right to create a strong 3D sticker effect. '
            f'Add a swoosh underline under the second word, also white with black outline and black shadow. '
            f'High contrast, crisp edges, no textures, no gradients, bright green background (it will be keyed out), no extra elements. '
            f'Export as a logo/wordmark.'
        )
    elif prompt_selection == "1990s Memphis Style Logo":
        text_input = questionary.text("What text to write?").ask()
        if not text_input:
            print("Error: Text input is required for this preset.")
            sys.exit(0)
        final_prompt = (
            f'Create a 1990s Memphis-inspired typographic logo that reads: "{text_input}". '
            f'Style: 90s TV commercial / snack packaging lettering, playful and energetic, slightly italic script with uneven hand-drawn feel, two-tone gradients and airbrushed shading, subtle halftone/print vibe, thin white highlight, no shadow at all, not a thick sticker outline). '
            f'Add a simple zig-zag / squiggle underline in Memphis style. '
            f'Background: solid flat chroma key green (#00FF00), perfectly uniform. '
            f'Strictly avoid: 3D chrome, rainbow neon, glossy metallic look, thick black outline sticker effect, modern esports logo style, glow effects, bevel/emboss, photorealism, extra objects, patterns, textures, collage, food images, shadow. '
            f'Centered composition, clean edges, high resolution.'
        )

    # Summary
    print("\n--- Summary ---")
    if image_path:
        print(f"Image:      {image_path}")
    else:
        print(f"Image:      None (Text-to-Image)")
    print(f"Provider:   {provider}")
    print(f"Model:      {model_choice}")
    if provider == "OpenAI":
        print(f"Resolution: {res_key}")
        print(f"Quality:    {quality_key}")
    else:
        print(f"Ratio:      {aspect_ratio}")
        print(f"Size:       {quality_key}")
    print(f"Prompt:     {final_prompt}")
    print(f"Total Cost: ${final_cost:.3f}")

    confirm = questionary.confirm("Proceed with API call?").ask()
    if not confirm:
        print("Cancelled.")
        return

    # 7. API Call
    if provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
            sys.exit(1)

        client_openai = OpenAI(api_key=api_key)
        print(f"\nSending request to OpenAI ({model_choice})...")
        try:
            if image_path:
                # EDIT MODE
                image_tuple = process_image_for_api(image_path, res_key)
                response = client_openai.images.edit(
                    model=model_choice,
                    image=image_tuple,
                    prompt=final_prompt,
                    n=1,
                    size=res_key,
                    quality=quality_key.lower(),
                )
            else:
                # GENERATE MODE
                response = client_openai.images.generate(
                    model=model_choice,
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
                print("\nError: Could not retrieve image data from the API response.")
                print(f"Debug Response: {response}")
        except Exception as e:
            print(f"\nAn error occurred during OpenAI call: {e}")

    else:
        # Google Provider
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
            sys.exit(1)

        client_google = genai.Client(api_key=api_key)
        print(f"\nSending request to Google ({model_choice})...")
        try:
            # Prepare Google Config
            config_args = {}
            if aspect_ratio != "Auto":
                config_args["aspect_ratio"] = aspect_ratio

            if model_choice == "gemini-3-pro-image-preview":
                config_args["image_size"] = quality_key

            req_contents = [final_prompt]
            img_context = None

            if image_path:
                img_context = Image.open(image_path)
                req_contents.append(img_context)

            response = client_google.models.generate_content(
                model=model_choice,
                contents=req_contents,
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(**config_args)
                ),
            )

            # Close image if opened
            if img_context:
                img_context.close()

            # Handle Response
            saved = False
            if hasattr(response, "parts"):
                for part in response.parts:
                    if part.inline_data is not None:
                        generated_image = part.as_image()
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        if image_path:
                            base_name = os.path.splitext(os.path.basename(image_path))[
                                0
                            ]
                            filename = f"edited_{timestamp}_{base_name}.png"
                        else:
                            filename = f"generated_{timestamp}.png"

                        generated_image.save(filename)
                        print(f"\nSuccess! File saved successfully as {filename}")
                        saved = True
                        break

            if not saved:
                print("\nError: No image found in Google API response.")
                if hasattr(response, "parts"):
                    for part in response.parts:
                        if part.text:
                            print(f"Response text: {part.text}")
                else:
                    print(f"Debug Response: {response}")

        except Exception as e:
            print(f"\nAn error occurred during Google call: {e}")


def save_api_image(image_url, image_b64, original_path):
    """Downloads or decodes an image and saves it to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if original_path:
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        filename = f"edited_{timestamp}_{base_name}.png"
    else:
        filename = f"generated_{timestamp}.png"

    if image_url:
        print(f"\nSuccess! Image available at:\n{image_url}")
        print(f"Downloading and saving to {filename}...")
        img_data = requests.get(image_url).content
    else:
        print(f"\nSuccess! Received base64 image data.")
        print(f"Decoding and saving to {filename}...")
        img_data = base64.b64decode(image_b64)

    with open(filename, "wb") as handler:
        handler.write(img_data)
    print(f"File saved successfully as {filename}")


if __name__ == "__main__":
    main()
