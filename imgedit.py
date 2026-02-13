#!/usr/bin/env python3
import os
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
    "Photorealistic restoration. Strictly preserve geometry and identity. No creative reinterpretation. No new details beyond what is implied by the pixels.",
    "Transform the input photo into a Japanese manga illustration. Preserve the person identity, pose, clothing, and background composition. Clean black ink lineart, confident contours, simplified shapes, screentone shading, high-contrast black and white, crisp lines, minimal gray tones, manga panel style, detailed eyes and hair with ink strokes, no photorealistic texture.",
    "Convert the input photo into a high-quality anime illustration. Preserve identity and facial features. Cel shading, clean linework, smooth gradient highlights, stylized but realistic proportions, vibrant but controlled colors, sharp eyes, defined hair shapes, studio anime lighting.",
    "Give this portrait a 1950s vintage film look.",
    "Turn the photo into a shoujo manga style illustration. Delicate lineart, soft screentones, elegant facial features, sparkly eyes, light blush, airy hair highlights, romantic composition, clean black-and-white manga look.",
    "Transform the uploaded photo into a black-and-white graphite pencil drawing. Use clean line art with cross-hatching for shadows and volume, visible paper texture, and no solid black fills. Keep the exact composition, subject identity, pose, proportions, and camera framing from the original photo. Simplify the background slightly but keep it consistent. No color. Avoid: color, watercolor, oil paint, digital painting, CGI/3D, cartoon/anime, vector-clean outlines, automatic sketch filter look, blur, noisy artifacts, soft airbrushed shading, heavy solid blacks.",
    "Convert the photo into a classic pencil illustration style: precise ink-like pencil outlines, diagonal and cross-hatching for skies/shadows, graphite-only shading (no smooth airbrush gradients), detailed textures on hair/clothing, and a sketchbook look. Maintain the original photo composition and subject identity exactly. Monochrome only. Avoid: color, watercolor, oil paint, digital painting, CGI/3D, cartoon/anime, vector-clean outlines, automatic sketch filter look, blur, noisy artifacts, soft airbrushed shading, heavy solid blacks.",
    "Restore this scanned page with maximum fidelity. Only perform non-destructive cleanup: remove dust/specks, scan noise, paper texture and stains; normalize the halftone/screen pattern to be uniform; correct slight skew. Do NOT redraw, reinterpret, or invent any content. Preserve all original linework, shapes, proportions, fonts, and text exactly. No style changes. Output a clean, flat, high-resolution image that matches the original as closely as possible.",
    "Rebuild the business card as a flat print file. Canvas size: 91×61 mm including 3 mm bleed on all sides (final trim 85×55 mm). Keep all text inside a 4 mm safe margin from the trim edge.Match the original layout from the reference photo. Output: 300 DPI.",
    "Perform conservative color restoration only on the provided 1970s photo. Correct color cast (yellow/magenta/green), restore faded colors, and rebalance white balance to a natural analog-photo look.Do not change any details: keep identical geometry, composition, crop, perspective, faces, skin texture, hair, edges, background, text, film grain, dust, scratches, stains, and any imperfections.No enhancement: no denoise, no sharpening, no deblur, no upscaling, no HDR, no relighting, no beautification. Output must match the original framing and resolution; only chroma/tonal color values may change.",
    "Convert this image into a clean, black and white line art. Use sharp black outlines on a pure white background. Remove all shading, colors, and gradients. It must look like a high-quality adult coloring book page, staying faithful to the original subject and background details.",
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
    "Business Card",
    "APPROVED Stamp",
    "Generic Logotype",
    "Custom Prompt",
]

PRESET_PROMPTS_DUAL = [
    "Combine the contents of IMG_1 and IMG_2 into a coherent scene.",
    "Use the composition of IMG_1 and the style of IMG_2.",
    "IMG_1 is the subject, IMG_2 is the background.",
    "Create a vintage etching / engraved illustration double exposure using two input photos. Use IMG_1 as the main subject silhouette and keep its pose, proportions, and outline faithful. Use IMG_2 as the internal scene, visible only inside the silhouette of IMG_1 (no spill outside the outline). Convert everything to black-and-white ink linework with cross-hatching and etched shading, consistent line weight, high detail. Fit and scale IMG_2 to the silhouette while preserving its aspect ratio; adjust position for a pleasing composition. Clean white background, no text, no frame, no extra objects.The outer area must remain blank white; all texture must be inside the silhouette only.",
    "Custom Prompt",
]


def get_images_in_cwd() -> List[str]:
    """Returns a list of image files in the current working directory."""
    extensions = (".pcx", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    return [f for f in os.listdir(".") if f.lower().endswith(extensions)]


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
    parser = argparse.ArgumentParser(description="GPT-Image & Gemini Image Editor")
    parser.add_argument("image", nargs="?", help="Path to the image to edit")
    parser.add_argument(
        "--free",
        action="store_true",
        help="Start in Text-to-Image mode (no base image)",
    )
    args = parser.parse_args()

    # 1. Select Image
    if args.free:
        input_images = []
    else:
        input_images = select_inputs(args.image)

    # Legacy support variable
    image_path = input_images[0] if input_images else None

    if input_images:
        if len(input_images) == 1:
            print(f"\nSelected Image: {input_images[0]}")
        else:
            print(f"\nSelected Images: {', '.join(input_images)}")
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
            closest = get_closest_aspect_ratio(
                image_path, list(GEMINI_RESOLUTIONS.keys())
            )
        else:
            closest = default_ratio

        aspect_ratio = questionary.select(
            "Select aspect ratio:", choices=ratio_options, default=closest
        ).ask()
        if not aspect_ratio:
            sys.exit(0)
        res_key = GEMINI_RESOLUTIONS.get(aspect_ratio, "Auto")

    # 5. Select Quality / Size and show costs
    quality_key = "2K"  # Default for Google
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
    if len(input_images) > 1:
        prompt_choices = PRESET_PROMPTS_DUAL
    elif image_path:
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
            sys.exit(0)
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
            sys.exit(0)

        print(
            "Enter the rest of the details (multiline). Press Alt+Enter or Esc+Enter to submit:"
        )
        details = questionary.text("Details:", multiline=True).ask()
        if not details:
            print("Error: Details are required.")
            sys.exit(0)

        final_prompt = (
            f"Create a 2D graphic design for a business card without mockup: no 3D rendering, no scene, no photography, no perspective.\n"
            f"Canvas: 85x55 mm (aspect ratio 1.545:1), horizontal, equivalent to 300 dpi, 3 mm safety margins, text strictly within the safe area.\n"
            f"Text Layout: Centered in a single column. Visual hierarchy: “{main_title}” must be very large and bold. The rest smaller but clearly legible.\n"
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
            sys.exit(0)

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
            sys.exit(0)

        final_prompt = f"A typographic logo, with centered text, with the following text: {logo_text}"

    # Summary
    print("\n--- Summary ---")
    if input_images:
        print(f"Images:     {', '.join(input_images)}")
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
            if input_images:
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

            response = client_google.models.generate_content(
                model=model_choice,
                contents=req_contents,
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(**config_args)
                ),
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

                # Check for prompt feedback blocks (common in gemini-3-pro-image-preview)
                if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                    if hasattr(response.prompt_feedback, "block_reason"):
                        print(
                            f"Prompt Feedback Block Reason: {response.prompt_feedback.block_reason}"
                        )
                        reason_str = str(response.prompt_feedback.block_reason)
                        if any(x in reason_str for x in ["SAFETY", "BLOCK", "OTHER"]):
                            print(
                                ">> The request was likely blocked due to safety settings or policy violations."
                            )
                            print(
                                ">> This often happens with images of famous people, children, or restricted content."
                            )

                # Check for safety blocks or other finish reasons
                if hasattr(response, "candidates") and response.candidates:
                    for i, candidate in enumerate(response.candidates):
                        if hasattr(candidate, "finish_reason"):
                            print(
                                f"Candidate {i+1} Finish Reason: {candidate.finish_reason}"
                            )
                            # Convert to string to be safe, though it's likely an enum
                            reason_str = str(candidate.finish_reason)
                            if any(
                                x in reason_str for x in ["SAFETY", "BLOCK", "OTHER"]
                            ):
                                print(
                                    ">> The request was likely blocked due to safety settings or policy violations."
                                )
                                print(
                                    ">> This often happens with images of famous people, children, or restricted content."
                                )

                if hasattr(response, "parts") and response.parts:
                    for part in response.parts:
                        if part.text:
                            print(f"Response text: {part.text}")

                if not hasattr(response, "parts") or not response.parts:
                    # print(f"Debug Response (Parts is None/Empty): {response}")
                    pass

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
