import os
import io
import base64
import requests
from datetime import datetime
from typing import Optional
from PIL import Image

from imgprompt.presets import ASPECT_RATIO_VALUES

IMAGE_EXTENSIONS = (".pcx", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def get_images_in_cwd() -> list[str]:
    """Returns a list of image files in the current working directory."""
    return [f for f in os.listdir(".") if f.lower().endswith(IMAGE_EXTENSIONS)]


def process_image_for_api(image_path: str, target_res: str) -> tuple:
    """
    Checks if the image needs resizing and returns a tuple (filename, data, mime_type).
    If the image is larger than the target resolution in any dimension, it is resized.
    """
    target_width, target_height = map(int, target_res.split("x"))
    filename = os.path.basename(image_path)

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = mime_types.get(ext, "image/png")

    with Image.open(image_path) as img:
        original_width, original_height = img.size

        if original_width > target_width or original_height > target_height:
            print(
                f"Resizing input image from {original_width}x{original_height} to fit within {target_width}x{target_height}..."
            )
            img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

            output = io.BytesIO()
            fmt = img.format if img.format else "PNG"
            if ext in (".jpg", ".jpeg"):
                fmt = "JPEG"
                mime_type = "image/jpeg"

            img.save(output, format=fmt)
            output.seek(0)
            return (filename, output, mime_type)
        else:
            print(
                f"Input image {original_width}x{original_height} is within limits. Sending untouched."
            )
            with open(image_path, "rb") as f:
                return (filename, io.BytesIO(f.read()), mime_type)


def get_closest_aspect_ratio(image_path: str, supported_ratios: list[str]) -> str:
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


def get_image_extension(img_data: bytes) -> str:
    """Detects the image format from bytes and returns the appropriate extension."""
    try:
        img = Image.open(io.BytesIO(img_data))
        fmt = img.format
        if fmt:
            fmt = fmt.upper()
            if fmt in ("JPEG", "JPG", "MPO"):
                return ".jpg"
            elif fmt == "PNG":
                return ".png"
            elif fmt == "WEBP":
                return ".webp"
    except Exception:
        pass
    return ".png"


def save_api_image(
    image_url: Optional[str], image_b64: Optional[str], original_path: Optional[str]
) -> None:
    """Downloads or decodes an image and saves it to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if image_url:
        print(f"\nSuccess! Image available at:\n{image_url}")
        print(f"Downloading image...")
        img_data = requests.get(image_url).content
    else:
        print(f"\nSuccess! Received base64 image data.")
        print(f"Decoding image...")
        img_data = base64.b64decode(image_b64)

    ext = get_image_extension(img_data)

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

    with open(output_path, "wb") as handler:
        handler.write(img_data)
    print(f"File saved successfully as {output_path}")
