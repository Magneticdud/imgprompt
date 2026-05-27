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
    Pass target_res="auto" to skip resizing and send the image as-is.
    """
    filename = os.path.basename(image_path)

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = mime_types.get(ext, "image/png")

    if target_res == "auto":
        print("Auto size: sending image as-is.")
        with open(image_path, "rb") as f:
            return (filename, io.BytesIO(f.read()), mime_type)

    target_width, target_height = map(int, target_res.split("x"))

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


def save_image_bytes(img_bytes: bytes, original_path: str | None) -> None:
    """Detect image format from bytes and save to disk with timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = get_image_extension(img_bytes)
    if original_path:
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_dir = os.path.dirname(original_path) or "."
        output_path = os.path.join(output_dir, f"edited_{timestamp}_{base_name}{ext}")
    else:
        output_path = f"generated_{timestamp}{ext}"
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"File saved successfully as {output_path}")


def save_api_image(
    image_url: Optional[str], image_b64: Optional[str], original_path: Optional[str]
) -> None:
    """Downloads or decodes an image and saves it to disk."""
    if image_url:
        print(f"\nSuccess! Image available at:\n{image_url}")
        print("Downloading image...")
        img_data = requests.get(image_url).content
    else:
        print("\nSuccess! Received base64 image data.")
        print("Decoding image...")
        img_data = base64.b64decode(image_b64)
    save_image_bytes(img_data, original_path)
