import os
import io
import sys
import base64
import shutil
import tempfile
import subprocess
import requests
from datetime import datetime
from typing import Optional
from PIL import Image

from imgprompt.presets import ASPECT_RATIO_VALUES

IMAGE_EXTENSIONS = (".pcx", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
PDF_EXTENSIONS = (".pdf",)

# Inline terminal preview of saved results. Enabled by default and toggled off
# with --no-preview (or automatically for batch/multi-variant runs, where a
# preview per result would flood the scrollback). Rendering is best-effort: it
# only fires on an interactive TTY, delegates to whichever image-to-terminal
# tool the user has installed, and swallows any failure silently so a missing
# tool or an unsupported terminal never disrupts a real run.
_PREVIEW_ENABLED = True
_PREVIEW_MAX_HEIGHT = 20


def configure_preview(enabled: bool) -> None:
    """Enable or disable the inline terminal preview for the rest of the run."""
    global _PREVIEW_ENABLED
    _PREVIEW_ENABLED = enabled


def _preview_command(path: str, max_height: int) -> Optional[list[str]]:
    """Build the argv for the first available terminal image renderer, or None.

    Prefers tools that let us cap the height so a large image never scrolls the
    wizard output (cost, saved path) out of view: chafa and viu both take a
    cell height. kitty's icat is a last resort (kitty terminals only) and is
    left uncapped. All three auto-detect the terminal's graphics capability."""
    if shutil.which("chafa"):
        # chafa fits within WIDTHxHEIGHT preserving aspect ratio, so a wide
        # terminal width plus a small height effectively caps the height.
        cols = shutil.get_terminal_size((80, 24)).columns
        return ["chafa", f"--size={cols}x{max_height}", path]
    if shutil.which("viu"):
        return ["viu", "-h", str(max_height), path]
    if shutil.which("kitty") and (
        os.environ.get("KITTY_WINDOW_ID") or "kitty" in os.environ.get("TERM", "")
    ):
        return ["kitty", "+kitten", "icat", "--align=left", path]
    return None


def preview_image_file(path: str, max_height: int = _PREVIEW_MAX_HEIGHT) -> None:
    """Best-effort inline preview of a saved image file in the terminal.

    Delegates to an installed image-to-terminal tool (chafa/viu/kitty icat),
    each of which auto-detects the terminal's graphics protocol. Silently
    no-ops when previews are disabled, stdout is not a TTY, no renderer is
    installed, or the renderer fails for any reason."""
    if not _PREVIEW_ENABLED or not sys.stdout.isatty():
        return
    cmd = _preview_command(path, max_height)
    if not cmd:
        return
    try:
        print()
        subprocess.run(cmd, check=False)
    except Exception:
        # A preview is a nicety, never a hard requirement; never abort the run.
        pass

# DPI used when rasterizing PDF pages to bitmaps. ~200 DPI keeps an A4 page
# around 1654x2339, sharp enough for editing while the later resize step trims
# it to the chosen target resolution.
PDF_RASTER_DPI = 200


def get_images_in_cwd() -> list[str]:
    """Returns a list of image (and PDF) files in the current working directory."""
    return [
        f
        for f in os.listdir(".")
        if f.lower().endswith(IMAGE_EXTENSIONS + PDF_EXTENSIONS)
    ]


def is_pdf(path: str) -> bool:
    """True if the path points to a PDF (by extension)."""
    return path.lower().endswith(PDF_EXTENSIONS)


def pdf_page_count(pdf_path: str) -> int:
    """Returns the number of pages in a PDF. Requires PyMuPDF (fitz)."""
    import fitz  # PyMuPDF, imported lazily so non-PDF runs don't need it

    with fitz.open(pdf_path) as doc:
        return doc.page_count


def rasterize_pdf(pdf_path: str, page_index: int = 0, dpi: int = PDF_RASTER_DPI) -> str:
    """Render a single PDF page to a PNG saved next to the source PDF.

    APIs accept raster formats only, so a PDF must be rasterized before upload.
    The PNG is written alongside the original (named after it) so downstream
    output filenames and directories stay sensible, and the user keeps a visible
    record of exactly what was sent. If the source directory is not writable
    (e.g. a read-only SMB/GVFS mount) it falls back to a temp directory.
    Returns the PNG path.
    """
    import fitz  # PyMuPDF, imported lazily

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    src_dir = os.path.dirname(pdf_path) or "."

    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        if not 0 <= page_index < page_count:
            raise ValueError(
                f"Page {page_index + 1} out of range (PDF has {page_count} pages)."
            )
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)

    suffix = "" if page_count == 1 else f"_p{page_index + 1}"
    filename = f"{stem}{suffix}.png"

    # Prefer saving next to the PDF; fall back to a temp dir if that directory
    # is not writable. os.access can be wrong on some mounts, so the save itself
    # is also guarded and retried in the temp dir.
    out_dir = src_dir if os.access(src_dir, os.W_OK) else tempfile.gettempdir()

    def _unique(directory: str) -> str:
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            root, ext = os.path.splitext(candidate)
            counter = 2
            while os.path.exists(f"{root}_{counter}{ext}"):
                counter += 1
            candidate = f"{root}_{counter}{ext}"
        return candidate

    out_path = _unique(out_dir)
    try:
        pix.save(out_path)
    except OSError:
        if out_dir == tempfile.gettempdir():
            raise
        out_path = _unique(tempfile.gettempdir())
        pix.save(out_path)
        print(f"Note: '{src_dir}' is not writable; saved rasterized page to {out_path}")
    return out_path


def rasterize_pdf_all_pages(pdf_path: str, dpi: int = PDF_RASTER_DPI) -> list[str]:
    """Render every page of a PDF to a PNG. Returns the list of PNG paths.

    Thin loop over `rasterize_pdf` so the whole document can be fed into batch
    mode (one input image per page). Pages come back in document order; each PNG
    is named `<stem>_pN.png` (see `rasterize_pdf`) so the order is also visible
    on disk.
    """
    return [
        rasterize_pdf(pdf_path, page_index=i, dpi=dpi)
        for i in range(pdf_page_count(pdf_path))
    ]


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

            if fmt == "JPEG" and img.mode != "RGB":
                img = img.convert("RGB")

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

    # Timestamps are second-granular, so multiple iterations in the same second
    # would clobber each other. Append a counter to keep every result.
    if os.path.exists(output_path):
        root, suffix = os.path.splitext(output_path)
        counter = 2
        while os.path.exists(f"{root}_{counter}{suffix}"):
            counter += 1
        output_path = f"{root}_{counter}{suffix}"

    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"File saved successfully as {output_path}")
    preview_image_file(output_path)


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
