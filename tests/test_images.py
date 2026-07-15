"""Tests for the pure image helpers in imgprompt.images.

Network/disk-heavy functions (save_api_image, save_image_bytes) are exercised
only through their pure pieces or with a tmp_path; the real provider calls are
out of scope.
"""

import io
import os
import tempfile

import pytest
from PIL import Image

from imgprompt.images import (
    get_closest_aspect_ratio,
    get_image_extension,
    get_images_in_cwd,
    is_pdf,
    pdf_page_count,
    rasterize_pdf_all_pages,
    process_image_for_api,
    rasterize_pdf,
    save_image_bytes,
)

fitz = pytest.importorskip("fitz")


def _write_pdf(path, pages=1):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()
    return str(path)


def _make_image_bytes(size=(64, 64), fmt="PNG", color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def _write_image(path, size=(64, 64), fmt="PNG"):
    Image.new("RGB", size, (10, 20, 30)).save(path, format=fmt)
    return str(path)


class TestGetImageExtension:
    def test_png(self):
        assert get_image_extension(_make_image_bytes(fmt="PNG")) == ".png"

    def test_jpeg(self):
        assert get_image_extension(_make_image_bytes(fmt="JPEG")) == ".jpg"

    def test_webp(self):
        assert get_image_extension(_make_image_bytes(fmt="WEBP")) == ".webp"

    def test_garbage_defaults_to_png(self):
        assert get_image_extension(b"not an image") == ".png"


class TestGetClosestAspectRatio:
    def test_square_image_picks_1_1(self, tmp_path):
        p = _write_image(tmp_path / "sq.png", size=(500, 500))
        assert get_closest_aspect_ratio(p, ["1:1", "16:9", "9:16"]) == "1:1"

    def test_wide_image_picks_16_9(self, tmp_path):
        p = _write_image(tmp_path / "wide.png", size=(1600, 900))
        assert get_closest_aspect_ratio(p, ["1:1", "16:9", "9:16"]) == "16:9"

    def test_tall_image_picks_9_16(self, tmp_path):
        p = _write_image(tmp_path / "tall.png", size=(900, 1600))
        assert get_closest_aspect_ratio(p, ["1:1", "16:9", "9:16"]) == "9:16"


class TestProcessImageForApi:
    def test_auto_sends_as_is(self, tmp_path):
        p = _write_image(tmp_path / "a.png", size=(2000, 2000))
        filename, data, mime = process_image_for_api(p, "auto")
        assert filename == "a.png"
        assert mime == "image/png"
        assert isinstance(data, io.BytesIO)

    def test_small_image_sent_untouched(self, tmp_path):
        p = _write_image(tmp_path / "small.png", size=(100, 100))
        _, data, _ = process_image_for_api(p, "1024x1024")
        # Untouched: bytes equal the original file content.
        with open(p, "rb") as f:
            assert data.getvalue() == f.read()

    def test_large_image_is_resized(self, tmp_path):
        p = _write_image(tmp_path / "big.png", size=(4000, 4000))
        _, data, _ = process_image_for_api(p, "1024x1024")
        resized = Image.open(data)
        assert resized.width <= 1024 and resized.height <= 1024

    def test_mime_type_for_jpeg(self, tmp_path):
        p = _write_image(tmp_path / "photo.jpg", size=(100, 100), fmt="JPEG")
        _, _, mime = process_image_for_api(p, "1024x1024")
        assert mime == "image/jpeg"


class TestSaveImageBytes:
    def test_generated_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_image_bytes(_make_image_bytes(fmt="PNG"), None)
        files = list(tmp_path.glob("generated_*.png"))
        assert len(files) == 1

    def test_edited_filename_from_original(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        original = _write_image(tmp_path / "cat.png")
        save_image_bytes(_make_image_bytes(fmt="PNG"), original)
        files = list(tmp_path.glob("edited_*_cat.png"))
        assert len(files) == 1

    def test_collision_appends_counter(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Two saves in the same second must not clobber each other.
        save_image_bytes(_make_image_bytes(fmt="PNG"), None)
        save_image_bytes(_make_image_bytes(fmt="PNG"), None)
        files = list(tmp_path.glob("generated_*.png"))
        assert len(files) == 2


class TestGetImagesInCwd:
    def test_lists_only_images(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_image(tmp_path / "pic.png")
        (tmp_path / "notes.txt").write_text("hi")
        (tmp_path / "data.json").write_text("{}")
        result = get_images_in_cwd()
        assert "pic.png" in result
        assert "notes.txt" not in result

    def test_case_insensitive_extension(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_image(tmp_path / "PIC.PNG")
        assert "PIC.PNG" in get_images_in_cwd()

    def test_lists_pdfs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_pdf(tmp_path / "report.pdf")
        assert "report.pdf" in get_images_in_cwd()


class TestIsPdf:
    def test_pdf(self):
        assert is_pdf("doc.pdf")
        assert is_pdf("DOC.PDF")

    def test_non_pdf(self):
        assert not is_pdf("photo.png")
        assert not is_pdf("archive.pdf.png")


class TestRasterizePdf:
    def test_page_count(self, tmp_path):
        pdf = _write_pdf(tmp_path / "multi.pdf", pages=3)
        assert pdf_page_count(pdf) == 3

    def test_single_page_naming(self, tmp_path):
        pdf = _write_pdf(tmp_path / "doc.pdf", pages=1)
        out = rasterize_pdf(pdf)
        assert out == str(tmp_path / "doc.png")
        img = Image.open(out)
        assert img.format == "PNG"

    def test_multipage_naming_includes_page(self, tmp_path):
        pdf = _write_pdf(tmp_path / "doc.pdf", pages=2)
        assert rasterize_pdf(pdf, page_index=0) == str(tmp_path / "doc_p1.png")
        assert rasterize_pdf(pdf, page_index=1) == str(tmp_path / "doc_p2.png")

    def test_out_of_range_raises(self, tmp_path):
        pdf = _write_pdf(tmp_path / "doc.pdf", pages=1)
        with pytest.raises(ValueError):
            rasterize_pdf(pdf, page_index=5)

    def test_no_clobber(self, tmp_path):
        pdf = _write_pdf(tmp_path / "doc.pdf", pages=1)
        first = rasterize_pdf(pdf)
        second = rasterize_pdf(pdf)
        assert first != second
        assert second == str(tmp_path / "doc_2.png")

    def test_readonly_dir_falls_back_to_tmp(self, tmp_path):
        src = tmp_path / "ro"
        src.mkdir()
        pdf = _write_pdf(src / "doc.pdf", pages=1)
        os.chmod(src, 0o500)  # read + execute, no write
        try:
            out = rasterize_pdf(pdf)
        finally:
            os.chmod(src, 0o700)
        assert os.path.dirname(out) != str(src)
        assert os.path.dirname(out) == tempfile.gettempdir()


class TestRasterizePdfAllPages:
    def test_returns_one_png_per_page_in_order(self, tmp_path):
        pdf = _write_pdf(tmp_path / "doc.pdf", pages=3)
        outs = rasterize_pdf_all_pages(pdf)
        assert outs == [
            str(tmp_path / "doc_p1.png"),
            str(tmp_path / "doc_p2.png"),
            str(tmp_path / "doc_p3.png"),
        ]
        for out in outs:
            assert Image.open(out).format == "PNG"

    def test_single_page_pdf(self, tmp_path):
        pdf = _write_pdf(tmp_path / "solo.pdf", pages=1)
        assert rasterize_pdf_all_pages(pdf) == [str(tmp_path / "solo.png")]
