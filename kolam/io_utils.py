"""kolam.io_utils -- Module 1: loading and saving images."""
from __future__ import annotations

import io
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_image(source, name: str | None = None) -> np.ndarray:
    """Load JPG/JPEG/PNG from a path, raw bytes or a file-like object -> BGR uint8.

    * EXIF orientation is honoured (phone photos).
    * Transparent PNGs are composited on a background that contrasts with the
      artwork (a white-flour kolam on a transparent background would vanish on white).
    """
    if isinstance(source, (str, os.PathLike)):
        ext = os.path.splitext(str(source))[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type {ext!r}; use {SUPPORTED_EXTENSIONS}")
        pil = Image.open(source)
    elif isinstance(source, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(source))
    else:
        if name is not None:
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type {ext!r}; use {SUPPORTED_EXTENSIONS}")
        pil = Image.open(source)
    pil = ImageOps.exif_transpose(pil)
    has_alpha = pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info)
    if has_alpha:
        rgba = pil.convert("RGBA")
        arr = np.array(rgba)
        opaque = arr[..., 3] > 128
        lum = arr[..., :3].mean(axis=2)[opaque].mean() if opaque.any() else 0
        bg_val = 0 if lum > 128 else 255
        bg = Image.new("RGBA", rgba.size, (bg_val, bg_val, bg_val, 255))
        bg.alpha_composite(rgba)
        pil = bg
    rgb = np.array(pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_image(path: str, img: np.ndarray) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, img):
        raise IOError(f"could not write {path}")
    return path


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError("PNG encoding failed")
    return buf.tobytes()
