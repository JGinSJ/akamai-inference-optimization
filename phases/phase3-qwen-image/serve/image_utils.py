"""
Image decoding utilities.

Deliberately kept free of PyTorch and transformers imports so that tests
can import and exercise this module without a GPU environment.
"""

import base64
import io

from PIL import Image


def decode_image(image_b64: str) -> Image.Image:
    """
    Decode a base64 string to a PIL Image in RGB mode.

    Accepts either:
      - Raw base64:  "/9j/4AAQ..."
      - Data URI:    "data:image/jpeg;base64,/9j/4AAQ..."

    The data-URI prefix is stripped before decoding.
    The result is always converted to RGB (handles grayscale, RGBA, etc.).
    """
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    raw_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
