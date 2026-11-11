"""Deterministic, bounding-box-based CUB background perturbations.

These are controlled stress tests, not true semantic foreground masks:
pixels *inside the bounding rectangle* are preserved, including any background there.
"""
from PIL import Image, ImageFilter, ImageOps
import numpy as np


def bbox_to_pixels(bbox, size):
    """Convert CUB (x, y, width, height) to a clipped PIL rectangle."""
    x, y, w, h = map(float, bbox)
    width, height = size
    left = max(0, min(width, int(x)))
    top = max(0, min(height, int(y)))
    right = max(0, min(width, int(x + w + 0.5)))
    bottom = max(0, min(height, int(y + h + 0.5)))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid bbox {bbox!r} for image size {size}")
    return left, top, right, bottom


def background_variant(image, bbox, kind="original", donor=None, blur_radius=22):
    """Keep CUB bbox pixels identical; alter only pixels outside the box.

    kind: original, blur, replace
    donor: optional PIL image used as source of replacement background; if
           omitted, deterministic neutral gray replaces the background.
    """
    image = image.convert("RGB")
    if kind == "original":  # unchanged copy, the reference condition
        return image.copy()
    box = bbox_to_pixels(bbox, image.size)
    if kind == "blur":
        # Strong Gaussian blur removes background texture but keeps its colours.
        changed = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    elif kind == "replace":
        if donor is None:
            # Flat gray (default): no second bird or species-typical habitat can leak in.
            changed = Image.new("RGB", image.size, (128, 128, 128))
        else:
            changed = ImageOps.fit(donor.convert("RGB"), image.size, method=Image.Resampling.BILINEAR)
            # Blur the donor to avoid highly salient second birds in the background.
            changed = changed.filter(ImageFilter.GaussianBlur(radius=15))
    else:
        raise ValueError("kind must be original, blur or replace")
    # Paste original rectangle back *exactly*; crop model's input must not change.
    changed.paste(image.crop(box), box)
    return changed


class SquarePad:
    """Pad an image to a square with neutral gray, keeping it centred.

    Used before resizing bird crops: resizing a tall or wide crop straight to a
    square stretches the bird out of shape, which kept the first crop model at 40%
    accuracy (README, problem 1).
    """

    def __call__(self, image):
        w, h = image.size
        size = max(w, h)
        left = (size - w) // 2
        top = (size - h) // 2
        return ImageOps.expand(image, (left, top, size - w - left, size - h - top), fill=(128, 128, 128))
