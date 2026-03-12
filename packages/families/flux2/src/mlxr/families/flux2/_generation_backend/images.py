from __future__ import annotations

import hashlib
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .constants import _LATENT_DOWNSAMPLE


def _center_crop_to_multiple(image: Image.Image, multiple: int) -> Image.Image:
    width, height = image.size
    cropped_width = (width // multiple) * multiple
    cropped_height = (height // multiple) * multiple
    left = (width - cropped_width) // 2
    top = (height - cropped_height) // 2
    return image.crop((left, top, left + cropped_width, top + cropped_height))


def _cap_pixels(image: Image.Image, limit_pixels: int) -> Image.Image:
    width, height = image.size
    pixels = width * height
    if pixels <= limit_pixels:
        return image
    scale = math.sqrt(limit_pixels / float(pixels))
    return image.resize(
        (int(width * scale), int(height * scale)),
        Image.Resampling.LANCZOS,
    )


def _validate_reference_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width < 64 or height < 64:
        raise ValueError(
            f"FLUX.2 reference images must be at least 64px on both sides, got {width}x{height}"
        )
    if width / height > 8.0 or height / width > 8.0:
        raise ValueError(
            f"FLUX.2 reference images must stay within 8:1 aspect ratio, got {width}x{height}"
        )
    return image


def _reference_tensor(image: Image.Image, *, limit_pixels: int) -> mx.array:
    image = _validate_reference_image(image.convert("RGB"))
    image = _cap_pixels(image, limit_pixels)
    image = _center_crop_to_multiple(image, _LATENT_DOWNSAMPLE)
    pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return mx.array(pixels, dtype=mx.float32)


def _conditioning_reference_limit(reference_count: int) -> int:
    return 1_024**2 if reference_count > 1 else 2_024**2


def _prompt_signature(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _resolved_dimensions(
    *,
    width: int | None,
    height: int | None,
    image_paths: tuple[Path, ...],
) -> tuple[int, int]:
    reference_width: int | None = None
    reference_height: int | None = None
    if image_paths:
        with Image.open(image_paths[0]) as reference_image:
            reference_width, reference_height = reference_image.size
        if reference_width is not None:
            reference_width = max(
                _LATENT_DOWNSAMPLE,
                reference_width - reference_width % _LATENT_DOWNSAMPLE,
            )
        if reference_height is not None:
            reference_height = max(
                _LATENT_DOWNSAMPLE,
                reference_height - reference_height % _LATENT_DOWNSAMPLE,
            )
    resolved_width = width if width is not None else reference_width or 1024
    resolved_height = height if height is not None else reference_height or 1024
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("FLUX.2 image dimensions must be positive")
    if (
        resolved_width % _LATENT_DOWNSAMPLE != 0
        or resolved_height % _LATENT_DOWNSAMPLE != 0
    ):
        raise ValueError("FLUX.2 image dimensions must be multiples of 16")
    return resolved_width, resolved_height
