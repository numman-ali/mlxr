from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import Resampling

_DEFAULT_IMAGE_TOKEN = "<|image_pad|>"


@dataclass(frozen=True)
class Qwen2VLImageProcessorConfig:
    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    rescale_factor: float
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    do_convert_rgb: bool = True
    do_resize: bool = True
    do_rescale: bool = True
    do_normalize: bool = True

    @classmethod
    def from_path(cls, path: Path) -> "Qwen2VLImageProcessorConfig":
        raw = json.loads(path.read_text("utf-8"))
        image_mean = _rgb_triplet(raw["image_mean"], field_name="image_mean")
        image_std = _rgb_triplet(raw["image_std"], field_name="image_std")
        return cls(
            min_pixels=int(raw["min_pixels"]),
            max_pixels=int(raw["max_pixels"]),
            patch_size=int(raw["patch_size"]),
            temporal_patch_size=int(raw["temporal_patch_size"]),
            merge_size=int(raw["merge_size"]),
            rescale_factor=float(raw["rescale_factor"]),
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=bool(raw.get("do_convert_rgb", True)),
            do_resize=bool(raw.get("do_resize", True)),
            do_rescale=bool(raw.get("do_rescale", True)),
            do_normalize=bool(raw.get("do_normalize", True)),
        )


@dataclass(frozen=True)
class Qwen2VLProcessedImages:
    pixel_values: np.ndarray
    image_grid_thw: np.ndarray


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(
            factor,
            math.floor(height / beta / factor) * factor,
        )
        resized_width = max(
            factor,
            math.floor(width / beta / factor) * factor,
        )
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


def preprocess_qwen2_vl_images(
    images: tuple[Image.Image, ...],
    *,
    config: Qwen2VLImageProcessorConfig,
) -> Qwen2VLProcessedImages:
    pixel_values: list[np.ndarray] = []
    image_grid_thw: list[tuple[int, int, int]] = []
    for image in images:
        flattened, grid = _preprocess_single_image(image, config=config)
        pixel_values.extend(flattened)
        image_grid_thw.append(grid)
    return Qwen2VLProcessedImages(
        pixel_values=np.asarray(pixel_values, dtype=np.float32),
        image_grid_thw=np.asarray(image_grid_thw, dtype=np.int32),
    )


def expand_image_placeholders(
    texts: list[str],
    *,
    image_grid_thw: np.ndarray,
    merge_size: int,
    image_token: str = _DEFAULT_IMAGE_TOKEN,
) -> list[str]:
    expanded = texts.copy()
    merge_length = merge_size**2
    index = 0
    for text_index, text in enumerate(expanded):
        while image_token in text:
            if index >= int(image_grid_thw.shape[0]):
                raise ValueError("Not enough processed images to expand image tokens")
            num_image_tokens = int(np.prod(image_grid_thw[index])) // merge_length
            text = text.replace(image_token, "<|placeholder|>" * num_image_tokens, 1)
            index += 1
        expanded[text_index] = text.replace("<|placeholder|>", image_token)
    if index != int(image_grid_thw.shape[0]):
        raise ValueError("Processed images remain after placeholder expansion")
    return expanded


def _preprocess_single_image(
    image: Image.Image,
    *,
    config: Qwen2VLImageProcessorConfig,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    if config.do_convert_rgb:
        image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32)
    height, width = int(array.shape[0]), int(array.shape[1])
    resized_height = height
    resized_width = width
    if config.do_resize:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=config.patch_size * config.merge_size,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        image = image.resize(
            (resized_width, resized_height), resample=Resampling.BICUBIC
        )
        array = np.asarray(image, dtype=np.float32)
    if config.do_rescale:
        array = array * config.rescale_factor
    if config.do_normalize:
        mean = np.asarray(config.image_mean, dtype=np.float32)
        std = np.asarray(config.image_std, dtype=np.float32)
        array = (array - mean) / std
    channels_first = np.transpose(array, (2, 0, 1))
    patches = np.expand_dims(channels_first, axis=0)
    remainder = int(patches.shape[0]) % config.temporal_patch_size
    if remainder != 0:
        repeats = np.repeat(
            patches[-1][np.newaxis],
            config.temporal_patch_size - remainder,
            axis=0,
        )
        patches = np.concatenate([patches, repeats], axis=0)
    channel_count = int(patches.shape[1])
    grid_t = int(patches.shape[0]) // config.temporal_patch_size
    grid_h = resized_height // config.patch_size
    grid_w = resized_width // config.patch_size
    patches = patches.reshape(
        grid_t,
        config.temporal_patch_size,
        channel_count,
        grid_h // config.merge_size,
        config.merge_size,
        config.patch_size,
        grid_w // config.merge_size,
        config.merge_size,
        config.patch_size,
    )
    patches = np.transpose(patches, (0, 3, 6, 4, 7, 2, 1, 5, 8))
    flattened = patches.reshape(
        grid_t * grid_h * grid_w,
        channel_count
        * config.temporal_patch_size
        * config.patch_size
        * config.patch_size,
    )
    return flattened, (grid_t, grid_h, grid_w)


def _rgb_triplet(raw: object, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"Qwen2-VL {field_name} must contain exactly 3 values")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


__all__ = [
    "Qwen2VLImageProcessorConfig",
    "Qwen2VLProcessedImages",
    "expand_image_placeholders",
    "preprocess_qwen2_vl_images",
    "smart_resize",
]
