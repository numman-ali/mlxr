from __future__ import annotations

import mlx.core as mx

from .types import MLXArray


def unpatchify_video(
    x: MLXArray,
    *,
    patch_size_hw: int = 4,
    patch_size_t: int = 1,
) -> MLXArray:
    batch, packed_channels, frames, height, width = x.shape
    channels = packed_channels // (patch_size_hw * patch_size_hw * patch_size_t)

    unpacked = mx.reshape(
        x,
        (
            batch,
            channels,
            patch_size_t,
            patch_size_hw,
            patch_size_hw,
            frames,
            height,
            width,
        ),
    )
    unpacked = mx.transpose(unpacked, (0, 1, 5, 2, 6, 4, 7, 3))
    return mx.reshape(
        unpacked,
        (
            batch,
            channels,
            frames * patch_size_t,
            height * patch_size_hw,
            width * patch_size_hw,
        ),
    )
