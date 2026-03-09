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

    # patchify_video packs channels as (c, p_t, p_w, p_h) to match official
    # LTX ops. Mirror that exact factorization here before restoring
    # (f * p_t, h * p_h, w * p_w).
    unpacked = mx.reshape(
        x,
        (
            batch,
            channels,
            patch_size_t,
            patch_size_hw,  # packed width axis
            patch_size_hw,  # packed height axis
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
