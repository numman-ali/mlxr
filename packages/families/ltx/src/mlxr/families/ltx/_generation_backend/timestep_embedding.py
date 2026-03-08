from __future__ import annotations

import math

import mlx.core as mx

from .types import MLXArray


def get_timestep_embedding(
    timesteps: MLXArray,
    *,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10_000,
) -> MLXArray:
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1D array")
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be positive")

    half_dim = embedding_dim // 2
    if half_dim == 0:
        return mx.zeros((int(timesteps.shape[0]), 1), dtype=mx.float32)
    if half_dim == int(downscale_freq_shift):
        raise ValueError("downscale_freq_shift makes timestep frequencies singular")

    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
    exponent = exponent / float(half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = mx.expand_dims(timesteps.astype(mx.float32), axis=1) * mx.expand_dims(
        emb, axis=0
    )
    emb = scale * emb
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class Timesteps:
    def __init__(
        self,
        *,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1.0,
    ) -> None:
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: MLXArray) -> MLXArray:
        return get_timestep_embedding(
            timesteps,
            embedding_dim=self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
