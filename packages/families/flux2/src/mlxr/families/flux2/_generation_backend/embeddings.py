from __future__ import annotations

import math

import mlx.core as mx

from .. import _nn_compat as nn
from .constants import _MAX_PERIOD


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _timestep_embedding(
    timesteps: mx.array,
    dim: int,
    *,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = _MAX_PERIOD,
) -> mx.array:
    half_dim = dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / float(half_dim - downscale_freq_shift)
    embedding = timesteps[:, None].astype(mx.float32) * mx.exp(exponent)[None, :]
    embedding = scale * embedding
    embedding = mx.concatenate([mx.sin(embedding), mx.cos(embedding)], axis=-1)
    if flip_sin_to_cos:
        embedding = mx.concatenate(
            [embedding[:, half_dim:], embedding[:, :half_dim]],
            axis=-1,
        )
    if dim % 2:
        embedding = mx.concatenate(
            [embedding, mx.zeros((embedding.shape[0], 1), dtype=embedding.dtype)],
            axis=-1,
        )
    return embedding.astype(timesteps.dtype)


class EmbedND(nn.Module):
    def __init__(self, *, theta: float, axes_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims

    def __call__(self, ids: mx.array) -> mx.array:
        from .transformer import _rope

        rope = mx.concatenate(
            [
                _rope(ids[..., index], axis_dim, self.theta)
                for index, axis_dim in enumerate(self.axes_dims)
            ],
            axis=-3,
        )
        return rope[:, None]


class TimestepProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels, bias=False)
        self.linear_2 = nn.Linear(out_channels, out_channels, bias=False)

    def __call__(self, sample: mx.array) -> mx.array:
        hidden = self.linear_1(sample)
        hidden = _silu(hidden)
        return self.linear_2(hidden)


class TimestepGuidanceEmbeddings(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_size: int,
        guidance_embeds: bool,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.timestep_embedder = TimestepProjection(in_channels, hidden_size)
        self.guidance_embedder = (
            TimestepProjection(in_channels, hidden_size) if guidance_embeds else None
        )

    def __call__(
        self,
        timestep: mx.array,
        guidance: mx.array | None,
    ) -> mx.array:
        timestep_proj = _timestep_embedding(
            timestep,
            self.in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        timestep_emb = self.timestep_embedder(timestep_proj.astype(timestep.dtype))
        if guidance is None or self.guidance_embedder is None:
            return timestep_emb
        guidance_proj = _timestep_embedding(
            guidance,
            self.in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        guidance_emb = self.guidance_embedder(guidance_proj.astype(guidance.dtype))
        return timestep_emb + guidance_emb
