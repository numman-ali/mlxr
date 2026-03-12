from __future__ import annotations

import math

import mlx.core as mx

from .. import _nn_compat as nn
from .config import QwenImageTransformerConfig

_TIMESTEP_EMBED_DIM = 256


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def timestep_embedding(
    timesteps: mx.array,
    dim: int,
    *,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1000.0,
    max_period: int = 10_000,
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
            [embedding, mx.zeros((int(embedding.shape[0]), 1), dtype=embedding.dtype)],
            axis=-1,
        )
    return embedding


class TimestepProjection(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(_TIMESTEP_EMBED_DIM, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, timesteps: mx.array) -> mx.array:
        projected = timestep_embedding(timesteps, _TIMESTEP_EMBED_DIM)
        hidden = self.linear_1(projected.astype(timesteps.dtype))
        hidden = silu(hidden)
        return self.linear_2(hidden)


class QwenEmbedRope:
    def __init__(self, config: QwenImageTransformerConfig) -> None:
        self.axes_dims = config.axes_dims_rope
        self.theta = 10_000.0
        self.scale_rope = True
        pos_index = mx.arange(4096, dtype=mx.float32)
        # Match upstream ordering exactly: the negative table is built from the
        # reversed positive index range, not by taking the negative of the
        # forward range directly. That ordering matters because scaled RoPE
        # later slices from the tail of this table for centered spatial axes.
        neg_index = mx.arange(4096, dtype=mx.float32)[::-1] * -1.0 - 1.0
        self.pos_freqs = mx.concatenate(
            [
                self._rope_params(pos_index, self.axes_dims[0]),
                self._rope_params(pos_index, self.axes_dims[1]),
                self._rope_params(pos_index, self.axes_dims[2]),
            ],
            axis=1,
        )
        self.neg_freqs = mx.concatenate(
            [
                self._rope_params(neg_index, self.axes_dims[0]),
                self._rope_params(neg_index, self.axes_dims[1]),
                self._rope_params(neg_index, self.axes_dims[2]),
            ],
            axis=1,
        )

    def _rope_params(self, index: mx.array, dim: int) -> mx.array:
        if dim % 2 != 0:
            raise ValueError("Qwen-Image RoPE axis dimensions must be even")
        steps = mx.arange(0, dim, 2, dtype=mx.float32) / float(dim)
        freqs = index[:, None] * (1.0 / (self.theta**steps))[None, :]
        return mx.exp(1j * freqs.astype(mx.complex64))

    def __call__(
        self,
        image_shape: tuple[int, int, int] | list[tuple[int, int, int]],
        *,
        max_text_seq_len: int,
    ) -> tuple[mx.array, mx.array]:
        shapes = image_shape if isinstance(image_shape, list) else [image_shape]
        image_freqs_list: list[mx.array] = []
        max_vid_index = 0
        for index, shape in enumerate(shapes):
            frame, height, width = shape
            image_freqs, current_max = self._image_freqs(
                frame,
                height,
                width,
                frame_offset=index,
            )
            image_freqs_list.append(image_freqs)
            max_vid_index = max(max_vid_index, current_max)
        image_freqs = mx.concatenate(image_freqs_list, axis=0)
        text_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_text_seq_len]
        return image_freqs, text_freqs

    def _image_freqs(
        self,
        frame: int,
        height: int,
        width: int,
        *,
        frame_offset: int = 0,
    ) -> tuple[mx.array, int]:
        frame_dim = self.axes_dims[0] // 2
        height_dim = self.axes_dims[1] // 2
        width_dim = self.axes_dims[2] // 2
        frame_freqs = self.pos_freqs[frame_offset : frame_offset + frame, :frame_dim]
        frame_freqs = mx.broadcast_to(
            frame_freqs[:, None, None, :],
            (frame, height, width, frame_dim),
        )
        if self.scale_rope:
            height_base = mx.concatenate(
                [
                    self.neg_freqs[
                        -(height - height // 2) :, frame_dim : frame_dim + height_dim
                    ],
                    self.pos_freqs[: height // 2, frame_dim : frame_dim + height_dim],
                ],
                axis=0,
            )
            width_base = mx.concatenate(
                [
                    self.neg_freqs[-(width - width // 2) :, frame_dim + height_dim :],
                    self.pos_freqs[: width // 2, frame_dim + height_dim :],
                ],
                axis=0,
            )
        else:
            height_base = self.pos_freqs[:height, frame_dim : frame_dim + height_dim]
            width_base = self.pos_freqs[:width, frame_dim + height_dim :]
        height_freqs = mx.broadcast_to(
            height_base[None, :, None, :],
            (frame, height, width, height_dim),
        )
        width_freqs = mx.broadcast_to(
            width_base[None, None, :, :],
            (frame, height, width, width_dim),
        )
        freqs = mx.concatenate([frame_freqs, height_freqs, width_freqs], axis=-1)
        max_vid_index = (
            max(height // 2, width // 2) if self.scale_rope else max(height, width)
        )
        return freqs.reshape(frame * height * width, -1), max_vid_index


def apply_rotary_emb_qwen(x: mx.array, freqs_cis: mx.array) -> mx.array:
    original_shape = tuple(int(size) for size in x.shape)
    complex_view = x.astype(mx.float32).reshape(*original_shape[:-1], -1, 2)
    complex_view = complex_view[..., 0] + 1j * complex_view[..., 1]
    rotated = complex_view * freqs_cis[None, None, :, :]
    stacked = mx.stack([mx.real(rotated), mx.imag(rotated)], axis=-1)
    return stacked.reshape(original_shape).astype(x.dtype)
