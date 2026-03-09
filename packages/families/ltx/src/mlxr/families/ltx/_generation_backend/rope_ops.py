from __future__ import annotations

import math
from enum import Enum

import mlx.core as mx
import numpy as np

from .types import MLXArray


class LTXRopeType(str, Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"

    @classmethod
    def from_value(cls, value: object) -> "LTXRopeType":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == cls.INTERLEAVED.value:
                return cls.INTERLEAVED
            if normalized == cls.SPLIT.value:
                return cls.SPLIT
        raise ValueError(f"Unsupported LTX rope type {value!r}")


def apply_rotary_emb(
    tensor: MLXArray,
    pe: tuple[MLXArray, MLXArray],
    rope_type: object,
) -> MLXArray:
    resolved_rope_type = LTXRopeType.from_value(rope_type)
    cos_freqs, sin_freqs = pe
    if resolved_rope_type is LTXRopeType.INTERLEAVED:
        return _apply_interleaved_rotary_emb(tensor, cos_freqs, sin_freqs)
    return _apply_split_rotary_emb(tensor, cos_freqs, sin_freqs)


def precompute_freqs_cis(
    positions: MLXArray,
    *,
    dim: int,
    out_dtype: mx.Dtype | None = None,
    theta: float,
    max_pos: list[int],
    use_middle_indices_grid: bool,
    num_attention_heads: int,
    rope_type: str,
    double_precision: bool,
) -> tuple[MLXArray, MLXArray]:
    resolved_rope_type = LTXRopeType.from_value(rope_type)
    indices = _generate_frequency_grid(
        positional_embedding_theta=theta,
        positional_embedding_max_pos_count=int(positions.shape[1]),
        inner_dim=dim,
        double_precision=double_precision,
    )
    freqs = _generate_freqs(
        indices=indices,
        positions=positions,
        max_pos=max_pos,
        use_middle_indices_grid=use_middle_indices_grid,
    )

    if resolved_rope_type is LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        pad_size = expected_freqs - int(freqs.shape[-1])
        cos_freq, sin_freq = _split_freqs_cis(
            freqs,
            pad_size=pad_size,
            num_attention_heads=num_attention_heads,
        )
    else:
        cos_freq, sin_freq = _interleaved_freqs_cis(
            freqs,
            pad_size=dim % (2 * int(positions.shape[1])),
        )
    if out_dtype is not None:
        cos_freq = cos_freq.astype(out_dtype)
        sin_freq = sin_freq.astype(out_dtype)
    return cos_freq, sin_freq


def _apply_interleaved_rotary_emb(
    tensor: MLXArray,
    cos_freqs: MLXArray,
    sin_freqs: MLXArray,
) -> MLXArray:
    if tensor.shape[-1] % 2 != 0:
        raise ValueError("Interleaved RoPE requires an even channel width")
    reshaped = mx.reshape(
        tensor,
        tuple(int(size) for size in tensor.shape[:-1]) + (tensor.shape[-1] // 2, 2),
    )
    first = reshaped[..., 0]
    second = reshaped[..., 1]
    rotated = mx.stack([-second, first], axis=-1)
    rotated = mx.reshape(rotated, tensor.shape)
    return tensor * cos_freqs + rotated * sin_freqs


def _apply_split_rotary_emb(
    tensor: MLXArray,
    cos_freqs: MLXArray,
    sin_freqs: MLXArray,
) -> MLXArray:
    needs_reshape = tensor.ndim != 4 and cos_freqs.ndim == 4
    original_shape = tuple(int(size) for size in tensor.shape)
    if needs_reshape:
        batch, heads, tokens, _ = (int(size) for size in cos_freqs.shape)
        tensor = mx.reshape(tensor, (batch, tokens, heads, -1))
        tensor = mx.swapaxes(tensor, 1, 2)

    if tensor.shape[-1] % 2 != 0:
        raise ValueError("Split RoPE requires an even channel width")
    half_width = int(tensor.shape[-1] // 2)
    first_half = tensor[..., :half_width]
    second_half = tensor[..., half_width:]
    rotated_first = first_half * cos_freqs - second_half * sin_freqs
    rotated_second = second_half * cos_freqs + first_half * sin_freqs
    output = mx.concatenate([rotated_first, rotated_second], axis=-1)

    if needs_reshape:
        output = mx.swapaxes(output, 1, 2)
        output = mx.reshape(output, original_shape)
    return output


def _generate_frequency_grid(
    *,
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
    double_precision: bool,
) -> MLXArray:
    theta = positional_embedding_theta
    if theta <= 0:
        raise ValueError("RoPE theta must be positive")
    n_elem = 2 * positional_embedding_max_pos_count
    count = inner_dim // n_elem
    if count < 1:
        raise ValueError(
            "RoPE inner_dim is too small for the requested positional dimensions"
        )
    if double_precision:
        exponents = np.linspace(0.0, 1.0, count, dtype=np.float64)
    else:
        exponents = np.linspace(0.0, 1.0, count, dtype=np.float32)
    indices = np.power(theta, exponents) * (math.pi / 2.0)
    return mx.array(indices)


def _generate_freqs(
    *,
    indices: MLXArray,
    positions: MLXArray,
    max_pos: list[int],
    use_middle_indices_grid: bool,
) -> MLXArray:
    indices_grid: MLXArray
    if positions.ndim == 4 and int(positions.shape[-1]) == 2:
        if use_middle_indices_grid:
            indices_grid = (positions[..., 0] + positions[..., 1]) / 2.0
        else:
            indices_grid = positions[..., 0]
    else:
        indices_grid = positions

    n_pos_dims = int(indices_grid.shape[1])
    if n_pos_dims != len(max_pos):
        raise ValueError(
            f"RoPE positions have {n_pos_dims} dimensions but max_pos has {len(max_pos)}"
        )
    fractional_positions = mx.stack(
        [indices_grid[:, index] / float(max_pos[index]) for index in range(n_pos_dims)],
        axis=-1,
    )
    freqs = indices * (fractional_positions[..., None] * 2.0 - 1.0)
    freqs = mx.swapaxes(freqs, -1, -2)
    return mx.reshape(
        freqs,
        (
            int(freqs.shape[0]),
            int(freqs.shape[1]),
            int(freqs.shape[2] * freqs.shape[3]),
        ),
    )


def _split_freqs_cis(
    freqs: MLXArray,
    *,
    pad_size: int,
    num_attention_heads: int,
) -> tuple[MLXArray, MLXArray]:
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)
    if pad_size > 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    batch = int(cos_freq.shape[0])
    tokens = int(cos_freq.shape[1])
    cos_freq = mx.reshape(cos_freq, (batch, tokens, num_attention_heads, -1))
    sin_freq = mx.reshape(sin_freq, (batch, tokens, num_attention_heads, -1))
    return mx.swapaxes(cos_freq, 1, 2), mx.swapaxes(sin_freq, 1, 2)


def _interleaved_freqs_cis(
    freqs: MLXArray,
    *,
    pad_size: int,
) -> tuple[MLXArray, MLXArray]:
    cos_freq = mx.repeat(mx.cos(freqs), 2, axis=-1)
    sin_freq = mx.repeat(mx.sin(freqs), 2, axis=-1)
    if pad_size > 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)
    return cos_freq, sin_freq
