from __future__ import annotations

import math
from typing import TypeAlias

import mlx.core as mx
import numpy as np

from .types import MLXArray

_LatentLike: TypeAlias = MLXArray | tuple[int, ...]

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


class LTX2Scheduler:
    """Owned MLX sigma scheduler matching the upstream LTX-2 default schedule."""

    def execute(
        self,
        *,
        steps: int,
        latent: _LatentLike | None = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        default_number_of_tokens: int = MAX_SHIFT_ANCHOR,
    ) -> tuple[float, ...]:
        if steps <= 0:
            raise ValueError("LTX2Scheduler requires steps > 0")

        tokens = (
            _token_count(latent) if latent is not None else default_number_of_tokens
        )
        sigmas = mx.linspace(1.0, 0.0, steps + 1)

        mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
        bias = base_shift - mm * BASE_SHIFT_ANCHOR
        sigma_shift = tokens * mm + bias
        exp_shift = math.exp(sigma_shift)
        transformed = mx.where(
            sigmas != 0,
            exp_shift / (exp_shift + mx.power(1.0 / sigmas - 1.0, 1)),
            mx.zeros_like(sigmas),
        )
        if stretch and steps > 0:
            one_minus_sigmas = 1.0 - transformed
            last_non_zero = float(one_minus_sigmas[steps - 1])
            scale_factor = last_non_zero / (1.0 - terminal)
            stretched = 1.0 - (one_minus_sigmas / scale_factor)
            transformed = mx.where(transformed != 0, stretched, transformed)
        values = np.asarray(transformed.astype(mx.float32)).reshape(-1)
        return tuple(float(value) for value in values)


def _token_count(latent: _LatentLike) -> int:
    shape = latent.shape if hasattr(latent, "shape") else latent
    if len(shape) < 3:
        raise ValueError(
            "LTX scheduler latent shape must have at least 3 trailing dims"
        )
    tokens = 1
    for size in shape[-3:]:
        tokens *= int(size)
    return tokens
