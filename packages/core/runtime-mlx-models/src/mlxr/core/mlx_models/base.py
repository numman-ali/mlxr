from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import mlx.core as mx


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params: dict[str, object]) -> BaseModelArgs:
        filtered = {
            key: value
            for key, value in params.items()
            if key in cls.__dataclass_fields__
        }
        return cls(**filtered)


def create_causal_mask(
    length: int,
    offset: int = 0,
    window_size: int | None = None,
    right_padding: mx.array | None = None,
    left_padding: mx.array | None = None,
) -> mx.array:
    right_indices = mx.arange(offset + length)
    left_indices = mx.arange(offset, offset + length) if offset else right_indices
    left_indices = left_indices[:, None]
    right_indices = right_indices[None]
    mask = left_indices >= right_indices
    if window_size is not None:
        mask = mask & (left_indices < right_indices + window_size)
    if right_padding is not None:
        mask = mask & (
            right_indices < mx.expand_dims((offset + length) - right_padding, (1, 2, 3))
        )
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= right_indices)
    return mask


@runtime_checkable
class MaskFactory(Protocol):
    def make_mask(
        self,
        sequence_length: int,
        *,
        return_array: bool,
        window_size: int | None,
    ) -> mx.array | str | None: ...


def create_attention_mask(
    hidden: mx.array,
    cache: object | None = None,
    *,
    window_size: int | None = None,
    return_array: bool = False,
) -> mx.array | str | None:
    sequence_length = int(hidden.shape[1])
    if isinstance(cache, MaskFactory):
        return cache.make_mask(
            sequence_length,
            return_array=return_array,
            window_size=window_size,
        )
    if sequence_length == 1:
        return None
    if return_array or (window_size is not None and sequence_length > window_size):
        return create_causal_mask(sequence_length, window_size=window_size)
    return "causal"
