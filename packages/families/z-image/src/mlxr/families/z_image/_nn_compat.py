from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as _nn

if TYPE_CHECKING:

    class Module:
        def __init__(self, *args: object, **kwargs: object) -> None: ...
        def parameters(self) -> object: ...
        def load_weights(
            self, weights: list[tuple[str, mx.array]], *, strict: bool = ...
        ) -> None: ...
        def set_dtype(self, dtype: mx.Dtype) -> "Module": ...

    class Linear(Module):
        weight: mx.array

        def __init__(
            self,
            input_dims: int,
            output_dims: int,
            bias: bool = True,
        ) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class LayerNorm(Module):
        def __init__(
            self,
            dims: int,
            eps: float = 1e-5,
            affine: bool = True,
            bias: bool = True,
        ) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class GroupNorm(Module):
        def __init__(
            self,
            num_groups: int,
            dims: int,
            eps: float = 1e-5,
            affine: bool = True,
            pytorch_compatible: bool = False,
        ) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class Conv2d(Module):
        weight: mx.array

        def __init__(self, *args: object, **kwargs: object) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class SiLU(Module):
        def __call__(self, x: mx.array) -> mx.array: ...

else:
    Module = _nn.Module
    Linear = _nn.Linear
    LayerNorm = _nn.LayerNorm
    GroupNorm = _nn.GroupNorm
    Conv2d = _nn.Conv2d
    SiLU = _nn.SiLU


__all__ = [
    "Conv2d",
    "GroupNorm",
    "LayerNorm",
    "Linear",
    "Module",
    "SiLU",
]
