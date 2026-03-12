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

    def gelu_approx(x: mx.array) -> mx.array: ...

else:
    Module = _nn.Module
    Linear = _nn.Linear
    LayerNorm = _nn.LayerNorm
    gelu_approx = _nn.gelu_approx


__all__ = [
    "LayerNorm",
    "Linear",
    "Module",
    "gelu_approx",
]
