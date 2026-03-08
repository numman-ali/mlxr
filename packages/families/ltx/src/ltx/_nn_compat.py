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

    class Linear(Module):
        weight: mx.array

        def __init__(
            self,
            input_dims: int,
            output_dims: int,
            bias: bool = True,
        ) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class RMSNorm(Module):
        def __init__(self, dims: int, eps: float = 1e-5) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class Dropout(Module):
        def __init__(self, p: float = 0.5) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class Conv1d(Module):
        def __init__(self, *args: object, **kwargs: object) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class Conv2d(Module):
        def __init__(self, *args: object, **kwargs: object) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class ConvTranspose1d(Module):
        def __init__(self, *args: object, **kwargs: object) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class LeakyReLU(Module):
        def __init__(self, *args: object, **kwargs: object) -> None: ...
        def __call__(self, x: mx.array) -> mx.array: ...

    class SiLU(Module):
        def __call__(self, x: mx.array) -> mx.array: ...

    def gelu_approx(x: mx.array) -> mx.array: ...
    def leaky_relu(x: mx.array) -> mx.array: ...
else:
    Module = _nn.Module
    Linear = _nn.Linear
    RMSNorm = _nn.RMSNorm
    Dropout = _nn.Dropout
    Conv1d = _nn.Conv1d
    Conv2d = _nn.Conv2d
    ConvTranspose1d = _nn.ConvTranspose1d
    LeakyReLU = _nn.LeakyReLU
    SiLU = _nn.SiLU
    gelu_approx = _nn.gelu_approx
    leaky_relu = _nn.leaky_relu


__all__ = [
    "Conv1d",
    "Conv2d",
    "ConvTranspose1d",
    "Dropout",
    "LeakyReLU",
    "Linear",
    "Module",
    "RMSNorm",
    "SiLU",
    "gelu_approx",
    "leaky_relu",
]
