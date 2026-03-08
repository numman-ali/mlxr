from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:

    class ModuleBase: ...

    class LinearLike(Protocol):
        weight: mx.array

        def __call__(self, x: mx.array) -> mx.array: ...

    class EmbeddingLike(Protocol):
        weight: mx.array

        def __call__(self, inputs: mx.array) -> mx.array: ...

    class RopeLike(Protocol):
        def __call__(self, x: mx.array, *, offset: int = 0) -> mx.array: ...

    def gelu_approx(x: mx.array) -> mx.array: ...

    def build_linear(
        input_dims: int,
        output_dims: int,
        *,
        bias: bool,
    ) -> LinearLike: ...

    def build_embedding(
        num_embeddings: int,
        dims: int,
    ) -> EmbeddingLike: ...

    def build_rope(
        dims: int,
        *,
        traditional: bool,
        base: float,
        scale: float,
    ) -> RopeLike: ...

else:
    ModuleBase = nn.Module
    RopeLike = nn.Module
    gelu_approx = nn.gelu_approx

    def build_linear(
        input_dims: int,
        output_dims: int,
        *,
        bias: bool,
    ) -> nn.Linear:
        return nn.Linear(input_dims, output_dims, bias=bias)

    def build_embedding(
        num_embeddings: int,
        dims: int,
    ) -> nn.Embedding:
        return nn.Embedding(num_embeddings, dims)

    def build_rope(
        dims: int,
        *,
        traditional: bool,
        base: float,
        scale: float,
    ) -> nn.Module:
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)


__all__ = [
    "ModuleBase",
    "RopeLike",
    "build_embedding",
    "build_linear",
    "build_rope",
    "gelu_approx",
    "mx",
    "nn",
]
