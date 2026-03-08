from __future__ import annotations

import mlx.core as mx

from .. import _nn_compat as nn
from .feed_forward import GELU


class TextProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        output_dim = hidden_size if out_features is None else out_features
        self.linear1 = nn.Linear(in_features, hidden_size, bias=bias)
        self.act = GELU(approximate="tanh")
        self.linear2 = nn.Linear(hidden_size, output_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.act(self.linear1(x)))
