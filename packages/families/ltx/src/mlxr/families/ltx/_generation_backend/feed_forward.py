from __future__ import annotations

import mlx.core as mx

from .. import _nn_compat as nn


class GELU(nn.Module):
    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu_approx(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_out: int | None = None,
        mult: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        output_dim = dim if dim_out is None else dim_out
        hidden_dim = int(dim * mult)
        self.proj_in = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_dim, output_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj_out(self.act(self.proj_in(x)))
