from __future__ import annotations

import math

import mlx.core as mx

from .. import _nn_compat as nn
from .rope_ops import LTXRopeType, apply_rotary_emb
from .types import MLXArray


def scaled_dot_product_attention(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    heads: int,
    mask: MLXArray | None = None,
) -> MLXArray:
    batch_size, query_length, inner_dim = query.shape
    _, key_length, _ = key.shape
    dim_head = inner_dim // heads

    query = mx.reshape(query, (batch_size, query_length, heads, dim_head))
    key = mx.reshape(key, (batch_size, key_length, heads, dim_head))
    value = mx.reshape(value, (batch_size, key_length, heads, dim_head))

    query = mx.swapaxes(query, 1, 2)
    key = mx.swapaxes(key, 1, 2)
    value = mx.swapaxes(value, 1, 2)

    if mask is not None:
        if mask.ndim == 2:
            mask = mx.expand_dims(mask, axis=0)
        if mask.ndim == 3:
            mask = mx.expand_dims(mask, axis=1)

    scale = 1.0 / math.sqrt(dim_head)
    output = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=scale,
        mask=mask,
    )
    output = mx.swapaxes(output, 1, 2)
    return mx.reshape(output, (batch_size, query_length, heads * dim_head))


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        *,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType | str = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        context_width = query_dim if context_dim is None else context_dim
        inner_dim = dim_head * heads

        self.rope_type = LTXRopeType.from_value(rope_type)
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_width, inner_dim, bias=True)
        self.to_v = nn.Linear(context_width, inner_dim, bias=True)
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.to_gate_logits = (
            nn.Linear(query_dim, heads, bias=True) if apply_gated_attention else None
        )
        self.to_out = nn.Linear(inner_dim, query_dim, bias=True)

    def __call__(
        self,
        x: MLXArray,
        *,
        context: MLXArray | None = None,
        mask: MLXArray | None = None,
        pe: tuple[MLXArray, MLXArray] | None = None,
        k_pe: tuple[MLXArray, MLXArray] | None = None,
        perturbation_mask: MLXArray | None = None,
        all_perturbed: bool = False,
    ) -> MLXArray:
        context_tensor = x if context is None else context
        value = self.to_v(context_tensor)

        if all_perturbed:
            output = value
        else:
            query = self.q_norm(self.to_q(x))
            key = self.k_norm(self.to_k(context_tensor))
            if pe is not None:
                query = apply_rotary_emb(query, pe, self.rope_type)
                key = apply_rotary_emb(
                    key, pe if k_pe is None else k_pe, self.rope_type
                )
            output = scaled_dot_product_attention(query, key, value, self.heads, mask)
            if perturbation_mask is not None:
                output = output * perturbation_mask + value * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            gates = 2.0 * mx.sigmoid(self.to_gate_logits(x))
            batch_size, seq_len, _ = output.shape
            reshaped = mx.reshape(
                output, (batch_size, seq_len, self.heads, self.dim_head)
            )
            output = mx.reshape(
                reshaped * mx.expand_dims(gates, axis=-1),
                (batch_size, seq_len, self.heads * self.dim_head),
            )
        return self.to_out(output)
