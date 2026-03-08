from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from transformers import Gemma3TextConfig

from ._nn_compat import (
    ModuleBase,
    build_embedding,
    build_linear,
    gelu_approx,
)
from .base import BaseModelArgs, create_attention_mask
from .cache import KVCache, RotatingKVCache
from .rope import initialize_rope

TextConfig = Gemma3TextConfig


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 1152
    num_hidden_layers: int = 26
    intermediate_size: int = 6912
    num_attention_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 262144
    num_key_value_heads: int = 1
    rope_theta: float = 1_000_000.0
    query_pre_attn_scalar: float = 256.0
    sliding_window: int = 512
    max_position_embeddings: int = 32768
    attention_bias: bool = False
    rope_parameters: dict[str, object] | None = None
    layer_types: list[str] | None = None
    _sliding_window_pattern: int = 6

    @property
    def sliding_window_pattern(self) -> int:
        if self.layer_types is not None:
            return max(
                1, sum(1 for layer in self.layer_types if layer == "sliding_attention")
            )
        return self._sliding_window_pattern

    @classmethod
    def from_text_config(cls, config: TextConfig) -> ModelArgs:
        raw = config.to_dict()
        rope_scaling = raw.get("rope_parameters")
        return cls(
            model_type=str(raw.get("model_type", "gemma3_text")),
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            head_dim=int(raw["head_dim"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            vocab_size=int(raw["vocab_size"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            rope_theta=float(raw.get("rope_theta", 1_000_000.0)),
            query_pre_attn_scalar=float(
                raw.get("query_pre_attn_scalar", raw["head_dim"])
            ),
            sliding_window=int(raw.get("sliding_window", 512)),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            attention_bias=bool(raw.get("attention_bias", False)),
            rope_parameters=(rope_scaling if isinstance(rope_scaling, dict) else None),
            layer_types=(
                [str(value) for value in raw["layer_types"]]
                if isinstance(raw.get("layer_types"), list)
                else None
            ),
            _sliding_window_pattern=int(raw.get("_sliding_window_pattern", 6)),
        )


class RMSNorm(ModuleBase):
    def __init__(self, dims: int, *, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(ModuleBase):
    def __init__(self, args: ModelArgs, *, layer_index: int) -> None:
        super().__init__()
        dim = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.repeats = self.num_heads // self.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = float(args.query_pre_attn_scalar) ** -0.5
        self.q_proj = build_linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = build_linear(
            dim, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = build_linear(
            dim, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = build_linear(self.num_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.is_sliding = (layer_index + 1) % args.sliding_window_pattern != 0
        rope_base = 10_000.0 if self.is_sliding else args.rope_theta
        self.rope = initialize_rope(
            dims=self.head_dim,
            base=rope_base,
            traditional=False,
            scaling_config=None if self.is_sliding else args.rope_parameters,
            max_position_embeddings=None
            if self.is_sliding
            else args.max_position_embeddings,
        )

    def __call__(
        self,
        hidden: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden.shape
        queries = self.q_proj(hidden)
        keys = self.k_proj(hidden)
        values = self.v_proj(hidden)
        queries = queries.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries, offset=0)
            keys = self.rope(keys, offset=0)

        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        return self.o_proj(output)


class MLP(ModuleBase):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = build_linear(dim, hidden_dim, bias=False)
        self.down_proj = build_linear(hidden_dim, dim, bias=False)
        self.up_proj = build_linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(ModuleBase):
    def __init__(self, args: ModelArgs, *, layer_index: int) -> None:
        super().__init__()
        self.self_attn = Attention(args, layer_index=layer_index)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        hidden: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        residual = self.self_attn(self.input_layernorm(hidden), mask, cache)
        hidden = hidden + self.post_attention_layernorm(residual)
        residual = self.mlp(self.pre_feedforward_layernorm(hidden))
        return hidden + self.post_feedforward_layernorm(residual)


class TokenEmbedding(ModuleBase):
    def __init__(self, num_embeddings: int, dims: int) -> None:
        super().__init__()
        self._embedding = build_embedding(num_embeddings, dims)

    @property
    def weight(self) -> mx.array:
        return self._embedding.weight

    def __call__(self, inputs: mx.array) -> mx.array:
        return self._embedding(inputs)

    def as_linear(self, hidden: mx.array) -> mx.array:
        return hidden @ self.weight.T


class Gemma3Model(ModuleBase):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.config = config
        self.args = ModelArgs.from_text_config(config)
        self.window_size = self.args.sliding_window
        self.sliding_window_pattern = self.args.sliding_window_pattern
        self.vocab_size = self.args.vocab_size
        self.num_hidden_layers = self.args.num_hidden_layers
        self.embed_tokens = TokenEmbedding(self.args.vocab_size, self.args.hidden_size)
        self.layers = [
            TransformerBlock(self.args, layer_index=layer_index)
            for layer_index in range(self.args.num_hidden_layers)
        ]
        self.norm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache | None] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        hidden *= mx.array(self.args.hidden_size**0.5, mx.bfloat16).astype(hidden.dtype)
        if cache is None:
            cache = [None] * len(self.layers)
        global_mask = create_attention_mask(
            hidden, cache[self.sliding_window_pattern - 1]
        )
        if self.sliding_window_pattern > 1:
            sliding_window_mask = create_attention_mask(
                hidden,
                cache[0],
                window_size=self.window_size,
            )
        else:
            sliding_window_mask = None
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            is_global = (
                index % self.sliding_window_pattern == self.sliding_window_pattern - 1
            )
            mask = global_mask if is_global else sliding_window_mask
            hidden = layer(hidden, mask, layer_cache)
        return self.norm(hidden)

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        caches: list[KVCache | RotatingKVCache] = []
        for index in range(self.args.num_hidden_layers):
            if (
                index % self.args.sliding_window_pattern
                == self.args.sliding_window_pattern - 1
            ):
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches
