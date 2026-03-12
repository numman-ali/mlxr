from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import Qwen3Config

from ._nn_compat import ModuleBase, build_linear
from .base import BaseModelArgs, create_attention_mask, create_causal_mask
from .cache import KVCache, RotatingKVCache
from .rope import initialize_rope

Qwen3TextConfig = Qwen3Config


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 22016
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = None
    rope_parameters: dict[str, object] | None = None

    @classmethod
    def from_text_config(cls, config: Qwen3TextConfig) -> ModelArgs:
        raw = config.to_dict()
        rope_parameters = raw.get("rope_parameters")
        raw_rope_theta = raw.get("rope_theta")
        if not isinstance(raw_rope_theta, (int, float)):
            raw_rope_theta = (
                rope_parameters.get("rope_theta", 1_000_000.0)
                if isinstance(rope_parameters, dict)
                else 1_000_000.0
            )
        return cls(
            model_type=str(raw.get("model_type", "qwen3")),
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            vocab_size=int(raw["vocab_size"]),
            rope_theta=float(raw_rope_theta),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            attention_bias=bool(raw.get("attention_bias", False)),
            use_sliding_window=bool(raw.get("use_sliding_window", False)),
            sliding_window=(
                int(raw["sliding_window"])
                if isinstance(raw.get("sliding_window"), int)
                else None
            ),
            rope_parameters=(
                rope_parameters if isinstance(rope_parameters, dict) else None
            ),
        )


class RMSNorm(ModuleBase):
    def __init__(self, dims: int, *, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        normalized = mx.fast.rms_norm(
            x.astype(mx.float32),
            self.weight.astype(mx.float32),
            self.eps,
        )
        return normalized.astype(x.dtype)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class Attention(ModuleBase):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.repeats = self.num_heads // self.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = build_linear(
            dim, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = build_linear(
            dim, self.num_key_value_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = build_linear(
            dim, self.num_key_value_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = build_linear(
            self.num_heads * self.head_dim, dim, bias=args.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            dims=self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_parameters,
            max_position_embeddings=args.max_position_embeddings,
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
        return self.down_proj(_silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(ModuleBase):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), mask, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TokenEmbedding(ModuleBase):
    def __init__(self, num_embeddings: int, dims: int) -> None:
        super().__init__()
        self.weight = mx.zeros((num_embeddings, dims))

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.weight[inputs]


class Qwen3Model(ModuleBase):
    def __init__(self, config: Qwen3TextConfig) -> None:
        super().__init__()
        self.config = config
        self.args = ModelArgs.from_text_config(config)
        self.embed_tokens = TokenEmbedding(self.args.vocab_size, self.args.hidden_size)
        self.layers = [
            TransformerBlock(self.args) for _ in range(self.args.num_hidden_layers)
        ]
        self.norm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)
        self.window_size = (
            self.args.sliding_window if self.args.use_sliding_window else None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache | None] | None = None,
        input_embeddings: mx.array | None = None,
        attention_mask: mx.array | None = None,
        *,
        return_hidden_states: bool = False,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        hidden = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        if cache is None:
            cache = [None] * len(self.layers)
        mask = _attention_mask(
            hidden,
            cache=cache[0],
            window_size=self.window_size,
            attention_mask=attention_mask,
        )
        hidden_states: list[mx.array] = [hidden] if return_hidden_states else []
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden = layer(hidden, mask, layer_cache)
            if return_hidden_states:
                hidden_states.append(hidden)
        hidden = self.norm(hidden)
        if return_hidden_states:
            hidden_states.append(hidden)
            return hidden, hidden_states
        return hidden


def load_local_qwen3_model(model_path: Path) -> Qwen3Model:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Qwen3 config.json not found under '{model_path}'")
    config = Qwen3TextConfig.from_dict(json.loads(config_path.read_text("utf-8")))
    model = Qwen3Model(config)
    loaded_keys: set[str] = set()
    for weight_file in _weight_files(model_path):
        shard = mx.load(str(weight_file))
        if not isinstance(shard, dict):
            raise RuntimeError(
                f"Qwen3 weight shard '{weight_file}' did not load into a weight mapping"
            )
        items: list[tuple[str, mx.array]] = []
        for key, value in shard.items():
            if not key.startswith("model."):
                continue
            stripped = key[len("model.") :]
            if hasattr(value, "dtype") and value.dtype == mx.float32:
                value = value.astype(mx.bfloat16)
            items.append((stripped, value))
            loaded_keys.add(stripped)
        if items:
            model.load_weights(items, strict=False)
        del shard
        mx.clear_cache()
    _assert_loaded_parameter_coverage(model, loaded_keys=loaded_keys)
    return model


def _weight_files(model_path: Path) -> tuple[Path, ...]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text("utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(
                f"Qwen3 weight index '{index_path}' is missing a valid weight_map"
            )
        ordered_names = list(dict.fromkeys(str(value) for value in weight_map.values()))
        return tuple(model_path / name for name in ordered_names)
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        return (single_file,)
    files = tuple(sorted(model_path.glob("*.safetensors")))
    if files:
        return files
    raise ValueError(f"No Qwen3 safetensors weights found under '{model_path}'")


def _assert_loaded_parameter_coverage(
    model: Qwen3Model, *, loaded_keys: set[str]
) -> None:
    from mlx.utils import tree_flatten

    expected = tree_flatten(model.parameters(), destination={})
    expected_keys = {str(key) for key in expected}
    missing = expected_keys - loaded_keys
    unexpected = loaded_keys - expected_keys
    if not missing and not unexpected:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing {len(missing)} keys")
    if unexpected:
        details.append(f"unexpected {len(unexpected)} keys")
    raise RuntimeError(f"Qwen3 weight coverage mismatch: {'; '.join(details)}")


def _attention_mask(
    hidden: mx.array,
    *,
    cache: KVCache | RotatingKVCache | None,
    window_size: int | None,
    attention_mask: mx.array | None,
) -> mx.array | str | None:
    if attention_mask is None:
        return create_attention_mask(hidden, cache, window_size=window_size)
    if attention_mask.ndim != 2:
        raise ValueError("Qwen3 attention_mask must be a rank-2 array")
    attention_mask_np = np.asarray(attention_mask)
    if attention_mask_np.ndim != 2:
        raise ValueError("Qwen3 attention_mask must be a rank-2 array")
    left_padding_values: list[int] = []
    right_padding_values: list[int] = []
    for row in attention_mask_np:
        valid_indices = np.flatnonzero(row)
        if valid_indices.size == 0:
            raise ValueError(
                "Qwen3 attention_mask must contain at least one valid token"
            )
        start = int(valid_indices[0])
        end = int(valid_indices[-1]) + 1
        if not np.all(row[start:end] == 1):
            raise ValueError(
                "Qwen3 attention_mask must contain one contiguous valid span"
            )
        left_padding_values.append(start)
        right_padding_values.append(int(row.shape[0] - end))
    left_padding = mx.array(left_padding_values, dtype=mx.int32)
    right_padding = mx.array(right_padding_values, dtype=mx.int32)
    return create_causal_mask(
        int(hidden.shape[1]),
        window_size=window_size,
        right_padding=right_padding,
        left_padding=left_padding,
    )


__all__ = ["Qwen3Model", "Qwen3TextConfig", "load_local_qwen3_model"]
