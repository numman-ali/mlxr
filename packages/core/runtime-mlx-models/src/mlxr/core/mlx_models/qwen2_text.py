from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
)

from ._nn_compat import ModuleBase, build_embedding, build_linear
from .base import BaseModelArgs, create_attention_mask, create_causal_mask
from .cache import KVCache, RotatingKVCache

Qwen2TextConfig = Qwen2_5_VLTextConfig


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    intermediate_size: int = 18944
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 152064
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 128000
    attention_dropout: float = 0.0
    use_sliding_window: bool = False
    sliding_window: int | None = None
    rope_parameters: dict[str, object] | None = None

    @classmethod
    def from_text_config(cls, config: Qwen2TextConfig) -> ModelArgs:
        raw = config.to_dict()
        rope_parameters = raw.get("rope_parameters") or raw.get("rope_scaling")
        hidden_size = int(raw["hidden_size"])
        num_attention_heads = int(raw["num_attention_heads"])
        return cls(
            model_type=str(raw.get("model_type", "qwen2_5_vl_text")),
            hidden_size=hidden_size,
            num_hidden_layers=int(raw["num_hidden_layers"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw.get("head_dim", hidden_size // num_attention_heads)),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            vocab_size=int(raw["vocab_size"]),
            rope_theta=float(raw.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            attention_dropout=float(raw.get("attention_dropout", 0.0)),
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


def _repeat_key_values(values: mx.array, repeats: int) -> mx.array:
    if repeats == 1:
        return values
    batch_size, num_heads, sequence_length, head_dim = values.shape
    expanded = mx.expand_dims(values, axis=2)
    expanded = mx.broadcast_to(
        expanded,
        (batch_size, num_heads, repeats, sequence_length, head_dim),
    )
    return expanded.reshape(batch_size, num_heads * repeats, sequence_length, head_dim)


def _rotate_half(x: mx.array) -> mx.array:
    half = int(x.shape[-1]) // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_multimodal_sections(
    values: mx.array,
    sections: tuple[int, ...],
) -> mx.array:
    split_lengths = [section * 2 for section in sections]
    if sum(split_lengths) != int(values.shape[-1]):
        raise ValueError(
            "Qwen2 multimodal rope sections do not sum to the attention head dimension"
        )
    parts: list[mx.array] = []
    start = 0
    for length in split_lengths:
        end = start + length
        parts.append(values[..., start:end])
        start = end
    remixed = [part[i % 3] for i, part in enumerate(parts)]
    combined = mx.concatenate(remixed, axis=-1)
    return mx.expand_dims(combined, axis=1)


class RotaryEmbedding(ModuleBase):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        rope_parameters = args.rope_parameters or {}
        rope_type = str(
            rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        )
        if rope_type != "default":
            raise ValueError(
                "Owned Qwen2.5-VL text runtime currently supports only the default rope type"
            )
        inv_freq = 1.0 / (
            args.rope_theta
            ** (mx.arange(0, args.head_dim, 2, dtype=mx.float32) / float(args.head_dim))
        )
        self._inv_freq = tuple(float(value.item()) for value in inv_freq)
        raw_sections = rope_parameters.get("mrope_section")
        self._mrope_sections = (
            tuple(int(value) for value in raw_sections)
            if isinstance(raw_sections, list)
            else None
        )

    def apply(
        self,
        queries: mx.array,
        keys: mx.array,
        *,
        offset: int = 0,
        position_ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if position_ids is None:
            cos, sin = self._cos_sin_default(
                sequence_length=int(queries.shape[2]),
                offset=offset,
                dtype=queries.dtype,
            )
        else:
            cos, sin = self._cos_sin_multimodal(position_ids, dtype=queries.dtype)
        return (
            (
                queries.astype(mx.float32) * cos
                + _rotate_half(queries.astype(mx.float32)) * sin
            ).astype(queries.dtype),
            (
                keys.astype(mx.float32) * cos
                + _rotate_half(keys.astype(mx.float32)) * sin
            ).astype(keys.dtype),
        )

    def _cos_sin_default(
        self,
        *,
        sequence_length: int,
        offset: int,
        dtype: mx.Dtype,
    ) -> tuple[mx.array, mx.array]:
        positions = mx.arange(
            offset,
            offset + sequence_length,
            dtype=mx.float32,
        )
        inv_freq = mx.array(self._inv_freq, dtype=mx.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.expand_dims(mx.cos(emb), axis=0), axis=0)
        sin = mx.expand_dims(mx.expand_dims(mx.sin(emb), axis=0), axis=0)
        return cos.astype(dtype), sin.astype(dtype)

    def _cos_sin_multimodal(
        self,
        position_ids: mx.array,
        *,
        dtype: mx.Dtype,
    ) -> tuple[mx.array, mx.array]:
        if position_ids.ndim != 3 or int(position_ids.shape[0]) != 3:
            raise ValueError(
                "Qwen2 multimodal position_ids must have shape (3, batch, sequence_length)"
            )
        if self._mrope_sections is None:
            raise ValueError(
                "Qwen2 multimodal position_ids require rope_parameters.mrope_section"
            )
        batch_size = int(position_ids.shape[1])
        inv_freq = mx.array(self._inv_freq, dtype=mx.float32)
        inv_freq_expanded = mx.broadcast_to(
            inv_freq.reshape(1, 1, inv_freq.shape[0], 1),
            (3, batch_size, inv_freq.shape[0], 1),
        )
        position_ids_expanded = position_ids.astype(mx.float32)[:, :, None, :]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 1, 3, 2)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return (
            _apply_multimodal_sections(cos, self._mrope_sections).astype(dtype),
            _apply_multimodal_sections(sin, self._mrope_sections).astype(dtype),
        )


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
            dim,
            self.num_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = build_linear(
            dim,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = build_linear(
            dim,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = build_linear(
            self.num_heads * self.head_dim,
            dim,
            bias=False,
        )
        self.rope = RotaryEmbedding(args)

    def __call__(
        self,
        hidden: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden.shape
        queries = self.q_proj(hidden)
        keys = self.k_proj(hidden)
        values = self.v_proj(hidden)
        queries = queries.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries, keys = self.rope.apply(queries, keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries, keys = self.rope.apply(
                queries,
                keys,
                position_ids=position_ids,
            )

        keys = _repeat_key_values(keys, self.repeats)
        values = _repeat_key_values(values, self.repeats)
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
        position_ids: mx.array | None = None,
    ) -> mx.array:
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden),
            mask,
            cache,
            position_ids=position_ids,
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen2TextModel(ModuleBase):
    def __init__(self, config: Qwen2TextConfig) -> None:
        super().__init__()
        self.config = config
        self.args = ModelArgs.from_text_config(config)
        self.embed_tokens = build_embedding(
            self.args.vocab_size,
            self.args.hidden_size,
        )
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
        position_ids: mx.array | None = None,
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
            hidden = layer(hidden, mask, layer_cache, position_ids=position_ids)
            if return_hidden_states:
                hidden_states.append(hidden)
        hidden = self.norm(hidden)
        if return_hidden_states:
            hidden_states.append(hidden)
            return hidden, hidden_states
        return hidden


def load_local_qwen2_text_model(model_path: Path) -> Qwen2TextModel:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Qwen2 config.json not found under '{model_path}'")
    raw = json.loads(config_path.read_text("utf-8"))
    if isinstance(raw.get("text_config"), dict):
        raw = raw["text_config"]
    config = Qwen2TextConfig.from_dict(raw)
    model = Qwen2TextModel(config)
    loaded_keys: set[str] = set()
    for weight_file in _weight_files(model_path):
        shard = mx.load(str(weight_file))
        if not isinstance(shard, dict):
            raise RuntimeError(
                f"Qwen2 weight shard '{weight_file}' did not load into a weight mapping"
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
                f"Qwen2 weight index '{index_path}' is missing a valid weight_map"
            )
        ordered_names = list(dict.fromkeys(str(value) for value in weight_map.values()))
        return tuple(model_path / name for name in ordered_names)
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        return (single_file,)
    files = tuple(sorted(model_path.glob("*.safetensors")))
    if files:
        return files
    raise ValueError(f"No Qwen2 safetensors weights found under '{model_path}'")


def _assert_loaded_parameter_coverage(
    model: Qwen2TextModel, *, loaded_keys: set[str]
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
    raise RuntimeError(f"Qwen2 weight coverage mismatch: {'; '.join(details)}")


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
        raise ValueError("Qwen2 attention_mask must be a rank-2 array")
    attention_mask_np = np.asarray(attention_mask)
    if attention_mask_np.ndim != 2:
        raise ValueError("Qwen2 attention_mask must be a rank-2 array")
    left_padding_values: list[int] = []
    right_padding_values: list[int] = []
    for row in attention_mask_np:
        valid_indices = np.flatnonzero(row)
        if valid_indices.size == 0:
            raise ValueError(
                "Qwen2 attention_mask must contain at least one valid token"
            )
        start = int(valid_indices[0])
        end = int(valid_indices[-1]) + 1
        if not np.all(row[start:end] == 1):
            raise ValueError(
                "Qwen2 attention_mask must contain one contiguous valid span"
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


__all__ = ["Qwen2TextConfig", "Qwen2TextModel", "load_local_qwen2_text_model"]
