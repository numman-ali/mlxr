# mypy: ignore-errors
from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .prompt_encoding import PromptEncoder, PromptEncodingResult

_RUNTIME_IMPORT_ERROR: Exception | None
try:
    import mlx.core as mx  # type: ignore[import-not-found, import-untyped]
    import mlx.nn as nn  # type: ignore[import-not-found, import-untyped]
    import numpy as np
    from mlx_lm.models.base import (  # type: ignore[import-not-found, import-untyped]
        create_causal_mask,
    )
    from mlx_vlm.models.gemma3.config import (  # type: ignore[import-not-found, import-untyped]
        TextConfig,
    )
    from mlx_vlm.models.gemma3.language import (  # type: ignore[import-not-found, import-untyped]
        Gemma3Model,
        create_attention_mask,
    )
    from safetensors import safe_open  # type: ignore[import-not-found, import-untyped]
    from transformers import (
        AutoTokenizer,  # type: ignore[import-not-found, import-untyped]
    )
except Exception as exc:  # pragma: no cover - exercised via error path tests
    _RUNTIME_IMPORT_ERROR = exc
else:
    _RUNTIME_IMPORT_ERROR = None


if _RUNTIME_IMPORT_ERROR is None:

    def _apply_quantization(
        model: nn.Module,
        weights: set[str],
        quantization: dict[str, object],
    ) -> None:
        def get_class_predicate(path: str, module: object) -> object:
            if path in quantization:
                return quantization[path]
            if not hasattr(module, "to_quantized"):
                return False
            if hasattr(module, "weight") and module.weight.shape[0] % 64 != 0:
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=int(quantization["group_size"]),
            bits=int(quantization["bits"]),
            mode=str(quantization.get("mode", "affine")),
            class_predicate=get_class_predicate,
        )

    def _rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
        return mx.fast.rms_norm(x, mx.ones((x.shape[-1],), dtype=x.dtype), eps)

    def _left_padding(attention_mask: mx.array) -> mx.array:
        sequence_length = int(attention_mask.shape[-1])
        valid_tokens = mx.sum(attention_mask.astype(mx.int32), axis=-1)
        return (
            mx.full(valid_tokens.shape, sequence_length, dtype=mx.int32) - valid_tokens
        )

    def _gemma_attention_masks(
        *,
        hidden: mx.array,
        attention_mask: mx.array | None,
        cache: list[object | None],
        config: TextConfig,
    ) -> tuple[object | None, object | None]:
        if attention_mask is None:
            global_mask = create_attention_mask(
                hidden, cache[config.sliding_window_pattern - 1]
            )
            if config.sliding_window_pattern > 1:
                sliding_window_mask = create_attention_mask(
                    hidden,
                    cache[0],
                    window_size=config.sliding_window,
                )
            else:
                sliding_window_mask = None
            return global_mask, sliding_window_mask

        if any(entry is not None for entry in cache):
            raise NotImplementedError(
                "Padded LTX MLX Gemma prompt encoding does not support KV cache"
            )

        left_padding = _left_padding(attention_mask)
        global_mask = create_causal_mask(
            int(hidden.shape[1]),
            left_padding=left_padding,
        )
        if config.sliding_window_pattern > 1:
            sliding_window_mask = create_causal_mask(
                int(hidden.shape[1]),
                left_padding=left_padding,
                window_size=config.sliding_window,
            )
        else:
            sliding_window_mask = None
        return global_mask, sliding_window_mask

    @functools.lru_cache(maxsize=16)
    def _connector_precomputed_freqs(
        seq_len: int,
        dim: int,
        num_heads: int,
        theta: float,
        max_pos: tuple[int, ...],
        rope_type: str,
        double_precision: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        np_dtype = np.float64 if double_precision else np.float32
        n_elem = 2 * len(max_pos)
        indices = np.power(
            theta,
            np.linspace(
                np.log(1.0) / np.log(theta),
                np.log(theta) / np.log(theta),
                dim // n_elem,
                dtype=np_dtype,
            ),
        )
        indices = indices * (np.pi / 2)

        fractional_positions = np.stack(
            [
                np.arange(seq_len, dtype=np_dtype) / np_dtype(position_max)
                for position_max in max_pos
            ],
            axis=-1,
        )
        freqs = ((fractional_positions[:, :, None] * 2) - 1) * indices[None, None, :]
        freqs = freqs.reshape(1, seq_len, -1)

        if rope_type == "split":
            expected_freqs = dim // 2
            pad_size = expected_freqs - freqs.shape[-1]
            cos_freq = np.cos(freqs)
            sin_freq = np.sin(freqs)
            if pad_size > 0:
                cos_padding = np.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = np.zeros_like(sin_freq[:, :, :pad_size])
                cos_freq = np.concatenate([cos_padding, cos_freq], axis=-1)
                sin_freq = np.concatenate([sin_padding, sin_freq], axis=-1)
            cos_freq = np.swapaxes(cos_freq.reshape(1, seq_len, num_heads, -1), 1, 2)
            sin_freq = np.swapaxes(sin_freq.reshape(1, seq_len, num_heads, -1), 1, 2)
        else:
            pad_size = dim % n_elem
            cos_freq = np.repeat(np.cos(freqs), 2, axis=-1)
            sin_freq = np.repeat(np.sin(freqs), 2, axis=-1)
            if pad_size > 0:
                cos_padding = np.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = np.zeros_like(sin_freq[:, :, :pad_size])
                cos_freq = np.concatenate([cos_padding, cos_freq], axis=-1)
                sin_freq = np.concatenate([sin_padding, sin_freq], axis=-1)
        return cos_freq, sin_freq

    class LanguageModel(nn.Module):
        def __init__(self, config: TextConfig):
            super().__init__()
            self.config = config
            self.model = Gemma3Model(config)

        def __call__(
            self,
            *,
            inputs: mx.array,
            input_embeddings: mx.array | None = None,
            attention_mask: mx.array | None = None,
            output_hidden_states: bool = False,
            cache: list[object | None] | None = None,
        ) -> tuple[mx.array, list[mx.array]]:
            hidden = (
                input_embeddings
                if input_embeddings is not None
                else self.model.embed_tokens(inputs)
            )
            hidden *= mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(
                hidden.dtype
            )

            all_hidden_states = [hidden] if output_hidden_states else []
            if cache is None:
                cache = [None] * len(self.model.layers)

            global_mask, sliding_window_mask = _gemma_attention_masks(
                hidden=hidden,
                attention_mask=attention_mask,
                cache=cache,
                config=self.config,
            )

            for index, layer in enumerate(self.model.layers):
                is_global = (
                    index % self.config.sliding_window_pattern
                    == self.config.sliding_window_pattern - 1
                )
                local_mask = global_mask if is_global else sliding_window_mask
                hidden = layer(hidden, local_mask, cache[index])
                if output_hidden_states and index < len(self.model.layers) - 1:
                    all_hidden_states.append(hidden)

            final_hidden = self.model.norm(hidden)
            if output_hidden_states:
                all_hidden_states.append(final_hidden)
                mx.eval(*all_hidden_states)
                return final_hidden, all_hidden_states
            mx.eval(final_hidden)
            return self.model.embed_tokens.as_linear(final_hidden), []

        @classmethod
        def from_pretrained(cls, model_path: Path) -> LanguageModel:
            config_file = model_path / "config.json"
            if not config_file.exists():
                raise ValueError(
                    f"Gemma text encoder config.json not found under '{model_path}'"
                )

            config_dict = json.loads(config_file.read_text(encoding="utf-8"))
            text_config = config_dict.get("text_config", config_dict)
            language_model = cls(config=TextConfig.from_dict(text_config))

            weight_files = _text_encoder_weight_files(model_path)
            quantization = config_dict.get("quantization")
            if isinstance(quantization, dict):
                scale_keys: set[str] = set()
                prefix = "language_model."
                for weight_file in weight_files:
                    with safe_open(str(weight_file), framework="numpy") as handle:
                        for key in handle.keys():
                            if key.startswith(prefix):
                                stripped = key[len(prefix) :]
                                if stripped.endswith(".scales"):
                                    scale_keys.add(stripped)
                _apply_quantization(language_model, scale_keys, quantization)

            prefix = "language_model."
            for weight_file in weight_files:
                shard = mx.load(str(weight_file))
                items: list[tuple[str, mx.array]] = []
                for key, value in shard.items():
                    if not key.startswith(prefix):
                        continue
                    stripped = key[len(prefix) :]
                    if hasattr(value, "dtype") and value.dtype == mx.float32:
                        value = value.astype(mx.bfloat16)
                    items.append((stripped, value))
                if items:
                    language_model.load_weights(items, strict=False)
                del shard
                mx.clear_cache()

            return language_model

    class ConnectorAttention(nn.Module):
        def __init__(
            self,
            dim: int = 3840,
            num_heads: int = 30,
            head_dim: int = 128,
            *,
            rope_type: str = "interleaved",
            apply_gated_attention: bool = False,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.rope_type = rope_type
            inner_dim = num_heads * head_dim
            self.scale = 1.0 / float(head_dim) ** 0.5
            self.to_q = nn.Linear(dim, inner_dim, bias=True)
            self.to_k = nn.Linear(dim, inner_dim, bias=True)
            self.to_v = nn.Linear(dim, inner_dim, bias=True)
            self.to_out = nn.Linear(inner_dim, dim, bias=True)
            self.q_norm = nn.RMSNorm(inner_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(inner_dim, eps=1e-6)
            self.to_gate_logits = (
                nn.Linear(dim, num_heads, bias=True) if apply_gated_attention else None
            )

        def __call__(
            self,
            x: mx.array,
            attention_mask: mx.array | None = None,
            pe: tuple[mx.array, mx.array] | None = None,
        ) -> mx.array:
            batch_size, seq_len, _ = x.shape
            query = self.q_norm(self.to_q(x))
            key = self.k_norm(self.to_k(x))
            value = self.to_v(x)

            query = mx.reshape(
                query, (batch_size, seq_len, self.num_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            key = mx.reshape(
                key, (batch_size, seq_len, self.num_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            value = mx.reshape(
                value, (batch_size, seq_len, self.num_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)

            if pe is not None:
                query = self._apply_rope(query, pe[0], pe[1])
                key = self._apply_rope(key, pe[0], pe[1])

            out = mx.fast.scaled_dot_product_attention(
                query, key, value, scale=self.scale, mask=attention_mask
            )
            out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            if self.to_gate_logits is not None:
                gates = 2.0 * mx.sigmoid(self.to_gate_logits(x))
                out = mx.reshape(
                    out, (batch_size, seq_len, self.num_heads, self.head_dim)
                )
                out = out * gates[..., None]
                out = mx.reshape(
                    out, (batch_size, seq_len, self.num_heads * self.head_dim)
                )
            return self.to_out(out)

        def _apply_rope(
            self,
            x: mx.array,
            cos_freq: mx.array,
            sin_freq: mx.array,
        ) -> mx.array:
            if self.rope_type == "split":
                return self._apply_split_rope(x, cos_freq, sin_freq)
            return self._apply_interleaved_rope(x, cos_freq, sin_freq)

        def _apply_interleaved_rope(
            self,
            x: mx.array,
            cos_freq: mx.array,
            sin_freq: mx.array,
        ) -> mx.array:
            input_dtype = x.dtype
            x = x.astype(mx.float32)
            cos_freq = cos_freq.astype(mx.float32)
            sin_freq = sin_freq.astype(mx.float32)
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            rotated = mx.stack([-x_odd, x_even], axis=-1)
            rotated = mx.reshape(rotated, x.shape)
            return (x * cos_freq + rotated * sin_freq).astype(input_dtype)

        def _apply_split_rope(
            self,
            x: mx.array,
            cos_freq: mx.array,
            sin_freq: mx.array,
        ) -> mx.array:
            input_dtype = x.dtype
            x = x.astype(mx.float32)
            cos_freq = cos_freq.astype(mx.float32)
            sin_freq = sin_freq.astype(mx.float32)
            half_dim = x.shape[-1] // 2
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            out1 = x1 * cos_freq - x2 * sin_freq
            out2 = x2 * cos_freq + x1 * sin_freq
            return mx.concatenate([out1, out2], axis=-1).astype(input_dtype)

    class ConnectorFeedForward(nn.Module):
        def __init__(self, dim: int = 3840, mult: int = 4, dropout: float = 0.0):
            super().__init__()
            inner_dim = dim * mult
            self.proj_in = nn.Linear(dim, inner_dim, bias=True)
            self.dropout = nn.Dropout(dropout)
            self.proj_out = nn.Linear(inner_dim, dim, bias=True)

        def __call__(self, x: mx.array) -> mx.array:
            x = nn.gelu_approx(self.proj_in(x))
            x = self.dropout(x)
            return self.proj_out(x)

    class ConnectorTransformerBlock(nn.Module):
        def __init__(
            self,
            dim: int = 3840,
            num_heads: int = 30,
            head_dim: int = 128,
            *,
            rope_type: str = "interleaved",
            apply_gated_attention: bool = False,
        ):
            super().__init__()
            self.attn1 = ConnectorAttention(
                dim,
                num_heads,
                head_dim,
                rope_type=rope_type,
                apply_gated_attention=apply_gated_attention,
            )
            self.ff = ConnectorFeedForward(dim)

        def __call__(
            self,
            x: mx.array,
            attention_mask: mx.array | None = None,
            pe: tuple[mx.array, mx.array] | None = None,
        ) -> mx.array:
            norm_x = _rms_norm(x)
            attn_out = self.attn1(norm_x, attention_mask, pe)
            x = x + attn_out
            ff_out = self.ff(_rms_norm(x))
            return x + ff_out

    class Embeddings1DConnector(nn.Module):
        def __init__(
            self,
            dim: int = 3840,
            num_heads: int = 30,
            head_dim: int = 128,
            num_layers: int = 2,
            num_learnable_registers: int = 128,
            positional_embedding_theta: float = 10000.0,
            positional_embedding_max_pos: list[int] | None = None,
            *,
            rope_type: str = "interleaved",
            double_precision_rope: bool = False,
            apply_gated_attention: bool = False,
        ):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.num_learnable_registers = num_learnable_registers
            self.positional_embedding_theta = positional_embedding_theta
            self.positional_embedding_max_pos = positional_embedding_max_pos or [4096]
            self.rope_type = rope_type
            self.double_precision_rope = double_precision_rope
            self.transformer_1d_blocks = {
                index: ConnectorTransformerBlock(
                    dim,
                    num_heads,
                    head_dim,
                    rope_type=rope_type,
                    apply_gated_attention=apply_gated_attention,
                )
                for index in range(num_layers)
            }
            if num_learnable_registers > 0:
                self.learnable_registers = mx.zeros((num_learnable_registers, dim))

        def __call__(
            self,
            hidden_states: mx.array,
            attention_mask: mx.array | None = None,
        ) -> tuple[mx.array, mx.array | None]:
            if self.num_learnable_registers > 0 and attention_mask is not None:
                hidden_states, attention_mask = self._replace_padded_with_registers(
                    hidden_states, attention_mask
                )

            frequencies = self._precompute_freqs_cis(
                int(hidden_states.shape[1]), hidden_states.dtype
            )
            for index in range(len(self.transformer_1d_blocks)):
                hidden_states = self.transformer_1d_blocks[index](
                    hidden_states, attention_mask, frequencies
                )
            return _rms_norm(hidden_states), attention_mask

        def _precompute_freqs_cis(
            self, seq_len: int, dtype: mx.Dtype
        ) -> tuple[mx.array, mx.array]:
            cos_freq, sin_freq = _connector_precomputed_freqs(
                seq_len,
                self.num_heads * self.head_dim,
                self.num_heads,
                self.positional_embedding_theta,
                tuple(int(position) for position in self.positional_embedding_max_pos),
                self.rope_type,
                self.double_precision_rope,
            )
            return (
                mx.array(cos_freq).astype(dtype),
                mx.array(sin_freq).astype(dtype),
            )

        def _replace_padded_with_registers(
            self,
            hidden_states: mx.array,
            attention_mask: mx.array,
        ) -> tuple[mx.array, mx.array]:
            batch_size, seq_len, dim = hidden_states.shape
            dtype = hidden_states.dtype
            mask_binary = (attention_mask.squeeze(1).squeeze(1) >= -9000.0).astype(
                mx.int32
            )
            num_tiles = math.ceil(seq_len / self.num_learnable_registers)
            registers = mx.tile(self.learnable_registers, (num_tiles, 1))[
                :seq_len
            ].astype(dtype)
            result_list: list[mx.array] = []
            for batch_index in range(batch_size):
                mask = mask_binary[batch_index]
                states = hidden_states[batch_index]
                num_valid = int(mx.sum(mask).item())
                valid_tokens = states[seq_len - num_valid :]
                pad_length = seq_len - num_valid
                if pad_length > 0:
                    padding = mx.zeros((pad_length, dim), dtype=dtype)
                    adjusted = mx.concatenate([valid_tokens, padding], axis=0)
                else:
                    adjusted = valid_tokens
                flipped_mask = mx.concatenate(
                    [
                        mx.ones((num_valid,), dtype=mx.int32),
                        mx.zeros((pad_length,), dtype=mx.int32),
                    ],
                    axis=0,
                )
                flipped_mask = flipped_mask[:, None].astype(dtype)
                combined = flipped_mask * adjusted + (1 - flipped_mask) * registers
                result_list.append(combined)
            hidden_states = mx.stack(result_list, axis=0)
            return hidden_states, mx.zeros_like(attention_mask)

    def _text_encoder_weight_files(model_path: Path) -> list[Path]:
        if (model_path / "diffusion_pytorch_model.safetensors.index.json").exists():
            files = sorted(model_path.glob("diffusion_pytorch_model-*.safetensors"))
        elif (model_path / "model.safetensors.index.json").exists():
            files = sorted(model_path.glob("model-*.safetensors"))
        elif (model_path / "diffusion_pytorch_model.safetensors").exists():
            files = [model_path / "diffusion_pytorch_model.safetensors"]
        elif (model_path / "model.safetensors").exists():
            files = [model_path / "model.safetensors"]
        else:
            files = sorted(model_path.glob("*.safetensors"))
        if not files:
            raise ValueError(
                f"Gemma text encoder weights were not found under '{model_path}'"
            )
        return files

    def _norm_and_concat_hidden_states(
        hidden_states: list[mx.array],
        attention_mask: mx.array,
        *,
        padding_side: str = "left",
    ) -> mx.array:
        stacked = mx.stack(hidden_states, axis=-1)
        dtype = stacked.dtype
        batch_size, seq_len, dim, num_layers = stacked.shape
        sequence_lengths = mx.sum(attention_mask, axis=-1)
        token_indices = mx.arange(seq_len)[None, :]
        if padding_side == "right":
            mask = token_indices < sequence_lengths[:, None]
        else:
            start_indices = seq_len - sequence_lengths[:, None]
            mask = token_indices >= start_indices
        mask = mask[:, :, None, None]
        eps = mx.array(1e-6, dtype=dtype)
        masked = mx.where(mask, stacked, mx.zeros_like(stacked))
        denom = (sequence_lengths * dim).reshape(batch_size, 1, 1, 1).astype(dtype)
        mean = mx.sum(masked, axis=(1, 2), keepdims=True) / (denom + eps)
        x_for_min = mx.where(
            mask, stacked, mx.full(stacked.shape, float("inf"), dtype=dtype)
        )
        x_for_max = mx.where(
            mask, stacked, mx.full(stacked.shape, float("-inf"), dtype=dtype)
        )
        x_min = mx.min(x_for_min, axis=(1, 2), keepdims=True)
        x_max = mx.max(x_for_max, axis=(1, 2), keepdims=True)
        range_value = x_max - x_min
        normalized = 8 * (stacked - mean) / (range_value + eps)
        normalized = mx.reshape(normalized, (batch_size, seq_len, -1))
        flat_mask = mx.broadcast_to(
            mask[:, :, :, 0], (batch_size, seq_len, dim * num_layers)
        )
        return mx.where(flat_mask, normalized, mx.zeros_like(normalized))

    def _norm_and_concat_per_token_rms(
        hidden_states: list[mx.array],
        attention_mask: mx.array,
    ) -> mx.array:
        stacked = mx.stack(hidden_states, axis=-1)
        variance = mx.mean(stacked * stacked, axis=2, keepdims=True)
        normalized = stacked * mx.rsqrt(variance + mx.array(1e-6, dtype=stacked.dtype))
        flattened = mx.reshape(
            normalized,
            (stacked.shape[0], stacked.shape[1], stacked.shape[2] * stacked.shape[3]),
        )
        mask = attention_mask.astype(mx.bool_)[:, :, None]
        return mx.where(mask, flattened, mx.zeros_like(flattened))

    def _rescale_norm(x: mx.array, target_dim: int, source_dim: int) -> mx.array:
        scale = math.sqrt(float(target_dim) / float(source_dim))
        return x * mx.array(scale, dtype=x.dtype)

    def _convert_to_additive_mask(
        attention_mask: mx.array, dtype: mx.Dtype
    ) -> mx.array:
        mask = (attention_mask.astype(mx.int32) - 1).astype(dtype)
        return (
            mask.reshape(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])
            * mx.finfo(dtype).max
        )

    def _to_binary_mask(
        encoded: mx.array, encoded_mask: mx.array | None
    ) -> tuple[mx.array, mx.array]:
        if encoded_mask is None:
            binary_mask = mx.ones((encoded.shape[0], encoded.shape[1]), dtype=mx.int32)
            return encoded, binary_mask
        binary_mask = (
            encoded_mask >= mx.array(-1e-6, dtype=encoded_mask.dtype)
        ).astype(mx.int32)
        binary_mask = binary_mask.reshape(encoded.shape[0], encoded.shape[1], 1)
        return encoded * binary_mask.astype(encoded.dtype), binary_mask[:, :, 0]

    class GemmaFeatureExtractorV1(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.aggregate_embed = nn.Linear(input_dim, output_dim, bias=False)

        def __call__(
            self,
            hidden_states: list[mx.array],
            attention_mask: mx.array,
            *,
            padding_side: str = "left",
        ) -> tuple[mx.array, mx.array]:
            normalized = _norm_and_concat_hidden_states(
                hidden_states,
                attention_mask,
                padding_side=padding_side,
            )
            features = self.aggregate_embed(normalized)
            return features, features

    class GemmaFeatureExtractorV2(nn.Module):
        def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            video_output_dim: int,
            audio_output_dim: int | None,
        ):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.video_aggregate_embed = nn.Linear(
                input_dim, video_output_dim, bias=True
            )
            self.audio_aggregate_embed = (
                nn.Linear(input_dim, audio_output_dim, bias=True)
                if audio_output_dim is not None
                else None
            )

        def __call__(
            self,
            hidden_states: list[mx.array],
            attention_mask: mx.array,
            *,
            padding_side: str = "left",
        ) -> tuple[mx.array, mx.array | None]:
            del padding_side
            normalized = _norm_and_concat_per_token_rms(hidden_states, attention_mask)
            normalized = normalized.astype(hidden_states[0].dtype)
            video_output_dim = int(self.video_aggregate_embed.weight.shape[0])
            video_features = self.video_aggregate_embed(
                _rescale_norm(normalized, video_output_dim, self.embedding_dim)
            )
            if self.audio_aggregate_embed is None:
                return video_features, None
            audio_output_dim = int(self.audio_aggregate_embed.weight.shape[0])
            audio_features = self.audio_aggregate_embed(
                _rescale_norm(normalized, audio_output_dim, self.embedding_dim)
            )
            return video_features, audio_features

    _V2_EXPECTED_CONFIG = {
        "caption_proj_before_connector": True,
        "caption_projection_first_linear": False,
        "caption_proj_input_norm": False,
        "caption_projection_second_linear": False,
    }

    @dataclass(frozen=True, slots=True)
    class _PromptLayout:
        version: str
        flat_dim: int
        embedding_dim: int
        video_dim: int
        audio_dim: int | None
        transformer_context_dim: int
        video_heads: int
        video_head_dim: int
        video_layers: int
        audio_heads: int
        audio_head_dim: int
        audio_layers: int
        num_learnable_registers: int
        positional_embedding_theta: float
        positional_embedding_max_pos: list[int]
        rope_type: str
        double_precision_rope: bool
        connector_apply_gated_attention: bool
        caption_proj_before_connector: bool
        transformer_apply_gated_attention: bool | None
        transformer_cross_attention_adaln: bool | None
        config_source: str | None

    @dataclass(frozen=True, slots=True)
    class _ProjectionWeights:
        version: str
        video_weight: mx.array
        video_bias: mx.array | None
        audio_weight: mx.array | None
        audio_bias: mx.array | None

    @dataclass(frozen=True, slots=True)
    class _TransformerConfigResolution:
        config: dict[str, object]
        source: str | None

    class _MLXLTXPromptEncoder:
        def __init__(self, checkpoint_path: Path, text_encoder_path: Path):
            self.checkpoint_path = checkpoint_path
            self.text_encoder_path = text_encoder_path
            self.max_length = 1024
            self.language_model: LanguageModel | None = None
            self.feature_extractor: object | None = None
            self.video_connector: Embeddings1DConnector | None = None
            self.audio_connector: Embeddings1DConnector | None = None
            self.layout: _PromptLayout | None = None
            self._transformer_config_resolution: _TransformerConfigResolution | None = (
                None
            )
            self.tokenizer: object | None = None

        def encode(
            self,
            prompt: str,
            *,
            max_length: int = 1024,
            return_audio_context: bool = True,
        ) -> PromptEncodingResult:
            self.max_length = max_length
            self._ensure_loaded()
            if self.tokenizer is None:
                raise RuntimeError("LTX prompt encoder tokenizer failed to initialize")
            if self.language_model is None or self.feature_extractor is None:
                raise RuntimeError("LTX prompt encoder model failed to initialize")
            if self.video_connector is None:
                raise RuntimeError(
                    "LTX prompt encoder video connector failed to initialize"
                )

            inputs = self.tokenizer(
                prompt,
                return_tensors="np",
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
            input_ids = mx.array(inputs["input_ids"])
            attention_mask = mx.array(inputs["attention_mask"])
            _, all_hidden_states = self.language_model(
                inputs=input_ids,
                input_embeddings=None,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            video_features, audio_features = self.feature_extractor(
                all_hidden_states,
                attention_mask,
                padding_side="left",
            )
            additive_mask = _convert_to_additive_mask(
                attention_mask, video_features.dtype
            )

            video_context, video_mask = self.video_connector(
                video_features, additive_mask
            )
            video_context, binary_mask = _to_binary_mask(video_context, video_mask)
            audio_context: object | None = None
            audio_shape: tuple[int, ...] | None = None
            if return_audio_context and self.audio_connector is not None:
                if audio_features is None:
                    raise ValueError(
                        "Current V2 LTX prompt path requires audio feature projections"
                    )
                audio_context, _ = self.audio_connector(
                    audio_features,
                    additive_mask,
                )
                audio_shape = tuple(int(size) for size in audio_context.shape)
                mx.eval(audio_context)

            mx.eval(video_context, binary_mask)
            token_count = int(mx.sum(attention_mask).item())
            return PromptEncodingResult(
                video_context=video_context,
                audio_context=audio_context,
                attention_mask=binary_mask,
                prompt_text=prompt,
                token_count=token_count,
                sequence_length=int(binary_mask.shape[-1]),
                video_context_shape=tuple(int(size) for size in video_context.shape),
                attention_mask_shape=tuple(int(size) for size in binary_mask.shape),
                audio_context_shape=audio_shape,
                context_representation="post_connector",
                caption_proj_before_connector=self.layout.caption_proj_before_connector,
                rope_type=self.layout.rope_type,
                double_precision_rope=self.layout.double_precision_rope,
                connector_apply_gated_attention=(
                    self.layout.connector_apply_gated_attention
                ),
                transformer_context_dim=self.layout.transformer_context_dim,
                transformer_apply_gated_attention=(
                    self.layout.transformer_apply_gated_attention
                ),
                transformer_cross_attention_adaln=(
                    self.layout.transformer_cross_attention_adaln
                ),
                config_source=self.layout.config_source,
            )

        def close(self) -> None:
            self.tokenizer = None
            self.audio_connector = None
            self.video_connector = None
            self.feature_extractor = None
            self.language_model = None
            self._transformer_config_resolution = None
            mx.clear_cache()

        def _ensure_loaded(self) -> None:
            if self.language_model is not None and self.tokenizer is not None:
                return
            self.language_model = LanguageModel.from_pretrained(self.text_encoder_path)
            self.layout = self._prompt_layout()
            if self.layout.version == "v2":
                self.feature_extractor = GemmaFeatureExtractorV2(
                    input_dim=self.layout.flat_dim,
                    embedding_dim=self.layout.embedding_dim,
                    video_output_dim=self.layout.video_dim,
                    audio_output_dim=self.layout.audio_dim,
                )
            else:
                self.feature_extractor = GemmaFeatureExtractorV1(
                    input_dim=self.layout.flat_dim,
                    output_dim=self.layout.video_dim,
                )
            self.video_connector = Embeddings1DConnector(
                dim=self.layout.video_dim,
                num_heads=self.layout.video_heads,
                head_dim=self.layout.video_head_dim,
                num_layers=self.layout.video_layers,
                num_learnable_registers=self.layout.num_learnable_registers,
                positional_embedding_theta=self.layout.positional_embedding_theta,
                positional_embedding_max_pos=self.layout.positional_embedding_max_pos,
                rope_type=self.layout.rope_type,
                double_precision_rope=self.layout.double_precision_rope,
                apply_gated_attention=self.layout.connector_apply_gated_attention,
            )
            self.audio_connector = (
                Embeddings1DConnector(
                    dim=self.layout.audio_dim,
                    num_heads=self.layout.audio_heads,
                    head_dim=self.layout.audio_head_dim,
                    num_layers=self.layout.audio_layers,
                    num_learnable_registers=self.layout.num_learnable_registers,
                    positional_embedding_theta=self.layout.positional_embedding_theta,
                    positional_embedding_max_pos=self.layout.positional_embedding_max_pos,
                    rope_type=self.layout.rope_type,
                    double_precision_rope=self.layout.double_precision_rope,
                    apply_gated_attention=self.layout.connector_apply_gated_attention,
                )
                if self.layout.audio_dim is not None
                else None
            )
            self._load_connector_weights()
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.text_encoder_path),
                local_files_only=True,
                model_max_length=self.max_length,
            )
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        def _load_connector_weights(self) -> None:
            if (
                self.feature_extractor is None
                or self.video_connector is None
                or self.layout is None
            ):
                raise RuntimeError("LTX prompt encoder connector modules are missing")

            projection_weights: _ProjectionWeights | None = None
            video_weights: dict[str, mx.array] = {}
            audio_weights: dict[str, mx.array] = {}

            for candidate in self._connector_sources():
                weights = mx.load(str(candidate))
                if projection_weights is None:
                    projection_weights = self._feature_projection(weights)
                if not video_weights:
                    video_weights = self._connector_weights(
                        weights,
                        primary_prefix="model.diffusion_model.video_embeddings_connector.",
                        secondary_prefix="connector.video_embeddings_connector.",
                        tertiary_prefix="video_connector.",
                    )
                if not audio_weights:
                    audio_weights = self._connector_weights(
                        weights,
                        primary_prefix="model.diffusion_model.audio_embeddings_connector.",
                        secondary_prefix="connector.audio_embeddings_connector.",
                        tertiary_prefix="audio_connector.",
                    )
                del weights
                mx.clear_cache()
                if projection_weights is not None and video_weights:
                    break

            if projection_weights is None:
                raise ValueError(
                    "LTX checkpoint is missing text embedding projection weights required for prompt encoding"
                )
            if not video_weights:
                raise ValueError(
                    "LTX checkpoint is missing video connector weights required for prompt encoding"
                )

            if projection_weights.version == "v2":
                if projection_weights.audio_weight is None:
                    raise ValueError(
                        "Current V2 LTX prompt path requires audio_aggregate_embed weights"
                    )
                feature_weights = [
                    ("video_aggregate_embed.weight", projection_weights.video_weight),
                ]
                if projection_weights.video_bias is not None:
                    feature_weights.append(
                        ("video_aggregate_embed.bias", projection_weights.video_bias)
                    )
                if projection_weights.audio_weight is not None:
                    feature_weights.append(
                        (
                            "audio_aggregate_embed.weight",
                            projection_weights.audio_weight,
                        )
                    )
                if projection_weights.audio_bias is not None:
                    feature_weights.append(
                        ("audio_aggregate_embed.bias", projection_weights.audio_bias)
                    )
                self.feature_extractor.load_weights(feature_weights, strict=False)
            else:
                self.feature_extractor.load_weights(
                    [("aggregate_embed.weight", projection_weights.video_weight)],
                    strict=False,
                )
            if (
                self.layout.num_learnable_registers > 0
                and "learnable_registers" not in video_weights
            ):
                raise ValueError(
                    "LTX checkpoint is missing video connector learnable registers"
                )
            mapped_video = [
                (self._normalize_connector_weight_key(key), value)
                for key, value in video_weights.items()
                if key != "learnable_registers"
            ]
            self.video_connector.load_weights(mapped_video, strict=False)
            if "learnable_registers" in video_weights:
                self.video_connector.learnable_registers = video_weights[
                    "learnable_registers"
                ]

            if self.audio_connector is not None and not audio_weights:
                raise ValueError(
                    "Current V2 LTX prompt path requires audio connector weights"
                )
            if (
                self.audio_connector is not None
                and self.layout.num_learnable_registers > 0
                and "learnable_registers" not in audio_weights
            ):
                raise ValueError(
                    "LTX checkpoint is missing audio connector learnable registers"
                )
            if self.audio_connector is not None and audio_weights:
                mapped_audio = [
                    (self._normalize_connector_weight_key(key), value)
                    for key, value in audio_weights.items()
                    if key != "learnable_registers"
                ]
                self.audio_connector.load_weights(mapped_audio, strict=False)
                if "learnable_registers" in audio_weights:
                    self.audio_connector.learnable_registers = audio_weights[
                        "learnable_registers"
                    ]
            else:
                self.audio_connector = None

        def _connector_sources(self) -> list[Path]:
            if self.checkpoint_path.is_file():
                root = self.checkpoint_path.parent
                sources: list[Path] = []
            else:
                root = self.checkpoint_path
                sources = []
            connector_candidates = [
                root / "connectors" / "ltx_text_connectors.safetensors",
                root / "connectors" / "diffusion_pytorch_model.safetensors",
            ]
            for candidate in connector_candidates:
                if candidate.exists():
                    sources.append(candidate)
            if self.checkpoint_path.is_file():
                sources.append(self.checkpoint_path)
            else:
                if (root / "model.safetensors").exists():
                    sources.append(root / "model.safetensors")
                sources.extend(sorted(root.glob("*.safetensors")))
            unique_sources: list[Path] = []
            seen: set[Path] = set()
            for source in sources:
                if source.exists() and source not in seen:
                    unique_sources.append(source)
                    seen.add(source)
            if not unique_sources:
                raise ValueError(
                    f"LTX checkpoint source '{self.checkpoint_path}' does not contain prompt-encoding weights"
                )
            return unique_sources

        def _prompt_layout(self) -> _PromptLayout:
            if self.language_model is None:
                raise RuntimeError(
                    "LTX prompt encoder language model is not initialized"
                )
            hidden_size = int(self.language_model.config.hidden_size)
            num_layers = int(self.language_model.config.num_hidden_layers) + 1
            flat_dim = hidden_size * num_layers
            config_resolution = self._resolve_transformer_config()
            transformer_config = config_resolution.config

            if self._is_v2_layout(transformer_config):
                rope_type = self._required_transformer_string(
                    transformer_config, "rope_type"
                )
                frequencies_precision = self._required_transformer_string(
                    transformer_config, "frequencies_precision"
                )
                caption_proj_before_connector = self._required_transformer_bool(
                    transformer_config, "caption_proj_before_connector"
                )
                connector_apply_gated_attention = self._required_transformer_bool(
                    transformer_config, "connector_apply_gated_attention"
                )
                video_heads = int(
                    transformer_config.get("connector_num_attention_heads", 32)
                )
                video_head_dim = int(
                    transformer_config.get("connector_attention_head_dim", 128)
                )
                video_dim = video_heads * video_head_dim
                audio_heads = int(
                    transformer_config.get(
                        "audio_connector_num_attention_heads",
                        transformer_config.get("connector_num_attention_heads", 32),
                    )
                )
                audio_head_dim = int(
                    transformer_config.get(
                        "audio_connector_attention_head_dim",
                        transformer_config.get("connector_attention_head_dim", 128),
                    )
                )
                audio_dim = audio_heads * audio_head_dim
                transformer_context_dim = int(
                    transformer_config.get("cross_attention_dim", video_dim)
                )
                if transformer_context_dim != video_dim:
                    raise NotImplementedError(
                        "V2 prompt layouts with transformer context dim "
                        f"{transformer_context_dim} and connector dim {video_dim} "
                        "are not supported yet"
                    )
                return _PromptLayout(
                    version="v2",
                    flat_dim=flat_dim,
                    embedding_dim=hidden_size,
                    video_dim=video_dim,
                    audio_dim=audio_dim,
                    transformer_context_dim=transformer_context_dim,
                    video_heads=video_heads,
                    video_head_dim=video_head_dim,
                    video_layers=int(transformer_config.get("connector_num_layers", 8)),
                    audio_heads=audio_heads,
                    audio_head_dim=audio_head_dim,
                    audio_layers=int(
                        transformer_config.get(
                            "audio_connector_num_layers",
                            transformer_config.get("connector_num_layers", 8),
                        )
                    ),
                    num_learnable_registers=int(
                        transformer_config.get("connector_num_learnable_registers", 128)
                    ),
                    positional_embedding_theta=float(
                        transformer_config.get("positional_embedding_theta", 10000.0)
                    ),
                    positional_embedding_max_pos=list(
                        transformer_config.get(
                            "connector_positional_embedding_max_pos", [4096]
                        )
                    ),
                    rope_type=rope_type,
                    double_precision_rope=frequencies_precision == "float64",
                    connector_apply_gated_attention=connector_apply_gated_attention,
                    caption_proj_before_connector=caption_proj_before_connector,
                    transformer_apply_gated_attention=(
                        self._optional_transformer_bool(
                            transformer_config, "apply_gated_attention"
                        )
                    ),
                    transformer_cross_attention_adaln=(
                        self._optional_transformer_bool(
                            transformer_config, "cross_attention_adaln"
                        )
                    ),
                    config_source=config_resolution.source,
                )

            return _PromptLayout(
                version="v1",
                flat_dim=flat_dim,
                embedding_dim=hidden_size,
                video_dim=hidden_size,
                audio_dim=hidden_size,
                transformer_context_dim=hidden_size,
                video_heads=30,
                video_head_dim=128,
                video_layers=2,
                audio_heads=30,
                audio_head_dim=128,
                audio_layers=2,
                num_learnable_registers=128,
                positional_embedding_theta=10000.0,
                positional_embedding_max_pos=[4096],
                rope_type="interleaved",
                double_precision_rope=False,
                connector_apply_gated_attention=False,
                caption_proj_before_connector=False,
                transformer_apply_gated_attention=None,
                transformer_cross_attention_adaln=None,
                config_source=config_resolution.source,
            )

        def _transformer_config(self) -> dict[str, object]:
            return self._resolve_transformer_config().config

        def _resolve_transformer_config(self) -> _TransformerConfigResolution:
            if self._transformer_config_resolution is not None:
                return self._transformer_config_resolution
            for candidate in self._connector_sources():
                with safe_open(str(candidate), framework="numpy") as handle:
                    metadata = handle.metadata() or {}
                config_raw = metadata.get("config")
                if not config_raw:
                    continue
                try:
                    config = json.loads(config_raw)
                except json.JSONDecodeError:
                    continue
                transformer_config = config.get("transformer")
                if isinstance(transformer_config, dict):
                    self._transformer_config_resolution = _TransformerConfigResolution(
                        config=transformer_config,
                        source=str(candidate),
                    )
                    return self._transformer_config_resolution
            self._transformer_config_resolution = _TransformerConfigResolution(
                config={},
                source=None,
            )
            return self._transformer_config_resolution

        def _required_transformer_bool(
            self, transformer_config: dict[str, object], key: str
        ) -> bool:
            if key not in transformer_config:
                raise NotImplementedError(
                    f"Current V2 LTX prompt config is missing required field '{key}'"
                )
            return bool(transformer_config[key])

        def _optional_transformer_bool(
            self, transformer_config: dict[str, object], key: str
        ) -> bool | None:
            if key not in transformer_config:
                return None
            return bool(transformer_config[key])

        def _required_transformer_string(
            self, transformer_config: dict[str, object], key: str
        ) -> str:
            value = transformer_config.get(key)
            if not isinstance(value, str):
                raise NotImplementedError(
                    f"Current V2 LTX prompt config is missing required field '{key}'"
                )
            return value

        def _is_v2_layout(self, transformer_config: dict[str, object]) -> bool:
            overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
            if not overlapping_keys:
                for candidate in self._connector_sources():
                    with safe_open(str(candidate), framework="numpy") as handle:
                        keys = set(handle.keys())
                    if "text_embedding_projection.video_aggregate_embed.weight" in keys:
                        return True
                return False
            missing_keys = _V2_EXPECTED_CONFIG.keys() - overlapping_keys
            if missing_keys:
                raise NotImplementedError(
                    "Partial V2 LTX prompt config is unsupported: missing "
                    + ", ".join(sorted(missing_keys))
                )
            unexpected = [
                key
                for key in _V2_EXPECTED_CONFIG
                if transformer_config.get(key) != _V2_EXPECTED_CONFIG[key]
            ]
            if unexpected:
                raise NotImplementedError(
                    "Unknown V2 LTX prompt config values: "
                    + ", ".join(
                        f"{key}={transformer_config.get(key)!r}" for key in unexpected
                    )
                )
            return True

        def _feature_projection(
            self, weights: dict[str, mx.array]
        ) -> _ProjectionWeights | None:
            if "text_embedding_projection.video_aggregate_embed.weight" in weights:
                return _ProjectionWeights(
                    version="v2",
                    video_weight=weights[
                        "text_embedding_projection.video_aggregate_embed.weight"
                    ],
                    video_bias=weights.get(
                        "text_embedding_projection.video_aggregate_embed.bias"
                    ),
                    audio_weight=weights.get(
                        "text_embedding_projection.audio_aggregate_embed.weight"
                    ),
                    audio_bias=weights.get(
                        "text_embedding_projection.audio_aggregate_embed.bias"
                    ),
                )
            for key in (
                "text_embedding_projection.aggregate_embed.weight",
                "text_proj_in.weight",
            ):
                if key in weights:
                    return _ProjectionWeights(
                        version="v1",
                        video_weight=weights[key],
                        video_bias=None,
                        audio_weight=None,
                        audio_bias=None,
                    )
            return None

        def _connector_weights(
            self,
            weights: dict[str, mx.array],
            *,
            primary_prefix: str,
            secondary_prefix: str | None = None,
            tertiary_prefix: str | None = None,
        ) -> dict[str, mx.array]:
            connector_weights: dict[str, mx.array] = {}
            for key, value in weights.items():
                if key.startswith(primary_prefix):
                    connector_weights[key.replace(primary_prefix, "")] = value
                elif secondary_prefix is not None and key.startswith(secondary_prefix):
                    connector_weights[key.replace(secondary_prefix, "")] = value
                elif tertiary_prefix is not None and key.startswith(tertiary_prefix):
                    connector_weights[key.replace(tertiary_prefix, "")] = value
            return connector_weights

        def _normalize_connector_weight_key(self, key: str) -> str:
            key = key.replace(".ff.net.0.proj.", ".ff.proj_in.")
            key = key.replace(".ff.net.2.", ".ff.proj_out.")
            return key.replace(".to_out.0.", ".to_out.")

    def create_prompt_encoder(
        checkpoint_path: Path,
        text_encoder_path: Path,
    ) -> PromptEncoder:
        return _MLXLTXPromptEncoder(
            checkpoint_path=checkpoint_path,
            text_encoder_path=text_encoder_path,
        )

else:

    def create_prompt_encoder(
        checkpoint_path: Path,
        text_encoder_path: Path,
    ) -> PromptEncoder:
        del checkpoint_path, text_encoder_path
        raise RuntimeError(
            "Local MLX LTX prompt encoder unavailable. Install the "
            "`mlx-runtime-family-ltx` prompt-encode dependencies "
            "(mlx, mlx-vlm, transformers, sentencepiece, safetensors)."
        ) from _RUNTIME_IMPORT_ERROR
