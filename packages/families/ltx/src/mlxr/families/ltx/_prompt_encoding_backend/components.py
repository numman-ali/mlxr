from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import runtime
from .compat import (
    Dropout,
    Gemma3Model,
    Linear,
    Module,
    RMSNorm,
    TextConfig,
    gelu_approx,
    mx,
    safe_open,
)
from .masks import (
    _apply_quantization,
    _connector_precomputed_freqs,
    _gemma_attention_masks,
    _rms_norm,
)

if runtime._RUNTIME_IMPORT_ERROR is None:

    class LanguageModel(Module):
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
                if not isinstance(shard, dict):
                    raise RuntimeError(
                        f"Gemma text encoder shard '{weight_file}' did not load into a weight mapping"
                    )
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

    class ConnectorAttention(Module):
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
            self.to_q = Linear(dim, inner_dim, bias=True)
            self.to_k = Linear(dim, inner_dim, bias=True)
            self.to_v = Linear(dim, inner_dim, bias=True)
            self.to_out = Linear(inner_dim, dim, bias=True)
            self.q_norm = RMSNorm(inner_dim, eps=1e-6)
            self.k_norm = RMSNorm(inner_dim, eps=1e-6)
            self.to_gate_logits = (
                Linear(dim, num_heads, bias=True) if apply_gated_attention else None
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

    class ConnectorFeedForward(Module):
        def __init__(self, dim: int = 3840, mult: int = 4, dropout: float = 0.0):
            super().__init__()
            inner_dim = dim * mult
            self.proj_in = Linear(dim, inner_dim, bias=True)
            self.dropout = Dropout(dropout)
            self.proj_out = Linear(inner_dim, dim, bias=True)

        def __call__(self, x: mx.array) -> mx.array:
            x = gelu_approx(self.proj_in(x))
            x = self.dropout(x)
            return self.proj_out(x)

    class ConnectorTransformerBlock(Module):
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

    class Embeddings1DConnector(Module):
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

    class GemmaFeatureExtractorV1(Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.aggregate_embed = Linear(input_dim, output_dim, bias=False)

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

    class GemmaFeatureExtractorV2(Module):
        def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            video_output_dim: int,
            audio_output_dim: int | None,
        ):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.video_aggregate_embed = Linear(input_dim, video_output_dim, bias=True)
            self.audio_aggregate_embed = (
                Linear(input_dim, audio_output_dim, bias=True)
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
