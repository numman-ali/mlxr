# mypy: ignore-errors
from __future__ import annotations

import json
from pathlib import Path

from .prompt_encoding import PromptEncoder, PromptEncodingResult

_RUNTIME_IMPORT_ERROR: Exception | None
try:
    import mlx.core as mx  # type: ignore[import-not-found, import-untyped]
    import mlx.nn as nn  # type: ignore[import-not-found, import-untyped]
    import numpy as np
    from mlx_vlm.models.gemma3.config import (  # type: ignore[import-not-found, import-untyped]
        TextConfig,
    )
    from mlx_vlm.models.gemma3.language import (  # type: ignore[import-not-found, import-untyped]
        Gemma3Model,
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
            batch_size, seq_len = inputs.shape
            hidden = (
                input_embeddings
                if input_embeddings is not None
                else self.model.embed_tokens(inputs)
            )
            hidden *= mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(
                hidden.dtype
            )
            mx.eval(hidden)

            all_hidden_states = [hidden] if output_hidden_states else []
            if cache is None:
                cache = [None] * len(self.model.layers)

            full_causal_mask = self._create_causal_mask_with_padding(
                seq_len, attention_mask, hidden.dtype
            )

            for index, layer in enumerate(self.model.layers):
                is_global = (
                    index % self.config.sliding_window_pattern
                    == self.config.sliding_window_pattern - 1
                )
                local_mask = full_causal_mask if is_global else full_causal_mask
                hidden = layer(hidden, local_mask, cache[index])
                mx.eval(hidden)
                if output_hidden_states and index < len(self.model.layers) - 1:
                    all_hidden_states.append(hidden)

            final_hidden = self.model.norm(hidden)
            mx.eval(final_hidden)
            if output_hidden_states:
                all_hidden_states.append(final_hidden)
                return final_hidden, all_hidden_states
            return self.model.embed_tokens.as_linear(final_hidden), []

        def _create_causal_mask_with_padding(
            self,
            seq_len: int,
            attention_mask: mx.array | None,
            dtype: mx.Dtype,
        ) -> mx.array:
            causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_), 0)
            min_value = (
                mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
            )
            if attention_mask is None:
                return mx.where(
                    causal_mask,
                    mx.zeros((seq_len, seq_len), dtype=dtype),
                    mx.full((seq_len, seq_len), min_value, dtype=dtype),
                )[None, None, :, :]

            padding_mask = attention_mask.astype(mx.bool_)
            combined = causal_mask[None, :, :] & padding_mask[:, None, :]
            return mx.where(
                combined,
                mx.zeros(combined.shape, dtype=dtype),
                mx.full(combined.shape, min_value, dtype=dtype),
            )[:, None, :, :]

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
        ):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            inner_dim = num_heads * head_dim
            self.scale = 1.0 / float(head_dim) ** 0.5
            self.to_q = nn.Linear(dim, inner_dim, bias=True)
            self.to_k = nn.Linear(dim, inner_dim, bias=True)
            self.to_v = nn.Linear(dim, inner_dim, bias=True)
            self.to_out = nn.Linear(inner_dim, dim, bias=True)
            self.q_norm = nn.RMSNorm(inner_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(inner_dim, eps=1e-6)

        def __call__(
            self,
            x: mx.array,
            attention_mask: mx.array | None = None,
            pe: tuple[mx.array, mx.array] | None = None,
        ) -> mx.array:
            del attention_mask
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
                query = self._apply_split_rope(query, pe[0], pe[1])
                key = self._apply_split_rope(key, pe[0], pe[1])

            out = mx.fast.scaled_dot_product_attention(
                query, key, value, scale=self.scale, mask=None
            )
            out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            return self.to_out(out)

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
            x = nn.gelu(self.proj_in(x))
            x = self.dropout(x)
            return self.proj_out(x)

    class ConnectorTransformerBlock(nn.Module):
        def __init__(self, dim: int = 3840, num_heads: int = 30, head_dim: int = 128):
            super().__init__()
            self.attn1 = ConnectorAttention(dim, num_heads, head_dim)
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
        ):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.num_learnable_registers = num_learnable_registers
            self.positional_embedding_theta = positional_embedding_theta
            self.positional_embedding_max_pos = positional_embedding_max_pos or [4096]
            self.transformer_1d_blocks = {
                index: ConnectorTransformerBlock(dim, num_heads, head_dim)
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
            dim = self.num_heads * self.head_dim
            theta = self.positional_embedding_theta
            max_pos = self.positional_embedding_max_pos
            n_elem = 2 * len(max_pos)
            num_indices = dim // n_elem
            log_start = np.log(1.0) / np.log(theta)
            log_end = np.log(theta) / np.log(theta)
            lin_space = np.linspace(log_start, log_end, num_indices, dtype=np.float64)
            indices = (np.power(theta, lin_space) * (np.pi / 2)).astype(np.float64)
            positions = np.arange(seq_len, dtype=np.float64)
            fractional_positions = positions / max_pos[0]
            scaled_positions = fractional_positions * 2 - 1
            freqs = scaled_positions[:, None] * indices[None, :]
            cos_freq = np.cos(freqs).reshape(
                seq_len, self.num_heads, self.head_dim // 2
            )
            sin_freq = np.sin(freqs).reshape(
                seq_len, self.num_heads, self.head_dim // 2
            )
            cos_freq = np.transpose(cos_freq, (1, 0, 2))[np.newaxis, ...]
            sin_freq = np.transpose(sin_freq, (1, 0, 2))[np.newaxis, ...]
            return (
                mx.array(cos_freq.astype(np.float32)).astype(dtype),
                mx.array(sin_freq.astype(np.float32)).astype(dtype),
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
            num_tiles = seq_len // self.num_learnable_registers
            registers = mx.tile(self.learnable_registers, (num_tiles, 1)).astype(dtype)
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

    class GemmaFeaturesExtractor(nn.Module):
        def __init__(self, input_dim: int = 188160, output_dim: int = 3840):
            super().__init__()
            self.aggregate_embed = nn.Linear(input_dim, output_dim, bias=False)

        def __call__(self, x: mx.array) -> mx.array:
            return self.aggregate_embed(x)

    class _MLXLTXPromptEncoder:
        def __init__(self, checkpoint_path: Path, text_encoder_path: Path):
            self.checkpoint_path = checkpoint_path
            self.text_encoder_path = text_encoder_path
            self.max_length = 1024
            self.language_model: LanguageModel | None = None
            self.feature_extractor: GemmaFeaturesExtractor | None = None
            self.video_connector: Embeddings1DConnector | None = None
            self.audio_connector: Embeddings1DConnector | None = None
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
            concat_hidden = _norm_and_concat_hidden_states(
                all_hidden_states, attention_mask, padding_side="left"
            )
            features = self.feature_extractor(concat_hidden)
            additive_mask = (attention_mask - 1).astype(features.dtype)
            additive_mask = (
                additive_mask.reshape(
                    attention_mask.shape[0], 1, 1, attention_mask.shape[-1]
                )
                * 1e9
            )

            video_context, _ = self.video_connector(features, additive_mask)
            audio_context: object | None = None
            audio_shape: tuple[int, ...] | None = None
            if return_audio_context and self.audio_connector is not None:
                audio_context, _ = self.audio_connector(features, additive_mask)
                audio_shape = tuple(int(size) for size in audio_context.shape)
                mx.eval(audio_context)

            mx.eval(video_context, attention_mask)
            token_count = int(mx.sum(attention_mask).item())
            return PromptEncodingResult(
                video_context=video_context,
                audio_context=audio_context,
                attention_mask=attention_mask,
                prompt_text=prompt,
                token_count=token_count,
                sequence_length=int(attention_mask.shape[-1]),
                video_context_shape=tuple(int(size) for size in video_context.shape),
                attention_mask_shape=tuple(int(size) for size in attention_mask.shape),
                audio_context_shape=audio_shape,
            )

        def close(self) -> None:
            self.tokenizer = None
            self.audio_connector = None
            self.video_connector = None
            self.feature_extractor = None
            self.language_model = None
            mx.clear_cache()

        def _ensure_loaded(self) -> None:
            if self.language_model is not None and self.tokenizer is not None:
                return
            self.language_model = LanguageModel.from_pretrained(self.text_encoder_path)
            self.feature_extractor = GemmaFeaturesExtractor(
                input_dim=3840 * 49, output_dim=3840
            )
            self.video_connector = Embeddings1DConnector(
                dim=3840,
                num_heads=30,
                head_dim=128,
                num_layers=2,
                num_learnable_registers=128,
                positional_embedding_max_pos=[4096],
            )
            self.audio_connector = Embeddings1DConnector(
                dim=3840,
                num_heads=30,
                head_dim=128,
                num_layers=2,
                num_learnable_registers=128,
                positional_embedding_max_pos=[4096],
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
            if self.feature_extractor is None or self.video_connector is None:
                raise RuntimeError("LTX prompt encoder connector modules are missing")

            feature_weight: mx.array | None = None
            video_weights: dict[str, mx.array] = {}
            audio_weights: dict[str, mx.array] = {}

            for candidate in self._connector_sources():
                weights = mx.load(str(candidate))
                if feature_weight is None:
                    feature_weight = self._feature_projection(weights)
                if not video_weights:
                    video_weights = self._connector_weights(
                        weights,
                        primary_prefix="model.diffusion_model.video_embeddings_connector.",
                        secondary_prefix="connector.video_embeddings_connector.",
                    )
                if not audio_weights:
                    audio_weights = self._connector_weights(
                        weights,
                        primary_prefix="model.diffusion_model.audio_embeddings_connector.",
                        secondary_prefix="connector.audio_embeddings_connector.",
                        tertiary_prefix="audio_connector.",
                    )
                if feature_weight is not None and video_weights:
                    break

            if feature_weight is None:
                raise ValueError(
                    "LTX checkpoint is missing text embedding projection weights required for prompt encoding"
                )
            if not video_weights:
                raise ValueError(
                    "LTX checkpoint is missing video connector weights required for prompt encoding"
                )

            self.feature_extractor.aggregate_embed.weight = feature_weight
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
                sources = [self.checkpoint_path]
            else:
                root = self.checkpoint_path
                sources = []
                if (root / "model.safetensors").exists():
                    sources.append(root / "model.safetensors")
                sources.extend(sorted(root.glob("*.safetensors")))
            connector_candidates = [
                root / "connectors" / "ltx_text_connectors.safetensors",
                root / "connectors" / "diffusion_pytorch_model.safetensors",
            ]
            for candidate in connector_candidates:
                if candidate.exists():
                    sources.append(candidate)
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

        def _feature_projection(self, weights: dict[str, mx.array]) -> mx.array | None:
            for key in (
                "text_embedding_projection.aggregate_embed.weight",
                "text_proj_in.weight",
            ):
                if key in weights:
                    return weights[key]
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
