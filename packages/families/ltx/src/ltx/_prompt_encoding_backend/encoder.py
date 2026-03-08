from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..prompt_encoding import PromptEncoder, PromptEncodingResult
from . import runtime
from .compat import safe_open
from .components import (
    _V2_EXPECTED_CONFIG,
    Embeddings1DConnector,
    GemmaFeatureExtractorV1,
    GemmaFeatureExtractorV2,
    LanguageModel,
    _convert_to_additive_mask,
    _ProjectionWeights,
    _PromptLayout,
    _to_binary_mask,
    _TransformerConfigResolution,
)


class _TokenizerLike(Protocol):
    padding_side: str
    pad_token: str | None
    eos_token: str

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
        max_length: int,
        truncation: bool,
        padding: str,
    ) -> dict[str, object]: ...


def _require_weight_mapping(weights: object, *, context: str) -> dict[str, mx.array]:
    if not isinstance(weights, dict):
        raise RuntimeError(context)
    return {
        key: value
        for key, value in weights.items()
        if isinstance(key, str) and isinstance(value, mx.array)
    }


def _int_value(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(context)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise RuntimeError(context)
    if isinstance(value, str):
        return int(value)
    raise RuntimeError(context)


def _bool_value(value: object, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise RuntimeError(context)


def _float_value(value: object, *, context: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise RuntimeError(context)


def _int_list(value: object, *, context: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(context)
    return [_int_value(item, context=context) for item in value]


def _require_array_input(
    value: object, *, context: str
) -> int | float | bool | list[object] | tuple[object, ...] | np.ndarray | mx.array:
    if isinstance(value, (int, float, bool, list, tuple, np.ndarray, mx.array)):
        return value
    raise RuntimeError(context)


@dataclass(frozen=True, slots=True)
class _SinglePromptEncoding:
    video_context: object
    audio_context: object | None
    attention_mask: object
    token_count: int
    sequence_length: int
    video_context_shape: tuple[int, ...]
    attention_mask_shape: tuple[int, ...]
    audio_context_shape: tuple[int, ...] | None = None


if runtime._RUNTIME_IMPORT_ERROR is None:
    mx = runtime.mx
    AutoTokenizer = runtime.AutoTokenizer

    class _MLXLTXPromptEncoder:
        def __init__(self, checkpoint_path: Path, text_encoder_path: Path):
            self.checkpoint_path = checkpoint_path
            self.text_encoder_path = text_encoder_path
            self.max_length = 1024
            self.language_model: LanguageModel | None = None
            self.feature_extractor: (
                GemmaFeatureExtractorV1 | GemmaFeatureExtractorV2 | None
            ) = None
            self.video_connector: Embeddings1DConnector | None = None
            self.audio_connector: Embeddings1DConnector | None = None
            self.layout: _PromptLayout | None = None
            self._transformer_config_resolution: _TransformerConfigResolution | None = (
                None
            )
            self.tokenizer: _TokenizerLike | None = None

        def encode(
            self,
            prompt: str,
            *,
            max_length: int = 1024,
            return_audio_context: bool = True,
            negative_prompt: str | None = None,
        ) -> PromptEncodingResult:
            self.max_length = max_length
            self._ensure_loaded()
            if self.layout is None:
                raise RuntimeError("LTX prompt encoder layout failed to initialize")
            layout = self.layout
            prompt_encoding = self._encode_single_prompt(
                prompt=prompt,
                max_length=max_length,
                return_audio_context=return_audio_context,
            )
            normalized_negative_prompt = (
                negative_prompt.strip() if negative_prompt else ""
            )
            negative_encoding: _SinglePromptEncoding | None = None
            if normalized_negative_prompt:
                negative_encoding = self._encode_single_prompt(
                    prompt=normalized_negative_prompt,
                    max_length=max_length,
                    return_audio_context=return_audio_context,
                )
            return PromptEncodingResult(
                video_context=prompt_encoding.video_context,
                audio_context=prompt_encoding.audio_context,
                attention_mask=prompt_encoding.attention_mask,
                prompt_text=prompt,
                token_count=prompt_encoding.token_count,
                sequence_length=prompt_encoding.sequence_length,
                video_context_shape=prompt_encoding.video_context_shape,
                attention_mask_shape=prompt_encoding.attention_mask_shape,
                audio_context_shape=prompt_encoding.audio_context_shape,
                context_representation="post_connector",
                caption_proj_before_connector=layout.caption_proj_before_connector,
                rope_type=layout.rope_type,
                double_precision_rope=layout.double_precision_rope,
                connector_apply_gated_attention=(
                    layout.connector_apply_gated_attention
                ),
                transformer_context_dim=layout.transformer_context_dim,
                transformer_apply_gated_attention=(
                    layout.transformer_apply_gated_attention
                ),
                transformer_cross_attention_adaln=(
                    layout.transformer_cross_attention_adaln
                ),
                config_source=layout.config_source,
                negative_prompt_text=normalized_negative_prompt or None,
                negative_video_context=(
                    negative_encoding.video_context
                    if negative_encoding is not None
                    else None
                ),
                negative_audio_context=(
                    negative_encoding.audio_context
                    if negative_encoding is not None
                    else None
                ),
                negative_video_context_shape=(
                    negative_encoding.video_context_shape
                    if negative_encoding is not None
                    else None
                ),
                negative_audio_context_shape=(
                    negative_encoding.audio_context_shape
                    if negative_encoding is not None
                    else None
                ),
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

        def _encode_single_prompt(
            self,
            *,
            prompt: str,
            max_length: int,
            return_audio_context: bool,
        ) -> _SinglePromptEncoding:
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
            input_ids = mx.array(
                _require_array_input(
                    inputs["input_ids"],
                    context="Tokenizer did not return input_ids",
                )
            )
            attention_mask = mx.array(
                _require_array_input(
                    inputs["attention_mask"],
                    context="Tokenizer did not return attention_mask",
                )
            )
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
            return _SinglePromptEncoding(
                video_context=video_context,
                audio_context=audio_context,
                attention_mask=binary_mask,
                token_count=token_count,
                sequence_length=int(binary_mask.shape[-1]),
                video_context_shape=tuple(int(size) for size in video_context.shape),
                attention_mask_shape=tuple(int(size) for size in binary_mask.shape),
                audio_context_shape=audio_shape,
            )

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
                weights = _require_weight_mapping(
                    mx.load(str(candidate)),
                    context=(
                        f"LTX prompt connector shard '{candidate}' did not load into a weight mapping"
                    ),
                )
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
                video_heads = _int_value(
                    transformer_config.get("connector_num_attention_heads", 32),
                    context="Expected integer connector_num_attention_heads",
                )
                video_head_dim = _int_value(
                    transformer_config.get("connector_attention_head_dim", 128),
                    context="Expected integer connector_attention_head_dim",
                )
                video_dim = video_heads * video_head_dim
                audio_heads = _int_value(
                    transformer_config.get(
                        "audio_connector_num_attention_heads",
                        transformer_config.get("connector_num_attention_heads", 32),
                    ),
                    context="Expected integer audio_connector_num_attention_heads",
                )
                audio_head_dim = _int_value(
                    transformer_config.get(
                        "audio_connector_attention_head_dim",
                        transformer_config.get("connector_attention_head_dim", 128),
                    ),
                    context="Expected integer audio_connector_attention_head_dim",
                )
                audio_dim = audio_heads * audio_head_dim
                transformer_context_dim = _int_value(
                    transformer_config.get("cross_attention_dim", video_dim),
                    context="Expected integer cross_attention_dim",
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
                    video_layers=_int_value(
                        transformer_config.get("connector_num_layers", 8),
                        context="Expected integer connector_num_layers",
                    ),
                    audio_heads=audio_heads,
                    audio_head_dim=audio_head_dim,
                    audio_layers=_int_value(
                        transformer_config.get(
                            "audio_connector_num_layers",
                            transformer_config.get("connector_num_layers", 8),
                        ),
                        context="Expected integer audio_connector_num_layers",
                    ),
                    num_learnable_registers=_int_value(
                        transformer_config.get(
                            "connector_num_learnable_registers", 128
                        ),
                        context="Expected integer connector_num_learnable_registers",
                    ),
                    positional_embedding_theta=_float_value(
                        transformer_config.get("positional_embedding_theta", 10000.0),
                        context="Expected float positional_embedding_theta",
                    ),
                    positional_embedding_max_pos=_int_list(
                        transformer_config.get(
                            "connector_positional_embedding_max_pos", [4096]
                        ),
                        context="Expected integer connector_positional_embedding_max_pos entries",
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
            return _bool_value(
                transformer_config[key],
                context=f"Current V2 LTX prompt config field '{key}' must be boolean",
            )

        def _optional_transformer_bool(
            self, transformer_config: dict[str, object], key: str
        ) -> bool | None:
            if key not in transformer_config:
                return None
            return _bool_value(
                transformer_config[key],
                context=f"Current V2 LTX prompt config field '{key}' must be boolean",
            )

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
            "MLX prompt encoding dependencies are unavailable"
        ) from runtime._RUNTIME_IMPORT_ERROR
