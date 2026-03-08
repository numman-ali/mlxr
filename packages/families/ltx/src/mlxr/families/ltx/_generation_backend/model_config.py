from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .rope_ops import LTXRopeType


class LTXModelType(str, Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class AttentionType(str, Enum):
    DEFAULT = "default"


@dataclass(slots=True)
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


@dataclass(slots=True)
class LTXModelConfig:
    model_type: LTXModelType = LTXModelType.AudioVideo
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 48
    cross_attention_dim: int = 4096
    caption_channels: int = 3840
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_cross_attention_dim: int = 2048
    audio_caption_channels: int = 3840
    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: list[int] | None = None
    audio_positional_embedding_max_pos: list[int] | None = None
    use_middle_indices_grid: bool = True
    rope_type: str = LTXRopeType.INTERLEAVED.value
    double_precision_rope: bool = False
    timestep_scale_multiplier: int = 1000
    av_ca_timestep_scale_multiplier: int = 1000
    norm_eps: float = 1e-6
    attention_type: AttentionType = AttentionType.DEFAULT
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False
    caption_proj_before_connector: bool = False

    @classmethod
    def from_dict(cls, raw_config: Mapping[str, object]) -> "LTXModelConfig":
        return cls(
            model_type=_coerce_model_type(
                raw_config.get("model_type", LTXModelType.AudioVideo)
            ),
            num_attention_heads=_coerce_int(
                raw_config.get("num_attention_heads", 32), "num_attention_heads"
            ),
            attention_head_dim=_coerce_int(
                raw_config.get("attention_head_dim", 128), "attention_head_dim"
            ),
            in_channels=_coerce_int(raw_config.get("in_channels", 128), "in_channels"),
            out_channels=_coerce_int(
                raw_config.get("out_channels", 128), "out_channels"
            ),
            num_layers=_coerce_int(raw_config.get("num_layers", 48), "num_layers"),
            cross_attention_dim=_coerce_int(
                raw_config.get("cross_attention_dim", 4096), "cross_attention_dim"
            ),
            caption_channels=_coerce_int(
                raw_config.get("caption_channels", 3840), "caption_channels"
            ),
            audio_num_attention_heads=_coerce_int(
                raw_config.get("audio_num_attention_heads", 32),
                "audio_num_attention_heads",
            ),
            audio_attention_head_dim=_coerce_int(
                raw_config.get("audio_attention_head_dim", 64),
                "audio_attention_head_dim",
            ),
            audio_in_channels=_coerce_int(
                raw_config.get("audio_in_channels", 128), "audio_in_channels"
            ),
            audio_out_channels=_coerce_int(
                raw_config.get("audio_out_channels", 128), "audio_out_channels"
            ),
            audio_cross_attention_dim=_coerce_int(
                raw_config.get("audio_cross_attention_dim", 2048),
                "audio_cross_attention_dim",
            ),
            audio_caption_channels=_coerce_int(
                raw_config.get("audio_caption_channels", 3840),
                "audio_caption_channels",
            ),
            positional_embedding_theta=_coerce_float(
                raw_config.get("positional_embedding_theta", 10000.0),
                "positional_embedding_theta",
            ),
            positional_embedding_max_pos=_coerce_int_list_or_none(
                raw_config.get("positional_embedding_max_pos"),
                "positional_embedding_max_pos",
            ),
            audio_positional_embedding_max_pos=_coerce_int_list_or_none(
                raw_config.get("audio_positional_embedding_max_pos"),
                "audio_positional_embedding_max_pos",
            ),
            use_middle_indices_grid=_coerce_bool(
                raw_config.get("use_middle_indices_grid", True),
                "use_middle_indices_grid",
            ),
            rope_type=LTXRopeType.from_value(
                raw_config.get("rope_type", LTXRopeType.INTERLEAVED)
            ).value,
            double_precision_rope=_coerce_bool(
                raw_config.get("double_precision_rope", False),
                "double_precision_rope",
            ),
            timestep_scale_multiplier=_coerce_int(
                raw_config.get("timestep_scale_multiplier", 1000),
                "timestep_scale_multiplier",
            ),
            av_ca_timestep_scale_multiplier=_coerce_int(
                raw_config.get("av_ca_timestep_scale_multiplier", 1000),
                "av_ca_timestep_scale_multiplier",
            ),
            norm_eps=_coerce_float(raw_config.get("norm_eps", 1e-6), "norm_eps"),
            attention_type=_coerce_attention_type(
                raw_config.get("attention_type", AttentionType.DEFAULT)
            ),
            apply_gated_attention=_coerce_bool(
                raw_config.get("apply_gated_attention", False),
                "apply_gated_attention",
            ),
            cross_attention_adaln=_coerce_bool(
                raw_config.get("cross_attention_adaln", False),
                "cross_attention_adaln",
            ),
            caption_proj_before_connector=_coerce_bool(
                raw_config.get("caption_proj_before_connector", False),
                "caption_proj_before_connector",
            ),
        )

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def audio_inner_dim(self) -> int:
        return self.audio_num_attention_heads * self.audio_attention_head_dim

    def get_video_config(self) -> TransformerConfig | None:
        if not self.model_type.is_video_enabled():
            return None
        return TransformerConfig(
            dim=self.inner_dim,
            heads=self.num_attention_heads,
            d_head=self.attention_head_dim,
            context_dim=self.cross_attention_dim,
            apply_gated_attention=self.apply_gated_attention,
            cross_attention_adaln=self.cross_attention_adaln,
        )

    def get_audio_config(self) -> TransformerConfig | None:
        if not self.model_type.is_audio_enabled():
            return None
        return TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.audio_num_attention_heads,
            d_head=self.audio_attention_head_dim,
            context_dim=self.audio_cross_attention_dim,
            apply_gated_attention=self.apply_gated_attention,
            cross_attention_adaln=self.cross_attention_adaln,
        )

    def __post_init__(self) -> None:
        if self.positional_embedding_max_pos is None:
            self.positional_embedding_max_pos = [20, 2048, 2048]
        if self.audio_positional_embedding_max_pos is None:
            self.audio_positional_embedding_max_pos = [20]


def _coerce_model_type(value: object) -> LTXModelType:
    if isinstance(value, LTXModelType):
        return value
    if isinstance(value, str):
        return LTXModelType(value)
    raise TypeError(f"Unsupported LTX model type {value!r}")


def _coerce_attention_type(value: object) -> AttentionType:
    if isinstance(value, AttentionType):
        return value
    if isinstance(value, str):
        return AttentionType(value)
    raise TypeError(f"Unsupported attention type {value!r}")


def _coerce_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    return value


def _coerce_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    return float(value)


def _coerce_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


def _coerce_int_list_or_none(value: object, name: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list of ints, got {value!r}")
    coerced: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} entries must be ints, got {item!r}")
        coerced.append(item)
    return coerced
