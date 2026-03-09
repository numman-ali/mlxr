from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from safetensors import safe_open

_V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


class _SafeOpenHandle(Protocol):
    def __enter__(self) -> "_SafeOpenHandle": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None: ...
    def metadata(self) -> dict[str, str] | None: ...
    def keys(self) -> list[str]: ...


def _safe_open_numpy(checkpoint_path: Path) -> _SafeOpenHandle:
    return safe_open(str(checkpoint_path), framework="numpy")  # type: ignore[no-untyped-call]


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


def _string_value(value: object, *, context: str) -> str:
    if isinstance(value, str):
        return value
    raise RuntimeError(context)


def _required_bool(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    if key not in mapping:
        raise NotImplementedError(context)
    return _bool_value(mapping[key], context=context)


def _required_string(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    if key not in mapping:
        raise NotImplementedError(context)
    return _string_value(mapping[key], context=context)


@dataclass(frozen=True, slots=True)
class TransformerConfigResolution:
    config: dict[str, object]
    source: str | None


@dataclass(frozen=True, slots=True)
class TransformerSemanticContract:
    source: str | None
    rope_type: str
    double_precision_rope: bool
    caption_proj_before_connector: bool
    apply_gated_attention: bool
    cross_attention_adaln: bool
    timestep_scale_multiplier: int
    av_ca_timestep_scale_multiplier: int
    connector_apply_gated_attention: bool | None
    is_v2_prompt_layout: bool


def resolve_transformer_config_from_sources(
    sources: list[Path],
) -> TransformerConfigResolution:
    for candidate in sources:
        with _safe_open_numpy(candidate) as handle:
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
            return TransformerConfigResolution(
                config=transformer_config,
                source=str(candidate),
            )
    return TransformerConfigResolution(config={}, source=None)


def has_v2_projection_weights(sources: list[Path]) -> bool:
    for candidate in sources:
        with _safe_open_numpy(candidate) as handle:
            keys = set(handle.keys())
        if "text_embedding_projection.video_aggregate_embed.weight" in keys:
            return True
    return False


def resolve_transformer_semantic_contract(
    *,
    transformer_config: Mapping[str, object],
    source: str | None,
    checkpoint_path: Path | None = None,
    fallback_apply_gated_attention: bool | None = None,
    fallback_cross_attention_adaln: bool | None = None,
    has_v2_weights: bool = False,
    enforce_prompt_v2_layout: bool = True,
) -> TransformerSemanticContract:
    def _ctx(field: str) -> str:
        if checkpoint_path is None:
            return f"Current LTX prompt config is missing required field '{field}'"
        return f"LTX checkpoint '{checkpoint_path}' has invalid or missing transformer field '{field}'"

    overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
    if not overlapping_keys or not enforce_prompt_v2_layout:
        is_v2_layout = has_v2_weights
    else:
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
        is_v2_layout = True

    rope_type = _required_string(
        transformer_config, "rope_type", context=_ctx("rope_type")
    )
    frequencies_precision = _required_string(
        transformer_config,
        "frequencies_precision",
        context=_ctx("frequencies_precision"),
    )
    caption_proj_before_connector = _required_bool(
        transformer_config,
        "caption_proj_before_connector",
        context=_ctx("caption_proj_before_connector"),
    )
    timestep_scale_multiplier = _int_value(
        transformer_config.get("timestep_scale_multiplier", 1000),
        context=_ctx("timestep_scale_multiplier"),
    )
    av_ca_timestep_scale_multiplier = _int_value(
        transformer_config.get("av_ca_timestep_scale_multiplier", 1),
        context=_ctx("av_ca_timestep_scale_multiplier"),
    )

    raw_apply_gated_attention = transformer_config.get("apply_gated_attention")
    if raw_apply_gated_attention is None:
        if fallback_apply_gated_attention is None:
            raise NotImplementedError(_ctx("apply_gated_attention"))
        apply_gated_attention = fallback_apply_gated_attention
    else:
        apply_gated_attention = _bool_value(
            raw_apply_gated_attention,
            context=_ctx("apply_gated_attention"),
        )

    raw_cross_attention_adaln = transformer_config.get("cross_attention_adaln")
    if raw_cross_attention_adaln is None:
        if fallback_cross_attention_adaln is None:
            raise NotImplementedError(_ctx("cross_attention_adaln"))
        cross_attention_adaln = fallback_cross_attention_adaln
    else:
        cross_attention_adaln = _bool_value(
            raw_cross_attention_adaln,
            context=_ctx("cross_attention_adaln"),
        )

    raw_connector_apply_gated_attention = transformer_config.get(
        "connector_apply_gated_attention"
    )
    connector_apply_gated_attention = (
        None
        if raw_connector_apply_gated_attention is None
        else _bool_value(
            raw_connector_apply_gated_attention,
            context=_ctx("connector_apply_gated_attention"),
        )
    )

    return TransformerSemanticContract(
        source=source,
        rope_type=rope_type,
        double_precision_rope=frequencies_precision.lower() == "float64",
        caption_proj_before_connector=caption_proj_before_connector,
        apply_gated_attention=apply_gated_attention,
        cross_attention_adaln=cross_attention_adaln,
        timestep_scale_multiplier=timestep_scale_multiplier,
        av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
        connector_apply_gated_attention=connector_apply_gated_attention,
        is_v2_prompt_layout=is_v2_layout,
    )
