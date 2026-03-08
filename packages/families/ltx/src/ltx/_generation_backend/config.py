from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Protocol

import mlx.core as mx
from safetensors import safe_open

from ..prompt_encoding import PromptEncodingResult
from .conditioning import _attention_mask, _context_width
from .reference_imports import _REFERENCE_MLX_VIDEO_ROOT
from .types import (
    _RuntimeAudioEncoderConfig,
    _RuntimeBWEConfig,
    _RuntimeModelConfig,
    _RuntimeVAEConfig,
    _RuntimeVocoderArchitectureConfig,
    _RuntimeVocoderConfig,
)

_DEFAULT_VAE_ENCODER_BLOCKS: tuple[tuple[str, object], ...] = (
    ("res_x", {"num_layers": 4}),
    ("compress_space_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_time_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_all_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
    ("compress_all_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
)

_SUPPORTED_VAE_LATENT_LOG_VAR = {"uniform", "per_channel", "constant", "none"}


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


def _int_list(value: object, *, context: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(context)
    return [_int_value(item, context=context) for item in value]


def _nested_int_tuples(
    value: object,
    *,
    context: str,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(context)
    nested: list[tuple[int, ...]] = []
    for item in value:
        if not isinstance(item, (list, tuple)):
            raise RuntimeError(context)
        nested.append(tuple(_int_value(entry, context=context) for entry in item))
    return tuple(nested)


def _int_or_default(value: object | None, default: int) -> int:
    if value is None:
        return default
    return _int_value(value, context="Expected an integer-compatible value")


def _float_or_default(value: object | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise RuntimeError("Expected a float-compatible value")


def _int_tuple(value: object | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("Expected a sequence of integer-compatible values")
    return tuple(
        _int_value(item, context="Expected a sequence of integer-compatible values")
        for item in value
    )


def _int_set(value: object | None) -> set[int]:
    if value is None:
        return set()
    if not isinstance(value, (list, tuple, set)):
        raise RuntimeError("Expected a collection of integer-compatible values")
    return {
        _int_value(item, context="Expected a collection of integer-compatible values")
        for item in value
    }


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, object]:
    with _safe_open_numpy(checkpoint_path) as checkpoint:
        metadata = checkpoint.metadata() or {}
    raw_config = metadata.get("config")
    if not isinstance(raw_config, str):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing JSON config metadata"
        )
    parsed = json.loads(raw_config)
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid config metadata"
        )
    return parsed


def _checkpoint_keys(checkpoint_path: Path) -> set[str]:
    with _safe_open_numpy(checkpoint_path) as checkpoint:
        return set(checkpoint.keys())


def _load_checkpoint_prefixed_weights(
    checkpoint_path: Path,
    *,
    prefixes: tuple[str, ...],
) -> dict[str, mx.array]:
    # Use the same MLX-native loader path as the main bridge. The audio branch in
    # the current 22B checkpoint includes dtypes that do not round-trip cleanly
    # through the numpy view returned by safetensors here.
    weights = mx.load(str(checkpoint_path))
    if not isinstance(weights, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' did not load into a weight mapping"
        )
    selected: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.startswith(prefixes):
            selected[key] = value
    if not selected:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing required prefixed weights for {prefixes!r}"
        )
    return selected


def _load_optional_json_config(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        return {}
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"LTX config '{config_path}' must decode to an object")
    return raw


def _first_present(
    mapping: Mapping[str, object], keys: tuple[str, ...]
) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _decoder_initial_feature_channels(
    *,
    base_channels: int,
    decoder_blocks: tuple[tuple[str, object], ...],
) -> int:
    feature_channels = base_channels
    for block_name, raw_params in decoder_blocks:
        if block_name == "res_x":
            continue
        params = (
            raw_params if isinstance(raw_params, dict) else {"num_layers": raw_params}
        )
        multiplier = _int_value(
            params.get("multiplier", 1),
            context=f"LTX decoder block '{block_name}' has invalid multiplier metadata",
        )
        if multiplier < 1:
            raise ValueError(
                f"LTX decoder block '{block_name}' has invalid multiplier {multiplier}"
            )
        feature_channels *= multiplier
    return feature_channels


def _runtime_vae_config(checkpoint_path: Path) -> _RuntimeVAEConfig:
    metadata = _checkpoint_metadata(checkpoint_path)
    raw_vae_config = metadata.get("vae")
    if not isinstance(raw_vae_config, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing VAE metadata"
        )

    def _parse_vae_blocks(
        raw_blocks: object | None,
        *,
        field_name: str,
        default: tuple[tuple[str, object], ...] | None = None,
    ) -> tuple[tuple[str, object], ...]:
        if raw_blocks is None:
            if default is None:
                raise RuntimeError(
                    f"LTX checkpoint '{checkpoint_path}' is missing {field_name} metadata"
                )
            return default
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise RuntimeError(
                f"LTX checkpoint '{checkpoint_path}' has invalid {field_name} metadata"
            )
        parsed_blocks: list[tuple[str, object]] = []
        for raw_block in raw_blocks:
            if (
                isinstance(raw_block, list)
                and len(raw_block) == 2
                and isinstance(raw_block[0], str)
            ):
                parsed_blocks.append((raw_block[0], raw_block[1]))
                continue
            raise RuntimeError(
                f"LTX checkpoint '{checkpoint_path}' has invalid {field_name} metadata: {raw_block!r}"
            )
        return tuple(parsed_blocks)

    decoder_blocks = _parse_vae_blocks(
        raw_vae_config.get("decoder_blocks"),
        field_name="decoder_blocks",
    )
    encoder_blocks = _parse_vae_blocks(
        raw_vae_config.get("encoder_blocks"),
        field_name="encoder_blocks",
        default=_DEFAULT_VAE_ENCODER_BLOCKS,
    )

    norm_layer = str(raw_vae_config.get("norm_layer", "pixel_norm"))
    if norm_layer != "pixel_norm":
        raise NotImplementedError(
            f"LTX real generation only supports pixel_norm VAE decoders, got {norm_layer!r}"
        )

    latent_log_var = str(raw_vae_config.get("latent_log_var", "uniform"))
    if latent_log_var not in _SUPPORTED_VAE_LATENT_LOG_VAR:
        raise NotImplementedError(
            f"LTX real generation only supports {sorted(_SUPPORTED_VAE_LATENT_LOG_VAR)!r} latent_log_var values, got {latent_log_var!r}"
        )

    return _RuntimeVAEConfig(
        in_channels=_int_value(
            raw_vae_config.get("in_channels", 3),
            context=f"LTX checkpoint '{checkpoint_path}' has invalid in_channels metadata",
        ),
        latent_channels=_int_value(
            raw_vae_config.get("latent_channels", 128),
            context=f"LTX checkpoint '{checkpoint_path}' has invalid latent_channels metadata",
        ),
        out_channels=_int_value(
            raw_vae_config.get("out_channels", 3),
            context=f"LTX checkpoint '{checkpoint_path}' has invalid out_channels metadata",
        ),
        patch_size=_int_value(
            raw_vae_config.get("patch_size", 4),
            context=f"LTX checkpoint '{checkpoint_path}' has invalid patch_size metadata",
        ),
        encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
        base_channels=_int_value(
            raw_vae_config.get("decoder_base_channels", 128),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid decoder_base_channels metadata"
            ),
        ),
        encoder_spatial_padding_mode=str(
            raw_vae_config.get("encoder_spatial_padding_mode", "zeros")
        ),
        spatial_padding_mode=str(
            raw_vae_config.get(
                "decoder_spatial_padding_mode",
                raw_vae_config.get("spatial_padding_mode", "reflect"),
            )
        ),
        latent_log_var=latent_log_var,
        timestep_conditioning=_bool_value(
            raw_vae_config.get("timestep_conditioning", True),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid timestep_conditioning metadata"
            ),
        ),
        norm_layer=norm_layer,
        causal_decoder=_bool_value(
            raw_vae_config.get("causal_decoder", False),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid causal_decoder metadata"
            ),
        ),
    )


def _runtime_vocoder_architecture_config(
    *,
    raw_config: dict[str, object],
    checkpoint_path: Path,
    context_label: str,
    output_sample_rate: int,
) -> _RuntimeVocoderArchitectureConfig:
    activation = str(raw_config.get("activation", "snake"))
    if activation not in {"snake", "snakebeta"}:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has unsupported {context_label} activation {activation!r}"
        )
    return _RuntimeVocoderArchitectureConfig(
        resblock_kernel_sizes=tuple(
            _int_list(
                raw_config.get("resblock_kernel_sizes", [3, 7, 11]),
                context=(
                    f"LTX checkpoint '{checkpoint_path}' has invalid {context_label} resblock_kernel_sizes metadata"
                ),
            )
        ),
        upsample_rates=tuple(
            _int_list(
                raw_config.get("upsample_rates", [6, 5, 2, 2, 2]),
                context=(
                    f"LTX checkpoint '{checkpoint_path}' has invalid {context_label} upsample_rates metadata"
                ),
            )
        ),
        upsample_kernel_sizes=tuple(
            _int_list(
                raw_config.get("upsample_kernel_sizes", [16, 15, 8, 4, 4]),
                context=(
                    f"LTX checkpoint '{checkpoint_path}' has invalid {context_label} upsample_kernel_sizes metadata"
                ),
            )
        ),
        resblock_dilation_sizes=_nested_int_tuples(
            raw_config.get(
                "resblock_dilation_sizes",
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            ),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid {context_label} resblock_dilation_sizes metadata"
            ),
        ),
        upsample_initial_channel=_int_value(
            raw_config.get("upsample_initial_channel", 1024),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid {context_label} upsample_initial_channel metadata"
            ),
        ),
        stereo=_bool_value(
            raw_config.get("stereo", True),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid {context_label} stereo metadata"
            ),
        ),
        resblock=str(raw_config.get("resblock", "1")),
        activation=activation,
        use_tanh_at_final=_bool_value(
            raw_config.get("use_tanh_at_final", True),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid {context_label} use_tanh_at_final metadata"
            ),
        ),
        apply_final_activation=_bool_value(
            raw_config.get("apply_final_activation", True),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid {context_label} apply_final_activation metadata"
            ),
        ),
        use_bias_at_final=_bool_value(
            raw_config.get("use_bias_at_final", True),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid {context_label} use_bias_at_final metadata"
            ),
        ),
        output_sample_rate=output_sample_rate,
    )


def _runtime_vocoder_config(checkpoint_path: Path) -> _RuntimeVocoderConfig:
    metadata = _checkpoint_metadata(checkpoint_path)
    raw_vocoder_config = metadata.get("vocoder")
    if not isinstance(raw_vocoder_config, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing vocoder metadata"
        )

    has_nested_bwe = isinstance(raw_vocoder_config.get("bwe"), dict)
    has_nested_vocoder = isinstance(raw_vocoder_config.get("vocoder"), dict)
    if has_nested_bwe != has_nested_vocoder:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has partial nested BWE vocoder metadata"
        )
    uses_bwe = has_nested_bwe and has_nested_vocoder
    base_config = raw_vocoder_config["vocoder"] if uses_bwe else raw_vocoder_config
    if not isinstance(base_config, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid base vocoder metadata"
        )
    bwe_config = raw_vocoder_config.get("bwe") if uses_bwe else None
    bwe_output_sample_rate = (
        _int_value(
            bwe_config.get("output_sampling_rate"),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid BWE output sample rate metadata"
            ),
        )
        if isinstance(bwe_config, dict)
        and bwe_config.get("output_sampling_rate") is not None
        else None
    )
    output_sample_rate = (
        _int_value(
            bwe_config.get("input_sampling_rate"),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid BWE input sample rate metadata"
            ),
        )
        if isinstance(bwe_config, dict)
        and bwe_config.get("input_sampling_rate") is not None
        else _int_value(
            base_config.get("output_sampling_rate", 24000),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid vocoder output sample rate metadata"
            ),
        )
    )

    vocoder_config = _runtime_vocoder_architecture_config(
        raw_config=base_config,
        checkpoint_path=checkpoint_path,
        context_label="vocoder",
        output_sample_rate=output_sample_rate,
    )

    bwe_runtime_config: _RuntimeBWEConfig | None = None
    if uses_bwe:
        if not isinstance(bwe_config, dict):
            raise RuntimeError(
                f"LTX checkpoint '{checkpoint_path}' has invalid BWE metadata"
            )
        bwe_input_sample_rate = _int_value(
            bwe_config.get("input_sampling_rate"),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid BWE input sample rate metadata"
            ),
        )
        bwe_output_sample_rate = _int_value(
            bwe_config.get("output_sampling_rate"),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid BWE output sample rate metadata"
            ),
        )
        bwe_runtime_config = _RuntimeBWEConfig(
            generator=_runtime_vocoder_architecture_config(
                raw_config=bwe_config,
                checkpoint_path=checkpoint_path,
                context_label="BWE generator",
                output_sample_rate=bwe_output_sample_rate,
            ),
            input_sample_rate=bwe_input_sample_rate,
            output_sample_rate=bwe_output_sample_rate,
            hop_length=_int_value(
                bwe_config.get("hop_length"),
                context=(
                    f"LTX checkpoint '{checkpoint_path}' has invalid BWE hop_length metadata"
                ),
            ),
            n_fft=_int_value(
                bwe_config.get("n_fft"),
                context=(
                    f"LTX checkpoint '{checkpoint_path}' has invalid BWE n_fft metadata"
                ),
            ),
            win_size=_int_value(
                bwe_config.get("win_size"),
                context=(
                    f"LTX checkpoint '{checkpoint_path}' has invalid BWE win_size metadata"
                ),
            ),
            num_mels=_int_value(
                bwe_config.get("num_mels"),
                context=(
                    f"LTX checkpoint '{checkpoint_path}' has invalid BWE num_mels metadata"
                ),
            ),
        )

    return _RuntimeVocoderConfig(
        vocoder=vocoder_config,
        bwe=bwe_runtime_config,
    )


def _runtime_audio_encoder_config(checkpoint_root: Path) -> _RuntimeAudioEncoderConfig:
    raw_config = _load_optional_json_config(
        checkpoint_root / "audio_vae" / "config.json"
    )

    raw_ch_mult = raw_config.get("ch_mult", (1, 2, 4))
    if not isinstance(raw_ch_mult, (list, tuple)):
        raise RuntimeError(
            f"LTX audio VAE config '{checkpoint_root / 'audio_vae' / 'config.json'}' has invalid ch_mult metadata"
        )
    ch_mult: tuple[int, ...] = tuple(
        _int_value(value, context="Expected integer ch_mult entries")
        for value in raw_ch_mult
    )

    raw_attn_resolutions = raw_config.get("attn_resolutions") or []
    if not isinstance(raw_attn_resolutions, (list, tuple, set)):
        raise RuntimeError(
            f"LTX audio VAE config '{checkpoint_root / 'audio_vae' / 'config.json'}' has invalid attn_resolutions metadata"
        )
    attn_resolutions = {
        _int_value(value, context="Expected integer attention resolutions")
        for value in raw_attn_resolutions
    }

    return _RuntimeAudioEncoderConfig(
        base_channels=_int_or_default(raw_config.get("base_channels"), 128),
        ch_mult=ch_mult,
        num_res_blocks=_int_or_default(raw_config.get("num_res_blocks"), 2),
        attn_resolutions=attn_resolutions,
        resolution=_int_or_default(raw_config.get("resolution"), 256),
        latent_channels=_int_or_default(raw_config.get("latent_channels"), 8),
        dropout=_float_or_default(raw_config.get("dropout"), 0.0),
        in_channels=_int_or_default(raw_config.get("in_channels"), 2),
        norm_type=str(raw_config.get("norm_type", "pixel")),
        causality_axis=str(raw_config.get("causality_axis", "height")),
        mid_block_add_attention=_bool_value(
            raw_config.get("mid_block_add_attention", True),
            context="Expected boolean mid_block_add_attention value",
        ),
        sample_rate=_int_or_default(raw_config.get("sample_rate"), 16000),
        mel_hop_length=_int_or_default(raw_config.get("mel_hop_length"), 160),
        mel_bins=_int_or_default(raw_config.get("mel_bins"), 64),
        n_fft=_int_or_default(raw_config.get("n_fft"), 1024),
        is_causal=_bool_value(
            raw_config.get("is_causal", True),
            context="Expected boolean is_causal value",
        ),
        double_z=_bool_value(
            raw_config.get("double_z", True),
            context="Expected boolean double_z value",
        ),
    )


def _validate_upsampler_layout(weights_path: Path) -> None:
    raw_weights = mx.load(str(weights_path))
    if not isinstance(raw_weights, dict):
        raise RuntimeError(
            f"LTX spatial upsampler '{weights_path}' did not load into a weight mapping"
        )
    if not any(
        key.startswith("upsampler.conv.") or key.startswith("upsampler.0.")
        for key in raw_weights
    ):
        raise RuntimeError(
            f"LTX spatial upsampler '{weights_path}' is missing a supported x2 conv layout"
        )


def can_use_reference_backend(
    checkpoint_path: Path,
    spatial_upsampler_path: Path | None = None,
) -> bool:
    try:
        _validate_reference_backend_compatibility(checkpoint_path)
        if spatial_upsampler_path is not None:
            _validate_upsampler_layout(spatial_upsampler_path)
        if not _REFERENCE_MLX_VIDEO_ROOT.is_dir():
            return False
    except Exception:
        return False
    return True


def _required_transformer_string(
    raw_transformer_config: dict[str, object],
    checkpoint_path: Path,
    key: str,
) -> str:
    value = raw_transformer_config.get(key)
    if not isinstance(value, str):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing required transformer field '{key}'"
        )
    return value


def _required_transformer_bool(
    raw_transformer_config: dict[str, object],
    checkpoint_path: Path,
    key: str,
) -> bool:
    if key not in raw_transformer_config:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing required transformer field '{key}'"
        )
    return _bool_value(
        raw_transformer_config[key],
        context=(
            f"LTX checkpoint '{checkpoint_path}' has invalid transformer boolean field '{key}'"
        ),
    )


def _resolved_transformer_flag(
    *,
    raw_transformer_config: dict[str, object],
    checkpoint_path: Path,
    key: str,
    inferred: bool,
) -> bool:
    explicit = raw_transformer_config.get(key)
    if explicit is None:
        return inferred
    explicit_bool = _bool_value(
        explicit,
        context=(
            f"LTX checkpoint '{checkpoint_path}' has invalid transformer boolean field '{key}'"
        ),
    )
    if inferred and not explicit_bool:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has transformer metadata {key}=False "
            "but matching weights indicate the feature is required"
        )
    return explicit_bool


def _runtime_model_config(checkpoint_path: Path) -> _RuntimeModelConfig:
    metadata = _checkpoint_metadata(checkpoint_path)
    raw_transformer_config = metadata.get("transformer")
    if not isinstance(raw_transformer_config, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing transformer metadata"
        )
    checkpoint_keys = _checkpoint_keys(checkpoint_path)
    default_audio_channels = 8 * 16

    audio_enabled = any(
        key.startswith("model.diffusion_model.audio_")
        or ".audio_" in key
        or "audio_to_video_attn" in key
        or "video_to_audio_attn" in key
        for key in checkpoint_keys
    )
    apply_gated_attention = _resolved_transformer_flag(
        raw_transformer_config=raw_transformer_config,
        checkpoint_path=checkpoint_path,
        key="apply_gated_attention",
        inferred=any(".to_gate_logits." in key for key in checkpoint_keys),
    )
    cross_attention_adaln = _resolved_transformer_flag(
        raw_transformer_config=raw_transformer_config,
        checkpoint_path=checkpoint_path,
        key="cross_attention_adaln",
        inferred=any(
            "prompt_scale_shift_table" in key or "prompt_adaln_single." in key
            for key in checkpoint_keys
        ),
    )
    return _RuntimeModelConfig(
        num_attention_heads=_int_value(
            raw_transformer_config.get("num_attention_heads", 32),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid num_attention_heads metadata"
            ),
        ),
        attention_head_dim=_int_value(
            raw_transformer_config.get("attention_head_dim", 128),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid attention_head_dim metadata"
            ),
        ),
        in_channels=_int_value(
            raw_transformer_config.get("in_channels", 128),
            context=f"LTX checkpoint '{checkpoint_path}' has invalid in_channels metadata",
        ),
        out_channels=_int_value(
            raw_transformer_config.get("out_channels", 128),
            context=f"LTX checkpoint '{checkpoint_path}' has invalid out_channels metadata",
        ),
        num_layers=_int_value(
            raw_transformer_config.get("num_layers", 48),
            context=f"LTX checkpoint '{checkpoint_path}' has invalid num_layers metadata",
        ),
        cross_attention_dim=_int_value(
            raw_transformer_config.get("cross_attention_dim", 4096),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid cross_attention_dim metadata"
            ),
        ),
        audio_enabled=audio_enabled,
        audio_num_attention_heads=_int_or_default(
            raw_transformer_config.get("audio_num_attention_heads"),
            32,
        ),
        audio_attention_head_dim=_int_or_default(
            raw_transformer_config.get("audio_attention_head_dim"),
            64,
        ),
        audio_in_channels=_int_or_default(
            raw_transformer_config.get("audio_in_channels"),
            default_audio_channels,
        ),
        audio_out_channels=_int_or_default(
            raw_transformer_config.get("audio_out_channels"),
            default_audio_channels,
        ),
        audio_cross_attention_dim=_int_or_default(
            raw_transformer_config.get("audio_cross_attention_dim"),
            2048,
        ),
        positional_embedding_theta=_float_or_default(
            raw_transformer_config.get("positional_embedding_theta"),
            10000.0,
        ),
        positional_embedding_max_pos=_int_list(
            raw_transformer_config.get(
                "positional_embedding_max_pos", [20, 2048, 2048]
            ),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid positional_embedding_max_pos metadata"
            ),
        ),
        audio_positional_embedding_max_pos=_int_list(
            raw_transformer_config.get("audio_positional_embedding_max_pos", [20]),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid audio_positional_embedding_max_pos metadata"
            ),
        ),
        use_middle_indices_grid=_bool_value(
            raw_transformer_config.get("use_middle_indices_grid", True),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid use_middle_indices_grid metadata"
            ),
        ),
        rope_type=_required_transformer_string(
            raw_transformer_config, checkpoint_path, "rope_type"
        ),
        double_precision_rope=(
            _required_transformer_string(
                raw_transformer_config, checkpoint_path, "frequencies_precision"
            ).lower()
            == "float64"
        ),
        timestep_scale_multiplier=_int_value(
            raw_transformer_config.get("timestep_scale_multiplier", 1000),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid timestep_scale_multiplier metadata"
            ),
        ),
        av_ca_timestep_scale_multiplier=_int_value(
            raw_transformer_config.get(
                "av_ca_timestep_scale_multiplier",
                raw_transformer_config.get("timestep_scale_multiplier", 1000),
            ),
            context=(
                f"LTX checkpoint '{checkpoint_path}' has invalid av_ca_timestep_scale_multiplier metadata"
            ),
        ),
        norm_eps=_float_or_default(raw_transformer_config.get("norm_eps"), 1e-6),
        apply_gated_attention=apply_gated_attention,
        cross_attention_adaln=cross_attention_adaln,
        caption_proj_before_connector=_required_transformer_bool(
            raw_transformer_config, checkpoint_path, "caption_proj_before_connector"
        ),
    )


def _validate_reference_backend_compatibility(
    checkpoint_path: Path,
) -> None:
    runtime_config = _runtime_model_config(checkpoint_path)
    _runtime_vae_config(checkpoint_path)
    checkpoint_keys = _checkpoint_keys(checkpoint_path)
    required_stats = (
        "vae.per_channel_statistics.mean-of-means",
        "vae.per_channel_statistics.std-of-means",
    )
    if not all(key in checkpoint_keys for key in required_stats):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing required VAE per-channel statistics"
        )
    if runtime_config.audio_enabled:
        required_audio_weights = (
            "model.diffusion_model.audio_patchify_proj.weight",
            "model.diffusion_model.audio_proj_out.weight",
            "model.diffusion_model.transformer_blocks.0.audio_attn1.to_q.weight",
            "model.diffusion_model.transformer_blocks.0.audio_attn2.to_q.weight",
            "model.diffusion_model.transformer_blocks.0.audio_to_video_attn.to_q.weight",
            "model.diffusion_model.transformer_blocks.0.video_to_audio_attn.to_q.weight",
        )
        if not all(key in checkpoint_keys for key in required_audio_weights):
            raise RuntimeError(
                f"LTX checkpoint '{checkpoint_path}' is missing required audio-video transformer weights"
            )
        if runtime_config.cross_attention_adaln:
            required_audio_prompt = (
                "model.diffusion_model.audio_prompt_adaln_single.linear.weight",
                "model.diffusion_model.transformer_blocks.0.audio_prompt_scale_shift_table",
            )
            if not all(key in checkpoint_keys for key in required_audio_prompt):
                raise RuntimeError(
                    f"LTX checkpoint '{checkpoint_path}' is missing required audio prompt AdaLN weights"
                )


def _assert_prompt_runtime_contract(
    prompt_context: PromptEncodingResult,
    runtime_config: _RuntimeModelConfig,
) -> None:
    if prompt_context.context_representation != "post_connector":
        raise ValueError(
            "LTX real generation requires post-connector prompt context for the "
            "current 22B bridge"
        )
    context_width = _context_width(prompt_context.video_context)
    if prompt_context.transformer_context_dim is not None:
        if prompt_context.transformer_context_dim != context_width:
            raise ValueError(
                "LTX prompt contract mismatch: prompt context width "
                f"{context_width} does not match declared transformer_context_dim "
                f"{prompt_context.transformer_context_dim}"
            )
        if prompt_context.transformer_context_dim != runtime_config.cross_attention_dim:
            raise ValueError(
                "LTX prompt/generation config mismatch: prompt transformer_context_dim "
                f"{prompt_context.transformer_context_dim} != runtime cross_attention_dim "
                f"{runtime_config.cross_attention_dim}"
            )
    if prompt_context.audio_context is None:
        raise ValueError(
            "LTX real generation requires audio_context from prompt_encode for the "
            "current 22B audio-video bridge"
        )
    if _attention_mask(prompt_context) is not None:
        raise ValueError(
            "LTX real generation requires an all-valid post-connector prompt mask "
            "for the current 22B audio-video bridge"
        )
    audio_context_width = _context_width(prompt_context.audio_context)
    if audio_context_width != runtime_config.audio_cross_attention_dim:
        raise ValueError(
            "LTX prompt/generation config mismatch: audio context width "
            f"{audio_context_width} != runtime audio_cross_attention_dim "
            f"{runtime_config.audio_cross_attention_dim}"
        )
    if (
        prompt_context.caption_proj_before_connector
        != runtime_config.caption_proj_before_connector
    ):
        raise ValueError(
            "LTX prompt/generation config mismatch for caption_proj_before_connector"
        )
    if prompt_context.rope_type != runtime_config.rope_type:
        raise ValueError("LTX prompt/generation config mismatch for rope_type")
    if prompt_context.double_precision_rope != runtime_config.double_precision_rope:
        raise ValueError(
            "LTX prompt/generation config mismatch for double_precision_rope"
        )
    if (
        prompt_context.transformer_apply_gated_attention is not None
        and prompt_context.transformer_apply_gated_attention
        != runtime_config.apply_gated_attention
    ):
        raise ValueError(
            "LTX prompt/generation config mismatch for apply_gated_attention"
        )
    if (
        prompt_context.transformer_cross_attention_adaln is not None
        and prompt_context.transformer_cross_attention_adaln
        != runtime_config.cross_attention_adaln
    ):
        raise ValueError(
            "LTX prompt/generation config mismatch for cross_attention_adaln"
        )
