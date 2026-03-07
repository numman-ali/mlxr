# mypy: ignore-errors
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from safetensors import safe_open

from ..prompt_encoding import PromptEncodingResult
from .conditioning import _attention_mask, _context_width
from .reference_imports import _REFERENCE_MLX_VIDEO_ROOT
from .types import _RuntimeModelConfig, _RuntimeVAEConfig, _RuntimeVocoderConfig


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, object]:
    with safe_open(str(checkpoint_path), framework="numpy") as checkpoint:
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
    with safe_open(str(checkpoint_path), framework="numpy") as checkpoint:
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


def _decode_conditioning_audio_file(
    audio_path: Path,
    *,
    sample_rate: int,
    start_time_seconds: float,
    max_duration_seconds: float | None,
) -> tuple[npt.NDArray[np.float32], int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
    ]
    if start_time_seconds > 0.0:
        command.extend(["-ss", str(start_time_seconds)])
    command.extend(["-i", str(audio_path)])
    if max_duration_seconds is not None:
        command.extend(["-t", str(max_duration_seconds)])
    command.extend(
        [
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode conditioning audio: {stderr}")
    waveform = np.frombuffer(result.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise RuntimeError(
            f"Conditioning audio '{audio_path}' decoded to an empty waveform"
        )
    channels = 2
    usable = waveform.size - (waveform.size % channels)
    if usable == 0:
        raise RuntimeError(
            f"Conditioning audio '{audio_path}' did not decode to stereo PCM frames"
        )
    waveform = waveform[:usable].reshape(-1, channels)
    return waveform.astype(np.float32), sample_rate


def _fit_audio_latents(audio_latents: object, *, target_frames: int) -> object:
    current_frames = int(audio_latents.shape[2])
    if current_frames == target_frames:
        return audio_latents
    if current_frames > target_frames:
        return audio_latents[:, :, :target_frames, :]
    pad_frames = target_frames - current_frames
    padding = mx.zeros(
        (
            int(audio_latents.shape[0]),
            int(audio_latents.shape[1]),
            pad_frames,
            int(audio_latents.shape[3]),
        ),
        dtype=audio_latents.dtype,
    )
    return mx.concatenate((audio_latents, padding), axis=2)


def _first_present(mapping: dict[str, object], keys: tuple[str, ...]) -> object | None:
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
        multiplier = int(params.get("multiplier", 1))
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

    raw_decoder_blocks = raw_vae_config.get("decoder_blocks")
    if not isinstance(raw_decoder_blocks, list) or not raw_decoder_blocks:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing decoder_blocks metadata"
        )

    decoder_blocks: list[tuple[str, object]] = []
    for raw_block in raw_decoder_blocks:
        if (
            isinstance(raw_block, list)
            and len(raw_block) == 2
            and isinstance(raw_block[0], str)
        ):
            decoder_blocks.append((raw_block[0], raw_block[1]))
            continue
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid decoder block metadata: {raw_block!r}"
        )

    norm_layer = str(raw_vae_config.get("norm_layer", "pixel_norm"))
    if norm_layer != "pixel_norm":
        raise NotImplementedError(
            f"LTX real generation only supports pixel_norm VAE decoders, got {norm_layer!r}"
        )

    return _RuntimeVAEConfig(
        latent_channels=int(raw_vae_config.get("latent_channels", 128)),
        out_channels=int(raw_vae_config.get("out_channels", 3)),
        patch_size=int(raw_vae_config.get("patch_size", 4)),
        decoder_blocks=tuple(decoder_blocks),
        base_channels=int(raw_vae_config.get("decoder_base_channels", 128)),
        spatial_padding_mode=str(
            raw_vae_config.get(
                "decoder_spatial_padding_mode",
                raw_vae_config.get("spatial_padding_mode", "reflect"),
            )
        ),
        timestep_conditioning=bool(raw_vae_config.get("timestep_conditioning", True)),
        norm_layer=norm_layer,
        causal_decoder=bool(raw_vae_config.get("causal_decoder", False)),
    )


def _runtime_vocoder_config(checkpoint_path: Path) -> _RuntimeVocoderConfig:
    metadata = _checkpoint_metadata(checkpoint_path)
    raw_vocoder_config = metadata.get("vocoder")
    if not isinstance(raw_vocoder_config, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing vocoder metadata"
        )

    uses_bwe = isinstance(raw_vocoder_config.get("bwe"), dict) and isinstance(
        raw_vocoder_config.get("vocoder"), dict
    )
    base_config = raw_vocoder_config["vocoder"] if uses_bwe else raw_vocoder_config
    if not isinstance(base_config, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid base vocoder metadata"
        )
    bwe_config = raw_vocoder_config.get("bwe") if uses_bwe else None
    bwe_output_sample_rate = (
        int(bwe_config.get("output_sampling_rate"))
        if isinstance(bwe_config, dict)
        and bwe_config.get("output_sampling_rate") is not None
        else None
    )
    output_sample_rate = (
        int(bwe_config.get("input_sampling_rate"))
        if isinstance(bwe_config, dict)
        and bwe_config.get("input_sampling_rate") is not None
        else int(base_config.get("output_sampling_rate", 24000))
    )

    return _RuntimeVocoderConfig(
        resblock_kernel_sizes=tuple(
            int(value) for value in base_config.get("resblock_kernel_sizes", [3, 7, 11])
        ),
        upsample_rates=tuple(
            int(value) for value in base_config.get("upsample_rates", [6, 5, 2, 2, 2])
        ),
        upsample_kernel_sizes=tuple(
            int(value)
            for value in base_config.get("upsample_kernel_sizes", [16, 15, 8, 4, 4])
        ),
        resblock_dilation_sizes=tuple(
            tuple(int(v) for v in block)
            for block in base_config.get(
                "resblock_dilation_sizes",
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            )
        ),
        upsample_initial_channel=int(base_config.get("upsample_initial_channel", 1024)),
        stereo=bool(base_config.get("stereo", True)),
        resblock=str(base_config.get("resblock", "1")),
        activation=str(base_config.get("activation", "snake")),
        use_tanh_at_final=bool(base_config.get("use_tanh_at_final", True)),
        apply_final_activation=bool(base_config.get("apply_final_activation", True)),
        use_bias_at_final=bool(base_config.get("use_bias_at_final", True)),
        output_sample_rate=output_sample_rate,
        uses_bwe=uses_bwe,
        bwe_output_sample_rate=bwe_output_sample_rate,
    )


def _validate_upsampler_layout(weights_path: Path) -> None:
    raw_weights = mx.load(str(weights_path))
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
    return bool(raw_transformer_config[key])


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
    explicit_bool = bool(explicit)
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

    def _int_or_default(value: object | None, default: int) -> int:
        if value is None:
            return default
        return int(value)

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
        num_attention_heads=int(raw_transformer_config.get("num_attention_heads", 32)),
        attention_head_dim=int(raw_transformer_config.get("attention_head_dim", 128)),
        in_channels=int(raw_transformer_config.get("in_channels", 128)),
        out_channels=int(raw_transformer_config.get("out_channels", 128)),
        num_layers=int(raw_transformer_config.get("num_layers", 48)),
        cross_attention_dim=int(
            raw_transformer_config.get("cross_attention_dim", 4096)
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
        positional_embedding_theta=float(
            raw_transformer_config.get("positional_embedding_theta", 10000.0)
        ),
        positional_embedding_max_pos=list(
            raw_transformer_config.get("positional_embedding_max_pos", [20, 2048, 2048])
        ),
        audio_positional_embedding_max_pos=list(
            raw_transformer_config.get("audio_positional_embedding_max_pos", [20])
        ),
        use_middle_indices_grid=bool(
            raw_transformer_config.get("use_middle_indices_grid", True)
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
        timestep_scale_multiplier=int(
            raw_transformer_config.get("timestep_scale_multiplier", 1000)
        ),
        av_ca_timestep_scale_multiplier=int(
            raw_transformer_config.get(
                "av_ca_timestep_scale_multiplier",
                raw_transformer_config.get("timestep_scale_multiplier", 1000),
            )
        ),
        norm_eps=float(raw_transformer_config.get("norm_eps", 1e-6)),
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
