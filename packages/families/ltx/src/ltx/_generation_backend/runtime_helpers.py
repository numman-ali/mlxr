from __future__ import annotations

import importlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from ..generation import AudioConditioningInput, ConditioningInput
from ..prompt_encoding import PromptEncodingResult
from .conditioning import (
    _context_width,
    _decode_conditioning_audio_file,
    _encode_conditioning_latent,
    _fit_audio_latents,
    _require_audio_context,
    _resolve_latent_frame_index,
)
from .config import (
    _load_checkpoint_prefixed_weights,
    _runtime_audio_encoder_config,
    _runtime_model_config,
)
from .debug import _looks_like_metal_oom
from .outputs import _audio_waveform_to_numpy
from .primitives import (
    STAGE_1_SIGMAS,
    STAGE_2_SIGMAS,
    LatentState,
    VideoConditionByLatentIndex,
    apply_conditioning,
    apply_denoise_mask,
    compute_audio_frames,
    create_audio_position_grid,
    create_position_grid,
    load_image,
    prepare_image_for_encoding,
    rms_norm,
    sanitize_audio_vae_weights,
    sanitize_vocoder_weights,
    to_denoised,
)
from .reference import _patch_reference_modules
from .reference_imports import _reference_path_on_sys_path
from .types import (
    MLXArray,
    _AudioDecoderLike,
    _AudioEncoderLike,
    _AudioProcessorLike,
    _AudioVideoTransformer,
    _ConditioningPlan,
    _ConditionLike,
    _LatentStateLike,
    _PaddedShape,
    _ReferenceImports,
    _RuntimeHelperHost,
    _RuntimeModelConfig,
    _UpsamplerLike,
    _VAEEncoder,
    _VideoDecoderLike,
    _VocoderLike,
)
from .video_stack import (
    _load_configured_upsampler,
    _load_configured_vae_decoder,
    _load_runtime_audio_decoder,
    _load_runtime_vae_encoder,
    _load_runtime_vocoder,
    _upsample_latents,
)
from .video_tiling import TilingConfig


def _to_denoised_ref(
    latents: MLXArray,
    velocity: MLXArray,
    sigma: MLXArray | float,
) -> MLXArray:
    return to_denoised(latents, velocity, sigma)


def _condition_ref(
    *,
    latent: MLXArray,
    frame_idx: int,
    strength: float,
) -> _ConditionLike:
    return VideoConditionByLatentIndex(
        latent=latent,
        frame_idx=frame_idx,
        strength=strength,
    )


def _apply_conditioning_ref(
    state: _LatentStateLike,
    conditionings: list[_ConditionLike],
) -> _LatentStateLike:
    return apply_conditioning(state, conditionings)


def _apply_denoise_mask_ref(
    denoised: MLXArray,
    clean_latent: MLXArray,
    denoise_mask: MLXArray,
) -> MLXArray:
    return apply_denoise_mask(denoised, clean_latent, denoise_mask)


def _create_position_grid_ref(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> MLXArray:
    return create_position_grid(
        batch_size,
        num_frames,
        height,
        width,
        temporal_scale=temporal_scale,
        spatial_scale=spatial_scale,
        fps=fps,
        causal_fix=causal_fix,
    )


def _create_audio_position_grid_ref(
    batch_size: int,
    audio_frames: int,
    *,
    sample_rate: int = 16000,
    hop_length: int = 160,
    downsample_factor: int = 4,
    is_causal: bool = True,
) -> MLXArray:
    return create_audio_position_grid(
        batch_size,
        audio_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        downsample_factor=downsample_factor,
        is_causal=is_causal,
    )


def _compute_audio_frames_ref(num_frames: int, fps: float) -> int:
    return compute_audio_frames(num_frames, fps)


def _load_image_ref(
    path: str | Path,
    *,
    height: int | None = None,
    width: int | None = None,
    dtype: mx.Dtype = mx.float32,
) -> MLXArray:
    return load_image(path, height=height, width=width, dtype=dtype)


def _prepare_image_for_encoding_ref(
    image: MLXArray,
    target_height: int,
    target_width: int,
    *,
    dtype: mx.Dtype = mx.float32,
) -> MLXArray:
    return prepare_image_for_encoding(
        image,
        target_height=target_height,
        target_width=target_width,
        dtype=dtype,
    )


def _imports(self: _RuntimeHelperHost) -> _ReferenceImports:
    if self._reference_imports is not None:
        return self._reference_imports

    with _reference_path_on_sys_path():
        config_module = importlib.import_module("mlx_video.models.ltx.config")
        ltx_module = importlib.import_module("mlx_video.models.ltx.ltx")
        attention_module = importlib.import_module("mlx_video.models.ltx.attention")
        adaln_module = importlib.import_module("mlx_video.models.ltx.adaln")
        rope_module = importlib.import_module("mlx_video.models.ltx.rope")
        transformer_module = importlib.import_module("mlx_video.models.ltx.transformer")
        audio_vae_module = importlib.import_module(
            "mlx_video.models.ltx.audio_vae.audio_vae"
        )
        audio_vae_init_module = importlib.import_module(
            "mlx_video.models.ltx.audio_vae"
        )

    audio_runtime_config = _runtime_audio_encoder_config(self.checkpoint_path.parent)
    runtime_model_config = _runtime_model_config(self.checkpoint_path)

    self._reference_imports = _ReferenceImports(
        model_class=ltx_module.LTXModel,
        model_config_class=config_module.LTXModelConfig,
        model_type_enum=config_module.LTXModelType,
        rope_type_enum=config_module.LTXRopeType,
        BasicAVTransformerBlock=transformer_module.BasicAVTransformerBlock,
        attention_class=attention_module.Attention,
        preprocessor_class=ltx_module.TransformerArgsPreprocessor,
        multi_preprocessor_class=ltx_module.MultiModalTransformerArgsPreprocessor,
        adaln_class=adaln_module.AdaLayerNormSingle,
        apply_rotary_emb=rope_module.apply_rotary_emb,
        precompute_freqs_cis=rope_module.precompute_freqs_cis,
        rms_norm=rms_norm,
        to_denoised=_to_denoised_ref,
        scaled_dot_product_attention=attention_module.scaled_dot_product_attention,
        latent_state_class=LatentState,
        condition_class=_condition_ref,
        stage_1_sigmas=STAGE_1_SIGMAS,
        stage_2_sigmas=STAGE_2_SIGMAS,
        apply_conditioning=_apply_conditioning_ref,
        apply_denoise_mask=_apply_denoise_mask_ref,
        create_position_grid=_create_position_grid_ref,
        create_audio_position_grid=_create_audio_position_grid_ref,
        compute_audio_frames=_compute_audio_frames_ref,
        load_image=_load_image_ref,
        load_vae_encoder=lambda checkpoint_path: _load_runtime_vae_encoder(
            checkpoint_path
        ),
        load_audio_decoder=lambda checkpoint_root, *, unified_weights: (
            _load_runtime_audio_decoder(
                checkpoint_root=checkpoint_root,
                audio_decoder_class=audio_vae_init_module.AudioDecoder,
                audio_norm_type_enum=audio_vae_init_module.NormType,
                audio_causality_axis_enum=audio_vae_init_module.CausalityAxis,
                sanitize_audio_vae_weights=sanitize_audio_vae_weights,
                unified_weights=unified_weights,
            )
        ),
        audio_encoder_class=audio_vae_module.AudioEncoder,
        audio_decoder_class=audio_vae_init_module.AudioDecoder,
        audio_processor_class=audio_vae_init_module.AudioProcessor,
        audio_norm_type_enum=audio_vae_init_module.NormType,
        audio_causality_axis_enum=audio_vae_init_module.CausalityAxis,
        decode_audio=audio_vae_module.decode_audio,
        prepare_image_for_encoding=_prepare_image_for_encoding_ref,
        upsample_latents=_upsample_latents,
        audio_latent_channels=audio_runtime_config.latent_channels,
        audio_mel_bins=runtime_model_config.audio_latent_mel_bins,
        audio_sample_rate=audio_runtime_config.sample_rate,
    )
    _patch_reference_modules(self._reference_imports)
    return self._reference_imports


def _ensure_transformer(
    self: _RuntimeHelperHost,
    imports: _ReferenceImports,
    runtime_config: _RuntimeModelConfig,
    prompt_context: PromptEncodingResult,
) -> _AudioVideoTransformer:
    if self._transformer is not None:
        return self._transformer

    config = imports.model_config_class.from_dict(
        {
            "model_type": imports.model_type_enum.AudioVideo,
            "num_attention_heads": runtime_config.num_attention_heads,
            "attention_head_dim": runtime_config.attention_head_dim,
            "in_channels": runtime_config.in_channels,
            "out_channels": runtime_config.out_channels,
            "num_layers": runtime_config.num_layers,
            "cross_attention_dim": runtime_config.cross_attention_dim,
            "caption_channels": _context_width(prompt_context.video_context),
            "audio_num_attention_heads": runtime_config.audio_num_attention_heads,
            "audio_attention_head_dim": runtime_config.audio_attention_head_dim,
            "audio_in_channels": runtime_config.audio_in_channels,
            "audio_out_channels": runtime_config.audio_out_channels,
            "audio_cross_attention_dim": runtime_config.audio_cross_attention_dim,
            "audio_caption_channels": _context_width(
                _require_audio_context(prompt_context)
            ),
            "positional_embedding_theta": runtime_config.positional_embedding_theta,
            "positional_embedding_max_pos": runtime_config.positional_embedding_max_pos,
            "audio_positional_embedding_max_pos": runtime_config.audio_positional_embedding_max_pos,
            "use_middle_indices_grid": runtime_config.use_middle_indices_grid,
            "rope_type": prompt_context.rope_type,
            "double_precision_rope": prompt_context.double_precision_rope,
            "timestep_scale_multiplier": runtime_config.timestep_scale_multiplier,
            "av_ca_timestep_scale_multiplier": runtime_config.av_ca_timestep_scale_multiplier,
            "norm_eps": runtime_config.norm_eps,
        }
    )
    config.apply_gated_attention = runtime_config.apply_gated_attention
    config.cross_attention_adaln = runtime_config.cross_attention_adaln
    config.caption_proj_before_connector = prompt_context.caption_proj_before_connector
    transformer = imports.model_class.from_pretrained(
        self.checkpoint_path,
        config=config,
        strict=True,
    )
    if runtime_config.apply_gated_attention:
        first_block = next(iter(transformer.transformer_blocks.values()))
        required_gate_attrs = (
            "attn1",
            "attn2",
            "audio_attn1",
            "audio_attn2",
            "audio_to_video_attn",
            "video_to_audio_attn",
        )
        missing_gate_attrs = [
            attr
            for attr in required_gate_attrs
            if not hasattr(getattr(first_block, attr, None), "to_gate_logits")
        ]
        if missing_gate_attrs:
            raise RuntimeError(
                "LTX checkpoint requires gated attention, but the MLX transformer "
                "bridge did not instantiate gate projections for "
                + ", ".join(missing_gate_attrs)
            )
    if runtime_config.cross_attention_adaln:
        if not hasattr(transformer, "prompt_adaln_single"):
            raise RuntimeError(
                "LTX checkpoint requires cross-attention AdaLN, but the MLX "
                "transformer bridge did not instantiate prompt AdaLN"
            )
        if not hasattr(transformer, "audio_prompt_adaln_single"):
            raise RuntimeError(
                "LTX checkpoint requires audio prompt AdaLN, but the MLX "
                "transformer bridge did not instantiate the audio prompt AdaLN"
            )
        linear = getattr(transformer.adaln_single, "linear", None)
        if linear is None or int(linear.weight.shape[0]) != 9 * transformer.inner_dim:
            raise RuntimeError(
                "LTX checkpoint requires 9-way AdaLN modulation, but the MLX "
                "transformer bridge is still using the wrong AdaLN shape"
            )
        audio_linear = getattr(transformer.audio_adaln_single, "linear", None)
        if (
            audio_linear is None
            or int(audio_linear.weight.shape[0]) != 9 * transformer.audio_inner_dim
        ):
            raise RuntimeError(
                "LTX checkpoint requires 9-way audio AdaLN modulation, but the "
                "MLX transformer bridge is still using the wrong audio AdaLN shape"
            )
        first_block = next(iter(transformer.transformer_blocks.values()))
        if not hasattr(first_block, "audio_prompt_scale_shift_table"):
            raise RuntimeError(
                "LTX checkpoint requires audio prompt AdaLN tables, but the MLX "
                "transformer bridge did not instantiate them"
            )
    mx.eval(transformer.parameters())
    self._transformer = transformer
    return transformer


def _ensure_vae_decoder(
    self: _RuntimeHelperHost, imports: _ReferenceImports
) -> _VideoDecoderLike:
    if self._vae_decoder is None:
        vae_decoder = _load_configured_vae_decoder(self.checkpoint_path)
        mx.eval(vae_decoder.parameters())
        self._vae_decoder = vae_decoder
    return self._vae_decoder


def _ensure_vae_encoder(
    self: _RuntimeHelperHost, imports: _ReferenceImports
) -> _VAEEncoder:
    if self._vae_encoder is None:
        self._vae_encoder = imports.load_vae_encoder(self.checkpoint_path)
        mx.eval(self._vae_encoder.parameters())
    return self._vae_encoder


def _ensure_upsampler(
    self: _RuntimeHelperHost, imports: _ReferenceImports
) -> _UpsamplerLike:
    if self._upsampler is None:
        upsampler = _load_configured_upsampler(
            self.spatial_upsampler_path,
        )
        mx.eval(upsampler.parameters())
        self._upsampler = upsampler
    return self._upsampler


def _ensure_audio_encoder(
    self: _RuntimeHelperHost, imports: _ReferenceImports
) -> tuple[_AudioEncoderLike, _AudioProcessorLike]:
    if self._audio_encoder is not None and self._audio_processor is not None:
        return self._audio_encoder, self._audio_processor

    checkpoint_audio_weights = _load_checkpoint_prefixed_weights(
        self.checkpoint_path,
        prefixes=("audio_vae.",),
    )
    sanitized = sanitize_audio_vae_weights(checkpoint_audio_weights)
    checkpoint_root = self.checkpoint_path.parent
    audio_config = _runtime_audio_encoder_config(checkpoint_root)
    norm_type = imports.audio_norm_type_enum(audio_config.norm_type)
    causality_axis = imports.audio_causality_axis_enum(audio_config.causality_axis)

    encoder = imports.audio_encoder_class(
        ch=audio_config.base_channels,
        ch_mult=audio_config.ch_mult,
        num_res_blocks=audio_config.num_res_blocks,
        attn_resolutions=audio_config.attn_resolutions,
        dropout=audio_config.dropout,
        resamp_with_conv=True,
        in_channels=audio_config.in_channels,
        resolution=audio_config.resolution,
        z_channels=audio_config.latent_channels,
        double_z=audio_config.double_z,
        norm_type=norm_type,
        causality_axis=causality_axis,
        mid_block_add_attention=audio_config.mid_block_add_attention,
        sample_rate=audio_config.sample_rate,
        mel_hop_length=audio_config.mel_hop_length,
        n_fft=audio_config.n_fft,
        mel_bins=audio_config.mel_bins,
        is_causal=audio_config.is_causal,
    )
    encoder_weights = {
        key.replace("encoder.", ""): value
        for key, value in sanitized.items()
        if key.startswith("encoder.")
    }
    if encoder_weights:
        encoder.load_weights(list(encoder_weights.items()), strict=False)
    if "per_channel_statistics._mean_of_means" in sanitized:
        encoder.per_channel_statistics._mean_of_means = sanitized[
            "per_channel_statistics._mean_of_means"
        ]
    if "per_channel_statistics._std_of_means" in sanitized:
        encoder.per_channel_statistics._std_of_means = sanitized[
            "per_channel_statistics._std_of_means"
        ]
    processor = imports.audio_processor_class(
        sample_rate=audio_config.sample_rate,
        mel_bins=audio_config.mel_bins,
        mel_hop_length=audio_config.mel_hop_length,
        n_fft=audio_config.n_fft,
    )
    mx.eval(encoder.parameters())
    self._audio_encoder = encoder
    self._audio_processor = processor
    return encoder, processor


def _encode_audio_conditioning(
    self: _RuntimeHelperHost,
    *,
    imports: _ReferenceImports,
    audio_conditioning: AudioConditioningInput,
    audio_frames: int,
    model_dtype: mx.Dtype,
) -> tuple[MLXArray, npt.NDArray[np.float32], int]:
    encoder, processor = _ensure_audio_encoder(self, imports)
    default_duration = None
    if audio_conditioning.max_duration_seconds is not None:
        default_duration = audio_conditioning.max_duration_seconds
    waveform, sample_rate = _decode_conditioning_audio_file(
        audio_conditioning.payload_path,
        sample_rate=processor.sample_rate,
        start_time_seconds=audio_conditioning.start_time_seconds,
        max_duration_seconds=default_duration,
    )
    mel = _normalize_audio_mel_layout(
        processor.waveform_to_mel(waveform.T, sample_rate),
        input_channels=encoder.in_channels,
    )
    audio_latents = encoder(mx.array(mel).astype(mx.float32)).astype(model_dtype)
    actual_shape = tuple(int(size) for size in audio_latents.shape)
    if (
        actual_shape[1] != imports.audio_latent_channels
        or actual_shape[3] != imports.audio_mel_bins
    ):
        raise RuntimeError(
            "LTX audio conditioning latent shape does not match the runtime transformer contract: "
            f"got {actual_shape}, expected channels={imports.audio_latent_channels} "
            f"and mel_bins={imports.audio_mel_bins}"
        )
    audio_latents = _fit_audio_latents(audio_latents, target_frames=audio_frames)
    mx.eval(audio_latents)
    return audio_latents, waveform.astype(np.float32), int(sample_rate)


def _normalize_audio_mel_layout(
    mel: npt.NDArray[np.float32], *, input_channels: int
) -> npt.NDArray[np.float32]:
    if mel.ndim != 4:
        raise RuntimeError(
            f"LTX audio conditioning mel spectrogram must be 4D, got shape {mel.shape}"
        )
    if mel.shape[1] == input_channels:
        return mel.astype(np.float32, copy=False)
    if mel.shape[2] == input_channels:
        return np.transpose(mel, (0, 2, 3, 1)).astype(np.float32, copy=False)
    if mel.shape[3] == input_channels:
        return np.transpose(mel, (0, 3, 1, 2)).astype(np.float32, copy=False)
    raise RuntimeError(
        "LTX audio conditioning mel spectrogram must expose the encoder input channels "
        f"on axis 1, 2, or 3; got shape {mel.shape}"
    )


def _ensure_audio_stack(
    self: _RuntimeHelperHost, imports: _ReferenceImports
) -> tuple[_AudioDecoderLike, _VocoderLike, int, str]:
    if (
        self._audio_decoder is not None
        and self._vocoder is not None
        and self._audio_output_sample_rate is not None
        and self._audio_backend is not None
    ):
        return (
            self._audio_decoder,
            self._vocoder,
            self._audio_output_sample_rate,
            self._audio_backend,
        )

    checkpoint_audio_weights = _load_checkpoint_prefixed_weights(
        self.checkpoint_path,
        prefixes=("audio_vae.", "vocoder."),
    )
    checkpoint_root = self.checkpoint_path.parent
    if self._audio_decoder is None:
        self._audio_decoder = imports.load_audio_decoder(
            checkpoint_root,
            unified_weights=checkpoint_audio_weights,
        )
        mx.eval(self._audio_decoder.parameters())
    if self._vocoder is None:
        (
            vocoder,
            output_sample_rate,
            backend_label,
        ) = _load_runtime_vocoder(
            checkpoint_path=self.checkpoint_path,
            checkpoint_weights=checkpoint_audio_weights,
            sanitize_vocoder_weights=sanitize_vocoder_weights,
        )
        mx.eval(vocoder.parameters())
        self._vocoder = vocoder
        self._audio_output_sample_rate = output_sample_rate
        self._audio_backend = backend_label
    mx.clear_cache()
    if (
        self._audio_decoder is None
        or self._vocoder is None
        or self._audio_output_sample_rate is None
        or self._audio_backend is None
    ):
        raise RuntimeError("LTX audio stack did not initialize correctly")
    return (
        self._audio_decoder,
        self._vocoder,
        self._audio_output_sample_rate,
        self._audio_backend,
    )


def _decode_audio_waveform(
    self: _RuntimeHelperHost,
    *,
    imports: _ReferenceImports,
    audio_latents: MLXArray,
) -> tuple[npt.NDArray[np.float32] | None, int, str]:
    audio_decoder, vocoder, output_sample_rate, backend_label = _ensure_audio_stack(
        self, imports
    )
    decoded_audio = imports.decode_audio(
        audio_latents.astype(mx.float32),
        audio_decoder,
        vocoder,
    )
    mx.eval(decoded_audio)
    return (
        _audio_waveform_to_numpy(decoded_audio),
        output_sample_rate,
        backend_label,
    )


def _prepare_conditionings(
    self: _RuntimeHelperHost,
    *,
    imports: _ReferenceImports,
    conditioning_inputs: tuple[ConditioningInput, ...],
    num_frames: int,
    latent_frames: int,
    padded_shape: _PaddedShape,
    model_dtype: mx.Dtype,
) -> _ConditioningPlan:
    if not conditioning_inputs:
        return _ConditioningPlan(stage1=(), stage2=())

    vae_encoder = _ensure_vae_encoder(self, imports)
    stage1_width = padded_shape.internal_width // 2
    stage1_height = padded_shape.internal_height // 2
    stage1_conditionings: list[_ConditionLike] = []
    stage2_conditionings: list[_ConditionLike] = []

    for conditioning_input in conditioning_inputs:
        resolved_index = _resolve_latent_frame_index(
            frame_index=conditioning_input.frame_index,
            num_frames=num_frames,
            latent_frames=latent_frames,
        )
        stage1_latent = _encode_conditioning_latent(
            imports=imports,
            vae_encoder=vae_encoder,
            payload_path=conditioning_input.payload_path,
            width=stage1_width,
            height=stage1_height,
            dtype=model_dtype,
        )
        stage2_latent = _encode_conditioning_latent(
            imports=imports,
            vae_encoder=vae_encoder,
            payload_path=conditioning_input.payload_path,
            width=padded_shape.internal_width,
            height=padded_shape.internal_height,
            dtype=model_dtype,
        )
        stage1_conditionings.append(
            imports.condition_class(
                latent=stage1_latent,
                frame_idx=resolved_index,
                strength=float(conditioning_input.strength),
            )
        )
        stage2_conditionings.append(
            imports.condition_class(
                latent=stage2_latent,
                frame_idx=resolved_index,
                strength=float(conditioning_input.strength),
            )
        )

    return _ConditioningPlan(
        stage1=tuple(stage1_conditionings),
        stage2=tuple(stage2_conditionings),
    )


def _apply_conditionings_to_stage(
    self: _RuntimeHelperHost,
    *,
    imports: _ReferenceImports,
    latents: MLXArray,
    conditionings: tuple[_ConditionLike, ...],
    sigmas: tuple[float, ...],
) -> _LatentStateLike:
    model_dtype = latents.dtype
    state = imports.latent_state_class(
        latent=latents,
        clean_latent=mx.zeros_like(latents),
        denoise_mask=mx.ones(
            (int(latents.shape[0]), 1, int(latents.shape[2]), 1, 1),
            dtype=model_dtype,
        ),
    )
    state = imports.apply_conditioning(state, list(conditionings))
    noise = mx.random.normal(latents.shape).astype(model_dtype)
    noise_scale = mx.array(float(sigmas[0]), dtype=model_dtype)
    scaled_mask = state.denoise_mask * noise_scale
    conditioned = (
        noise * scaled_mask
        + state.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask)
    ).astype(model_dtype)
    mx.eval(conditioned)
    updated_state = state.clone()
    updated_state.latent = conditioned
    return updated_state


def _decode_video(
    self: _RuntimeHelperHost,
    *,
    imports: _ReferenceImports,
    vae_decoder: _VideoDecoderLike,
    latents: MLXArray,
    padded_shape: _PaddedShape,
    num_frames: int,
) -> tuple[MLXArray, str]:
    tiling_config = TilingConfig.auto(
        padded_shape.internal_height,
        padded_shape.internal_width,
        num_frames,
    )
    if tiling_config is None:
        video = vae_decoder(latents, chunked_conv=False)
        mx.eval(video)
        return video, "none"

    try:
        video = vae_decoder(latents, chunked_conv=False)
        mx.eval(video)
        return video, "auto-fast"
    except Exception as exc:
        if not _looks_like_metal_oom(exc):
            raise

    try:
        video = vae_decoder(latents, chunked_conv=True)
        mx.eval(video)
        return video, "chunked"
    except Exception as exc:
        if not _looks_like_metal_oom(exc):
            raise

    video = vae_decoder.decode_tiled(
        latents,
        tiling_config=tiling_config,
        tiling_mode="auto",
        debug=False,
    )
    mx.eval(video)
    return video, "auto-tiled"
