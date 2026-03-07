# mypy: ignore-errors
from __future__ import annotations

import importlib
from dataclasses import replace

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
from .config import _load_checkpoint_prefixed_weights, _load_optional_json_config
from .debug import _looks_like_metal_oom
from .outputs import _audio_waveform_to_numpy
from .reference import _patch_reference_modules
from .reference_imports import _reference_path_on_sys_path
from .types import (
    _ConditioningPlan,
    _PaddedShape,
    _ReferenceImports,
    _RuntimeModelConfig,
)
from .video_stack import (
    _load_configured_upsampler,
    _load_configured_vae_decoder,
    _load_runtime_vocoder,
)


def _imports(self) -> _ReferenceImports:
    if self._reference_imports is not None:
        return self._reference_imports

    with _reference_path_on_sys_path():
        conditioning_module = importlib.import_module("mlx_video.conditioning")
        conditioning_latent_module = importlib.import_module(
            "mlx_video.conditioning.latent"
        )
        generate_module = importlib.import_module("mlx_video.generate")
        convert_module = importlib.import_module("mlx_video.convert")
        config_module = importlib.import_module("mlx_video.models.ltx.config")
        ltx_module = importlib.import_module("mlx_video.models.ltx.ltx")
        attention_module = importlib.import_module("mlx_video.models.ltx.attention")
        adaln_module = importlib.import_module("mlx_video.models.ltx.adaln")
        rope_module = importlib.import_module("mlx_video.models.ltx.rope")
        feed_forward_module = importlib.import_module(
            "mlx_video.models.ltx.feed_forward"
        )
        transformer_module = importlib.import_module("mlx_video.models.ltx.transformer")
        upsampler_module = importlib.import_module("mlx_video.models.ltx.upsampler")
        decoder_module = importlib.import_module(
            "mlx_video.models.ltx.video_vae.decoder"
        )
        encoder_module = importlib.import_module(
            "mlx_video.models.ltx.video_vae.encoder"
        )
        audio_vae_module = importlib.import_module(
            "mlx_video.models.ltx.audio_vae.audio_vae"
        )
        audio_vae_init_module = importlib.import_module(
            "mlx_video.models.ltx.audio_vae"
        )
        tiling_module = importlib.import_module("mlx_video.models.ltx.video_vae.tiling")
        utils_module = importlib.import_module("mlx_video.utils")

    self._reference_imports = _ReferenceImports(
        model_class=ltx_module.LTXModel,
        model_config_class=config_module.LTXModelConfig,
        model_type_enum=config_module.LTXModelType,
        rope_type_enum=config_module.LTXRopeType,
        BasicAVTransformerBlock=transformer_module.BasicAVTransformerBlock,
        attention_class=attention_module.Attention,
        preprocessor_class=ltx_module.TransformerArgsPreprocessor,
        multi_preprocessor_class=ltx_module.MultiModalTransformerArgsPreprocessor,
        feed_forward_class=feed_forward_module.FeedForward,
        adaln_class=adaln_module.AdaLayerNormSingle,
        apply_rotary_emb=rope_module.apply_rotary_emb,
        precompute_freqs_cis=rope_module.precompute_freqs_cis,
        rms_norm=utils_module.rms_norm,
        to_denoised=utils_module.to_denoised,
        scaled_dot_product_attention=attention_module.scaled_dot_product_attention,
        latent_state_class=conditioning_latent_module.LatentState,
        tiling_config_class=tiling_module.TilingConfig,
        condition_class=conditioning_module.VideoConditionByLatentIndex,
        stage_1_sigmas=tuple(float(value) for value in generate_module.STAGE_1_SIGMAS),
        stage_2_sigmas=tuple(float(value) for value in generate_module.STAGE_2_SIGMAS),
        apply_conditioning=conditioning_module.apply_conditioning,
        apply_denoise_mask=conditioning_latent_module.apply_denoise_mask,
        create_position_grid=generate_module.create_position_grid,
        create_audio_position_grid=generate_module.create_audio_position_grid,
        compute_audio_frames=generate_module.compute_audio_frames,
        load_image=utils_module.load_image,
        load_upsampler=upsampler_module.load_upsampler,
        load_vae_decoder=decoder_module.load_vae_decoder,
        load_vae_encoder=encoder_module.load_vae_encoder,
        load_audio_decoder=generate_module.load_audio_decoder,
        audio_encoder_class=audio_vae_module.AudioEncoder,
        audio_processor_class=audio_vae_init_module.AudioProcessor,
        audio_norm_type_enum=audio_vae_init_module.NormType,
        audio_causality_axis_enum=audio_vae_init_module.CausalityAxis,
        load_audio_vae_weights=convert_module.load_audio_vae_weights,
        load_vocoder=generate_module.load_vocoder,
        decode_audio=audio_vae_module.decode_audio,
        sanitize_audio_vae_weights=convert_module.sanitize_audio_vae_weights,
        sanitize_vocoder_weights=convert_module.sanitize_vocoder_weights,
        audio_vocoder_class=importlib.import_module(
            "mlx_video.models.ltx.audio_vae.vocoder"
        ).Vocoder,
        prepare_image_for_encoding=utils_module.prepare_image_for_encoding,
        upsample_latents=upsampler_module.upsample_latents,
        distilled_pipeline_type=generate_module.PipelineType.DISTILLED,
        audio_latent_channels=int(generate_module.AUDIO_LATENT_CHANNELS),
        audio_mel_bins=int(generate_module.AUDIO_MEL_BINS),
        audio_sample_rate=int(generate_module.AUDIO_SAMPLE_RATE),
    )
    _patch_reference_modules(self._reference_imports)
    return self._reference_imports


def _ensure_transformer(
    self,
    imports: _ReferenceImports,
    runtime_config: _RuntimeModelConfig,
    prompt_context: PromptEncodingResult,
) -> object:
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


def _ensure_vae_decoder(self, imports: _ReferenceImports) -> object:
    del imports
    if self._vae_decoder is None:
        self._vae_decoder = _load_configured_vae_decoder(self.checkpoint_path)
        mx.eval(self._vae_decoder.parameters())
    return self._vae_decoder


def _ensure_vae_encoder(self, imports: _ReferenceImports) -> object:
    if self._vae_encoder is None:
        self._vae_encoder = imports.load_vae_encoder(str(self.checkpoint_path))
        mx.eval(self._vae_encoder.parameters())
    return self._vae_encoder


def _ensure_upsampler(self, imports: _ReferenceImports) -> object:
    del imports
    if self._upsampler is None:
        self._upsampler = _load_configured_upsampler(self.spatial_upsampler_path)
        mx.eval(self._upsampler.parameters())
    return self._upsampler


def _ensure_audio_encoder(self, imports: _ReferenceImports) -> tuple[object, object]:
    if self._audio_encoder is not None and self._audio_processor is not None:
        return self._audio_encoder, self._audio_processor

    checkpoint_audio_weights = _load_checkpoint_prefixed_weights(
        self.checkpoint_path,
        prefixes=("audio_vae.",),
    )
    sanitized = imports.sanitize_audio_vae_weights(checkpoint_audio_weights)
    checkpoint_root = self.checkpoint_path.parent
    raw_config = _load_optional_json_config(
        checkpoint_root / "audio_vae" / "config.json"
    )

    ch = int(raw_config.get("base_channels", 128))
    ch_mult = tuple(raw_config.get("ch_mult", (1, 2, 4)))
    num_res_blocks = int(raw_config.get("num_res_blocks", 2))
    attn_resolutions = set(raw_config.get("attn_resolutions") or [])
    resolution = int(raw_config.get("resolution", 256))
    z_channels = int(raw_config.get("latent_channels", 8))
    dropout = float(raw_config.get("dropout", 0.0))
    in_channels = int(raw_config.get("in_channels", 2))
    norm_type = imports.audio_norm_type_enum(str(raw_config.get("norm_type", "pixel")))
    causality_axis = imports.audio_causality_axis_enum(
        str(raw_config.get("causality_axis", "height"))
    )
    mid_block_add_attention = bool(raw_config.get("mid_block_add_attention", True))
    sample_rate = int(raw_config.get("sample_rate", 16000))
    mel_hop_length = int(raw_config.get("mel_hop_length", 160))
    mel_bins = int(raw_config.get("mel_bins", 64))
    n_fft = int(raw_config.get("n_fft", 1024))
    is_causal = bool(raw_config.get("is_causal", True))

    encoder = imports.audio_encoder_class(
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
        resamp_with_conv=True,
        in_channels=in_channels,
        resolution=resolution,
        z_channels=z_channels,
        double_z=bool(raw_config.get("double_z", True)),
        norm_type=norm_type,
        causality_axis=causality_axis,
        mid_block_add_attention=mid_block_add_attention,
        sample_rate=sample_rate,
        mel_hop_length=mel_hop_length,
        n_fft=n_fft,
        mel_bins=mel_bins,
        is_causal=is_causal,
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
        sample_rate=sample_rate,
        mel_bins=mel_bins,
        mel_hop_length=mel_hop_length,
        n_fft=n_fft,
    )
    mx.eval(encoder.parameters())
    self._audio_encoder = encoder
    self._audio_processor = processor
    return encoder, processor


def _encode_audio_conditioning(
    self,
    *,
    imports: _ReferenceImports,
    audio_conditioning: AudioConditioningInput,
    audio_frames: int,
    model_dtype: mx.Dtype,
) -> tuple[object, npt.NDArray[np.float32], int]:
    encoder, processor = self._ensure_audio_encoder(imports)
    default_duration = None
    if audio_conditioning.max_duration_seconds is not None:
        default_duration = audio_conditioning.max_duration_seconds
    waveform, sample_rate = _decode_conditioning_audio_file(
        audio_conditioning.payload_path,
        sample_rate=processor.sample_rate,
        start_time_seconds=audio_conditioning.start_time_seconds,
        max_duration_seconds=default_duration,
    )
    mel = processor.waveform_to_mel(waveform.T, sample_rate)
    audio_latents = encoder(mx.array(mel).astype(mx.float32)).astype(model_dtype)
    audio_latents = _fit_audio_latents(audio_latents, target_frames=audio_frames)
    mx.eval(audio_latents)
    return audio_latents, waveform.astype(np.float32), int(sample_rate)


def _ensure_audio_stack(
    self, imports: _ReferenceImports
) -> tuple[object, object, int, str]:
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
    sanitized_audio_weights = {
        f"audio_vae.{key}": value
        for key, value in imports.sanitize_audio_vae_weights(
            checkpoint_audio_weights
        ).items()
    }
    checkpoint_root = self.checkpoint_path.parent
    if self._audio_decoder is None:
        self._audio_decoder = imports.load_audio_decoder(
            checkpoint_root,
            imports.distilled_pipeline_type,
            unified_weights=sanitized_audio_weights,
        )
        mx.eval(self._audio_decoder.parameters())
    if self._vocoder is None:
        (
            self._vocoder,
            self._audio_output_sample_rate,
            self._audio_backend,
        ) = _load_runtime_vocoder(
            checkpoint_path=self.checkpoint_path,
            checkpoint_weights=checkpoint_audio_weights,
            sanitize_vocoder_weights=imports.sanitize_vocoder_weights,
        )
        mx.eval(self._vocoder.parameters())
    mx.clear_cache()
    return (
        self._audio_decoder,
        self._vocoder,
        self._audio_output_sample_rate,
        self._audio_backend,
    )


def _decode_audio_waveform(
    self,
    *,
    imports: _ReferenceImports,
    audio_latents: object,
) -> tuple[npt.NDArray[np.float32] | None, int, str]:
    audio_decoder, vocoder, output_sample_rate, backend_label = (
        self._ensure_audio_stack(imports)
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
    self,
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

    vae_encoder = self._ensure_vae_encoder(imports)
    stage1_width = padded_shape.internal_width // 2
    stage1_height = padded_shape.internal_height // 2
    stage1_conditionings: list[object] = []
    stage2_conditionings: list[object] = []

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
    self,
    *,
    imports: _ReferenceImports,
    latents: object,
    conditionings: tuple[object, ...],
    sigmas: tuple[float, ...],
) -> object:
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
    return replace(state, latent=conditioned)


def _decode_video(
    self,
    *,
    imports: _ReferenceImports,
    vae_decoder: object,
    latents: object,
    padded_shape: _PaddedShape,
    num_frames: int,
) -> tuple[object, str]:
    tiling_config = imports.tiling_config_class.auto(
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
