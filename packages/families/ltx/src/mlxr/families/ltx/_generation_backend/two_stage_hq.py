from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from mlxr.core.runtime import TraceRecorder, mlx_memory_snapshot

from ..family_options import effective_seed
from ..generation import AudioConditioningInput, ConditioningInput, GeneratedVideo
from ..prompt_encoding import PromptEncodingResult
from .conditioning import (
    _optional_negative_audio_context,
    _optional_negative_video_context,
    _prompt_context_dtype,
    _prompt_signature,
    _require_audio_context,
    _require_video_context,
    _resolve_padded_shape,
)
from .config import _runtime_model_config
from .debug import _debug_trace_enabled, _debug_trace_sync_enabled, _elapsed_ms
from .one_stage import (
    _looks_like_distilled_checkpoint,
    _require_audio_latents,
    _require_audio_video_transformer,
    _require_video_transformer,
)
from .outputs import _decode_to_uint8_frames
from .runtime_helpers import _ensure_transformer_with_loras, _release_transformers
from .sampling import _assert_prompt_runtime_contract, _guided_prediction
from .scheduler import LTX2Scheduler
from .types import (
    MLXArray,
    _AudioVideoTransformer,
    _ConditionLike,
    _LatentStateLike,
    _PaddedShape,
    _PatchedModality,
    _ReferenceImports,
    _RuntimeHelperHost,
    _RuntimeModelConfig,
    _UpsamplerLike,
    _VideoDecoderLike,
    _VideoTransformer,
)

_HQ_STAGE_1_DEFAULT_LORA_STRENGTH = 0.25
_HQ_STAGE_2_DEFAULT_LORA_STRENGTH = 0.5
_HQ_VIDEO_RESCALE_SCALE = 0.45
_HQ_AUDIO_RESCALE_SCALE = 1.0
_HQ_VIDEO_CFG_SCALE = 3.0
_HQ_AUDIO_CFG_SCALE = 7.0
_RES2S_C2 = 0.5
_RES2S_BONGMATH_MAX_ITER = 100


class _TwoStageHQHost(_RuntimeHelperHost, Protocol):
    distilled_lora_path: Path | None
    hq_stage_1_distilled_lora_strength: float
    hq_stage_2_distilled_lora_strength: float

    def _imports(self) -> _ReferenceImports: ...

    def _release_checkpoint_reader(self) -> None: ...

    def _ensure_vae_statistics(self) -> tuple[MLXArray, MLXArray]: ...

    def _ensure_upsampler(self, imports: _ReferenceImports) -> _UpsamplerLike: ...

    def _ensure_vae_decoder(self, imports: _ReferenceImports) -> _VideoDecoderLike: ...

    def _prepare_conditionings(
        self,
        *,
        imports: _ReferenceImports,
        conditioning_inputs: tuple[ConditioningInput, ...],
        num_frames: int,
        latent_frames: int,
        padded_shape: _PaddedShape,
        model_dtype: mx.Dtype,
        replace_first_frame_latent: bool = True,
    ) -> object: ...

    def _apply_conditionings_to_stage(
        self,
        *,
        imports: _ReferenceImports,
        latents: MLXArray,
        conditionings: tuple[_ConditionLike, ...],
        sigmas: tuple[float, ...],
    ) -> _LatentStateLike: ...

    def _decode_audio_waveform(
        self,
        *,
        imports: _ReferenceImports,
        audio_latents: MLXArray,
    ) -> tuple[npt.NDArray[np.float32] | None, int, str]: ...

    def _decode_video(
        self,
        *,
        imports: _ReferenceImports,
        vae_decoder: _VideoDecoderLike,
        latents: MLXArray,
        padded_shape: _PaddedShape,
        num_frames: int,
    ) -> tuple[MLXArray, str]: ...


def generate_two_stage_hq(
    host: _TwoStageHQHost,
    *,
    prompt_context: PromptEncodingResult,
    conditioning_inputs: tuple[ConditioningInput, ...],
    audio_conditioning: AudioConditioningInput | None,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int | None,
    num_inference_steps: int | None,
    guidance_scale: float | None,
) -> GeneratedVideo:
    if _looks_like_distilled_checkpoint(host.checkpoint_path):
        raise ValueError(
            "LTX two_stage_hq requires the full dev checkpoint, not the distilled checkpoint"
        )
    if host.spatial_upsampler_path is None:
        raise ValueError(
            "LTX two_stage_hq requires the x2 spatial upsampler component in the current artifact"
        )
    if host.distilled_lora_path is None:
        raise ValueError(
            "LTX two_stage_hq requires the distilled LoRA component in the current artifact"
        )
    if audio_conditioning is not None:
        raise ValueError(
            "LTX two_stage_hq currently supports text/image generation only; "
            "audio conditioning remains the separate A2Vid row"
        )

    imports = host._imports()
    runtime_config = _runtime_model_config(host.checkpoint_path)
    _assert_prompt_runtime_contract(
        prompt_context,
        runtime_config,
        audio_required=host._audio_enabled,
    )

    video_context = _require_video_context(prompt_context)
    audio_context = (
        _require_audio_context(prompt_context) if host._audio_enabled else None
    )
    negative_guidance_active = prompt_context.negative_prompt_text is not None
    negative_video_context = (
        _optional_negative_video_context(prompt_context)
        if negative_guidance_active
        else None
    )
    negative_audio_context = (
        _optional_negative_audio_context(prompt_context)
        if host._audio_enabled and negative_guidance_active
        else None
    )

    padded_shape = _resolve_padded_shape(width=width, height=height)
    effective_generation_seed = effective_seed(seed=seed)
    model_dtype = _prompt_context_dtype(video_context)
    latent_frames = 1 + (num_frames - 1) // 8
    stage1_height = padded_shape.internal_height // 2 // 32
    stage1_width = padded_shape.internal_width // 2 // 32
    stage2_height = padded_shape.internal_height // 32
    stage2_width = padded_shape.internal_width // 32
    audio_frames = (
        int(imports.compute_audio_frames(num_frames, float(fps)))
        if host._audio_enabled
        else 0
    )
    steps = 15 if num_inference_steps is None else num_inference_steps
    if steps <= 0:
        raise ValueError("LTX two_stage_hq requires num_inference_steps > 0")
    video_cfg_scale = _HQ_VIDEO_CFG_SCALE if guidance_scale is None else guidance_scale
    audio_cfg_scale = _HQ_AUDIO_CFG_SCALE if guidance_scale is None else guidance_scale

    trace_recorder = TraceRecorder(enabled=_debug_trace_enabled())
    trace_sync = _debug_trace_sync_enabled()
    del trace_sync  # reserved for future step-local HQ tracing

    with trace_recorder.span("ltx.prepare_conditionings", snapshot=mlx_memory_snapshot):
        conditioning_plan = host._prepare_conditionings(
            imports=imports,
            conditioning_inputs=conditioning_inputs,
            num_frames=num_frames,
            latent_frames=latent_frames,
            padded_shape=padded_shape,
            model_dtype=model_dtype,
        )
    host._vae_encoder = None
    mx.clear_cache()
    with trace_recorder.span("ltx.ensure_transformer", snapshot=mlx_memory_snapshot):
        stage1_transformer = _ensure_transformer_with_loras(
            host,
            imports=imports,
            runtime_config=runtime_config,
            prompt_context=prompt_context,
            lora_paths=(host.distilled_lora_path,),
            lora_scales=(host.hq_stage_1_distilled_lora_strength,),
        )
    host._release_checkpoint_reader()

    mx.random.seed(effective_generation_seed)
    timings_ms: dict[str, float] = {}

    stage1_started = time.perf_counter()
    stage1_positions = imports.create_position_grid(
        1,
        latent_frames,
        stage1_height,
        stage1_width,
        fps=float(fps),
    )
    stage1_audio_positions = (
        imports.create_audio_position_grid(1, audio_frames)
        if host._audio_enabled
        else None
    )
    latents = mx.random.normal(
        (1, 128, latent_frames, stage1_height, stage1_width)
    ).astype(model_dtype)
    audio_latents = (
        mx.random.normal(
            (
                1,
                imports.audio_latent_channels,
                audio_frames,
                imports.audio_mel_bins,
            )
        ).astype(model_dtype)
        if host._audio_enabled
        else None
    )
    stage1_sigmas = LTX2Scheduler().execute(steps=steps, latent=latents)
    stage1_conditionings = getattr(conditioning_plan, "stage1", ())
    if stage1_conditionings:
        stage1_state = host._apply_conditionings_to_stage(
            imports=imports,
            latents=latents,
            conditionings=stage1_conditionings,
            sigmas=stage1_sigmas,
        )
        latents = stage1_state.latent
    else:
        stage1_state = None

    if host._audio_enabled:
        if (
            audio_context is None
            or stage1_audio_positions is None
            or audio_latents is None
        ):
            raise RuntimeError(
                "LTX two_stage_hq audio-video path requires audio context"
            )
        latents, audio_latents = _res2s_audio_video_loop(
            imports=imports,
            transformer=_require_audio_video_transformer(stage1_transformer),
            latents=latents,
            positions=stage1_positions,
            text_embeddings=video_context,
            audio_latents=audio_latents,
            audio_positions=stage1_audio_positions,
            audio_embeddings=audio_context,
            negative_text_embeddings=negative_video_context,
            negative_audio_embeddings=negative_audio_context,
            sigmas=stage1_sigmas,
            state=stage1_state,
            audio_state=None,
            runtime_config=runtime_config,
            video_cfg_scale=video_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            video_rescale_scale=_HQ_VIDEO_RESCALE_SCALE,
            audio_rescale_scale=_HQ_AUDIO_RESCALE_SCALE,
        )
        mx.eval(latents, audio_latents)
    else:
        latents = _res2s_video_only_loop(
            imports=imports,
            transformer=_require_video_transformer(stage1_transformer),
            latents=latents,
            positions=stage1_positions,
            text_embeddings=video_context,
            negative_text_embeddings=negative_video_context,
            sigmas=stage1_sigmas,
            state=stage1_state,
            runtime_config=runtime_config,
            video_cfg_scale=video_cfg_scale,
            video_rescale_scale=_HQ_VIDEO_RESCALE_SCALE,
        )
        mx.eval(latents)
    timings_ms["stage1_duration_ms"] = _elapsed_ms(stage1_started)
    # HQ uses distinct distilled-LoRA strengths per stage, so the stage-1 LoRA
    # variant must be dropped before stage 2 builds its own transformer.
    stage1_transformer = None
    stage1_state = None
    del stage1_positions
    del stage1_audio_positions
    del stage1_sigmas
    del stage1_conditionings
    _release_transformers(host)

    upsample_started = time.perf_counter()
    latent_mean, latent_std = host._ensure_vae_statistics()
    with trace_recorder.span("ltx.ensure_upsampler", snapshot=mlx_memory_snapshot):
        upsampler = host._ensure_upsampler(imports)
    latents = imports.upsample_latents(latents, upsampler, latent_mean, latent_std)
    mx.eval(latents)
    upsampler = None
    host._upsampler = None
    mx.clear_cache()
    timings_ms["upsample_duration_ms"] = _elapsed_ms(upsample_started)

    stage2_started = time.perf_counter()
    stage2_transformer = _ensure_transformer_with_loras(
        host,
        imports=imports,
        runtime_config=runtime_config,
        prompt_context=prompt_context,
        lora_paths=(host.distilled_lora_path,),
        lora_scales=(host.hq_stage_2_distilled_lora_strength,),
    )
    stage2_positions = imports.create_position_grid(
        1,
        latent_frames,
        stage2_height,
        stage2_width,
        fps=float(fps),
    )
    stage2_audio_positions = (
        imports.create_audio_position_grid(1, audio_frames)
        if host._audio_enabled
        else None
    )
    stage2_conditionings = getattr(conditioning_plan, "stage2", ())
    if stage2_conditionings:
        stage2_state = host._apply_conditionings_to_stage(
            imports=imports,
            latents=latents,
            conditionings=stage2_conditionings,
            sigmas=imports.stage_2_sigmas,
        )
        latents = stage2_state.latent
    else:
        stage2_state = None
    noise_scale = mx.array(float(imports.stage_2_sigmas[0]), dtype=model_dtype)
    one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
    latents = (
        mx.random.normal(latents.shape).astype(model_dtype) * noise_scale
        + latents * one_minus_scale
    ).astype(model_dtype)
    if host._audio_enabled:
        audio_latents = _require_audio_latents(audio_latents)
        audio_latents = (
            mx.random.normal(audio_latents.shape).astype(model_dtype) * noise_scale
            + audio_latents * one_minus_scale
        ).astype(model_dtype)
        mx.eval(latents, audio_latents)
    else:
        mx.eval(latents)

    if host._audio_enabled:
        if audio_context is None or stage2_audio_positions is None:
            raise RuntimeError("LTX two_stage_hq audio-video path requires audio state")
        latents, audio_latents = _res2s_audio_video_loop(
            imports=imports,
            transformer=_require_audio_video_transformer(stage2_transformer),
            latents=latents,
            positions=stage2_positions,
            text_embeddings=video_context,
            audio_latents=_require_audio_latents(audio_latents),
            audio_positions=stage2_audio_positions,
            audio_embeddings=audio_context,
            negative_text_embeddings=None,
            negative_audio_embeddings=None,
            sigmas=imports.stage_2_sigmas,
            state=stage2_state,
            audio_state=None,
            runtime_config=runtime_config,
            video_cfg_scale=1.0,
            audio_cfg_scale=1.0,
            video_rescale_scale=0.0,
            audio_rescale_scale=0.0,
        )
        mx.eval(latents, audio_latents)
    else:
        latents = _res2s_video_only_loop(
            imports=imports,
            transformer=_require_video_transformer(stage2_transformer),
            latents=latents,
            positions=stage2_positions,
            text_embeddings=video_context,
            negative_text_embeddings=None,
            sigmas=imports.stage_2_sigmas,
            state=stage2_state,
            runtime_config=runtime_config,
            video_cfg_scale=1.0,
            video_rescale_scale=0.0,
        )
        mx.eval(latents)
    timings_ms["stage2_duration_ms"] = _elapsed_ms(stage2_started)
    stage2_transformer = None
    stage2_state = None
    del stage2_positions
    del stage2_audio_positions
    del stage2_conditionings
    _release_transformers(host)

    decode_started = time.perf_counter()
    with trace_recorder.span(
        "ltx.ensure_vae_decoder.final", snapshot=mlx_memory_snapshot
    ):
        vae_decoder = host._ensure_vae_decoder(imports)
    decoded_video, tiling_mode = host._decode_video(
        imports=imports,
        vae_decoder=vae_decoder,
        latents=latents,
        padded_shape=padded_shape,
        num_frames=num_frames,
    )
    frames_uint8 = _decode_to_uint8_frames(decoded_video, padded_shape=padded_shape)
    decode_duration_ms = _elapsed_ms(decode_started)

    audio_waveform: npt.NDArray[np.float32] | None = None
    audio_sample_rate: int | None = None
    audio_backend: str | None = None
    if host._audio_enabled:
        audio_waveform, audio_sample_rate, audio_backend = host._decode_audio_waveform(
            imports=imports,
            audio_latents=_require_audio_latents(audio_latents),
        )

    metadata: dict[str, object] = {
        "pipeline_kind": "two_stage_hq"
        if host._audio_enabled
        else "two_stage_hq_video_only",
        "output_width": width,
        "output_height": height,
        "output_frames": num_frames,
        "padded_width": padded_shape.internal_width,
        "padded_height": padded_shape.internal_height,
        "latent_frames": latent_frames,
        "conditioning_count": len(conditioning_inputs),
        "audio_conditioned": False,
        "tiling_mode": tiling_mode,
        "hq_stage_1_distilled_lora_strength": host.hq_stage_1_distilled_lora_strength,
        "hq_stage_2_distilled_lora_strength": host.hq_stage_2_distilled_lora_strength,
        "timings_ms": {
            **timings_ms,
            "decode_duration_ms": decode_duration_ms,
        },
        "backend": "mlxr_ltx_two_stage_hq"
        if host._audio_enabled
        else "mlxr_ltx_two_stage_hq_video_only",
    }
    if trace_recorder.enabled:
        metadata["trace"] = trace_recorder.to_metadata()
    if audio_backend is not None:
        metadata["audio_backend"] = audio_backend

    return GeneratedVideo(
        frames=np.asarray(frames_uint8, dtype=np.uint8),
        fps=fps,
        seed=effective_generation_seed,
        backend="mlxr_ltx_two_stage_hq"
        if host._audio_enabled
        else "mlxr_ltx_two_stage_hq_video_only",
        conditioning_count=len(conditioning_inputs),
        prompt_signature=_prompt_signature(prompt_context.prompt_text),
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        metadata=metadata,
    )


def _predict_video_only(
    *,
    imports: _ReferenceImports,
    transformer: _VideoTransformer,
    latents: MLXArray,
    positions: MLXArray,
    text_embeddings: MLXArray,
    negative_text_embeddings: MLXArray | None,
    sigma_value: float,
    video_timesteps_mask: MLXArray,
    precomputed_rope: tuple[MLXArray, MLXArray],
    state: _LatentStateLike | None,
    video_cfg_scale: float,
    video_rescale_scale: float,
) -> MLXArray:
    latents_dtype = latents.dtype
    batch_size, channels, frames, latent_h, latent_w = latents.shape
    sigma = mx.array(float(sigma_value), dtype=latents_dtype)
    flat_latents = mx.transpose(
        mx.reshape(latents, (batch_size, channels, -1)), (0, 2, 1)
    )
    modality = _PatchedModality(
        latent=flat_latents,
        sigma=mx.full((batch_size,), float(sigma_value), dtype=latents_dtype),
        timesteps=sigma * video_timesteps_mask,
        positions=positions,
        context=text_embeddings,
        context_mask=None,
        enabled=True,
        positional_embeddings=precomputed_rope,
    )
    velocity, _ = transformer(video=modality, audio=None)
    if velocity is None:
        raise RuntimeError(
            "LTX HQ transformer returned empty video velocity for an enabled video-only step"
        )
    velocity = mx.reshape(
        mx.transpose(velocity, (0, 2, 1)),
        (batch_size, channels, frames, latent_h, latent_w),
    )
    denoised = imports.to_denoised(latents, velocity, sigma)
    negative_denoised: MLXArray | None = None
    if negative_text_embeddings is not None:
        negative_modality = _PatchedModality(
            latent=flat_latents,
            sigma=mx.full((batch_size,), float(sigma_value), dtype=latents_dtype),
            timesteps=sigma * video_timesteps_mask,
            positions=positions,
            context=negative_text_embeddings,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_rope,
        )
        negative_velocity, _ = transformer(video=negative_modality, audio=None)
        if negative_velocity is None:
            raise RuntimeError(
                "LTX HQ transformer returned empty negative video velocity for an enabled video-only step"
            )
        negative_velocity = mx.reshape(
            mx.transpose(negative_velocity, (0, 2, 1)),
            (batch_size, channels, frames, latent_h, latent_w),
        )
        negative_denoised = imports.to_denoised(latents, negative_velocity, sigma)
    denoised = _guided_prediction(
        denoised,
        negative_denoised,
        scale=video_cfg_scale if negative_text_embeddings is not None else 1.0,
        rescale_scale=video_rescale_scale
        if negative_text_embeddings is not None
        else 0.0,
    )
    if state is not None:
        denoised = imports.apply_denoise_mask(
            denoised, state.clean_latent, state.denoise_mask
        )
    return denoised.astype(latents_dtype)


def _predict_audio_video(
    *,
    imports: _ReferenceImports,
    transformer: _AudioVideoTransformer,
    latents: MLXArray,
    positions: MLXArray,
    text_embeddings: MLXArray,
    audio_latents: MLXArray,
    audio_positions: MLXArray,
    audio_embeddings: MLXArray,
    negative_text_embeddings: MLXArray | None,
    negative_audio_embeddings: MLXArray | None,
    sigma_value: float,
    video_timesteps_mask: MLXArray,
    audio_timesteps_mask: MLXArray,
    precomputed_rope: tuple[MLXArray, MLXArray],
    precomputed_audio_rope: tuple[MLXArray, MLXArray],
    state: _LatentStateLike | None,
    audio_state: _LatentStateLike | None,
    video_cfg_scale: float,
    audio_cfg_scale: float,
    video_rescale_scale: float,
    audio_rescale_scale: float,
) -> tuple[MLXArray, MLXArray]:
    latents_dtype = latents.dtype
    batch_size, channels, frames, latent_h, latent_w = latents.shape
    audio_batch, audio_channels, audio_frames, audio_bins = audio_latents.shape
    sigma = mx.array(float(sigma_value), dtype=latents_dtype)
    flat_latents = mx.transpose(
        mx.reshape(latents, (batch_size, channels, -1)), (0, 2, 1)
    )
    modality = _PatchedModality(
        latent=flat_latents,
        sigma=mx.full((batch_size,), float(sigma_value), dtype=latents_dtype),
        timesteps=sigma * video_timesteps_mask,
        positions=positions,
        context=text_embeddings,
        context_mask=None,
        enabled=True,
        positional_embeddings=precomputed_rope,
    )
    audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
    audio_flat = mx.reshape(
        audio_flat, (audio_batch, audio_frames, audio_channels * audio_bins)
    )
    audio_modality = _PatchedModality(
        latent=audio_flat,
        sigma=mx.full((audio_batch,), float(sigma_value), dtype=latents_dtype),
        timesteps=sigma * audio_timesteps_mask,
        positions=audio_positions,
        context=audio_embeddings,
        context_mask=None,
        enabled=True,
        positional_embeddings=precomputed_audio_rope,
    )
    velocity, audio_velocity = transformer(video=modality, audio=audio_modality)
    if velocity is None or audio_velocity is None:
        raise RuntimeError(
            "LTX HQ transformer returned empty video/audio velocities for an enabled AV step"
        )
    velocity = mx.reshape(
        mx.transpose(velocity, (0, 2, 1)),
        (batch_size, channels, frames, latent_h, latent_w),
    )
    denoised = imports.to_denoised(latents, velocity, sigma)
    audio_velocity = mx.reshape(
        audio_velocity,
        (audio_batch, audio_frames, audio_channels, audio_bins),
    )
    audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
    audio_denoised = imports.to_denoised(audio_latents, audio_velocity, sigma)
    negative_denoised: MLXArray | None = None
    negative_audio_denoised: MLXArray | None = None
    if negative_text_embeddings is not None and negative_audio_embeddings is not None:
        negative_modality = _PatchedModality(
            latent=flat_latents,
            sigma=mx.full((batch_size,), float(sigma_value), dtype=latents_dtype),
            timesteps=sigma * video_timesteps_mask,
            positions=positions,
            context=negative_text_embeddings,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_rope,
        )
        negative_audio_modality = _PatchedModality(
            latent=audio_flat,
            sigma=mx.full((audio_batch,), float(sigma_value), dtype=latents_dtype),
            timesteps=sigma * audio_timesteps_mask,
            positions=audio_positions,
            context=negative_audio_embeddings,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_audio_rope,
        )
        negative_velocity, negative_audio_velocity = transformer(
            video=negative_modality,
            audio=negative_audio_modality,
        )
        if negative_velocity is None or negative_audio_velocity is None:
            raise RuntimeError(
                "LTX HQ transformer returned empty negative video/audio velocities for an enabled AV step"
            )
        negative_velocity = mx.reshape(
            mx.transpose(negative_velocity, (0, 2, 1)),
            (batch_size, channels, frames, latent_h, latent_w),
        )
        negative_denoised = imports.to_denoised(latents, negative_velocity, sigma)
        negative_audio_velocity = mx.reshape(
            negative_audio_velocity,
            (audio_batch, audio_frames, audio_channels, audio_bins),
        )
        negative_audio_velocity = mx.transpose(negative_audio_velocity, (0, 2, 1, 3))
        negative_audio_denoised = imports.to_denoised(
            audio_latents,
            negative_audio_velocity,
            sigma,
        )
    denoised = _guided_prediction(
        denoised,
        negative_denoised,
        scale=video_cfg_scale
        if negative_text_embeddings is not None
        and negative_audio_embeddings is not None
        else 1.0,
        rescale_scale=video_rescale_scale
        if negative_text_embeddings is not None
        and negative_audio_embeddings is not None
        else 0.0,
    )
    audio_denoised = _guided_prediction(
        audio_denoised,
        negative_audio_denoised,
        scale=audio_cfg_scale
        if negative_text_embeddings is not None
        and negative_audio_embeddings is not None
        else 1.0,
        rescale_scale=audio_rescale_scale
        if negative_text_embeddings is not None
        and negative_audio_embeddings is not None
        else 0.0,
    )
    if state is not None:
        denoised = imports.apply_denoise_mask(
            denoised, state.clean_latent, state.denoise_mask
        )
    if audio_state is not None:
        audio_denoised = imports.apply_denoise_mask(
            audio_denoised,
            audio_state.clean_latent,
            audio_state.denoise_mask,
        )
    return denoised.astype(latents_dtype), audio_denoised.astype(latents_dtype)


def _res2s_video_only_loop(
    *,
    imports: _ReferenceImports,
    transformer: _VideoTransformer,
    latents: MLXArray,
    positions: MLXArray,
    text_embeddings: MLXArray,
    negative_text_embeddings: MLXArray | None,
    sigmas: tuple[float, ...],
    state: _LatentStateLike | None,
    runtime_config: _RuntimeModelConfig,
    video_cfg_scale: float,
    video_rescale_scale: float,
) -> MLXArray:
    latents_dtype = latents.dtype
    batch_size, _, frames, latent_h, latent_w = latents.shape
    num_tokens = int(frames * latent_h * latent_w)
    precomputed_rope = imports.precompute_freqs_cis(
        positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=runtime_config.double_precision_rope,
    )
    if state is not None:
        denoise_mask = mx.reshape(state.denoise_mask, (batch_size, 1, frames, 1, 1))
        denoise_mask = mx.broadcast_to(
            denoise_mask, (batch_size, 1, frames, latent_h, latent_w)
        )
        video_timesteps_mask = mx.reshape(
            denoise_mask, (batch_size, num_tokens)
        ).astype(latents_dtype)
    else:
        video_timesteps_mask = mx.ones((batch_size, num_tokens), dtype=latents_dtype)

    for sigma_value, sigma_next_value in zip(sigmas[:-1], sigmas[1:]):
        denoised = _predict_video_only(
            imports=imports,
            transformer=transformer,
            latents=latents,
            positions=positions,
            text_embeddings=text_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            sigma_value=float(sigma_value),
            video_timesteps_mask=video_timesteps_mask,
            precomputed_rope=precomputed_rope,
            state=state,
            video_cfg_scale=video_cfg_scale,
            video_rescale_scale=video_rescale_scale,
        )
        if float(sigma_next_value) == 0.0:
            latents = denoised.astype(latents_dtype)
            mx.eval(latents)
            break

        latents = _res2s_step_video_only(
            imports=imports,
            transformer=transformer,
            latents=latents,
            positions=positions,
            text_embeddings=text_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            sigma_value=float(sigma_value),
            sigma_next_value=float(sigma_next_value),
            video_timesteps_mask=video_timesteps_mask,
            precomputed_rope=precomputed_rope,
            state=state,
            video_cfg_scale=video_cfg_scale,
            video_rescale_scale=video_rescale_scale,
        )
        mx.eval(latents)
    return latents


def _res2s_audio_video_loop(
    *,
    imports: _ReferenceImports,
    transformer: _AudioVideoTransformer,
    latents: MLXArray,
    positions: MLXArray,
    text_embeddings: MLXArray,
    audio_latents: MLXArray,
    audio_positions: MLXArray,
    audio_embeddings: MLXArray,
    negative_text_embeddings: MLXArray | None,
    negative_audio_embeddings: MLXArray | None,
    sigmas: tuple[float, ...],
    state: _LatentStateLike | None,
    audio_state: _LatentStateLike | None,
    runtime_config: _RuntimeModelConfig,
    video_cfg_scale: float,
    audio_cfg_scale: float,
    video_rescale_scale: float,
    audio_rescale_scale: float,
) -> tuple[MLXArray, MLXArray]:
    latents_dtype = latents.dtype
    batch_size, _, frames, latent_h, latent_w = latents.shape
    num_tokens = int(frames * latent_h * latent_w)
    audio_batch, _, audio_frames, _ = audio_latents.shape
    precomputed_rope = imports.precompute_freqs_cis(
        positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=runtime_config.double_precision_rope,
    )
    precomputed_audio_rope = imports.precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=runtime_config.double_precision_rope,
    )
    if state is not None:
        denoise_mask = mx.reshape(state.denoise_mask, (batch_size, 1, frames, 1, 1))
        denoise_mask = mx.broadcast_to(
            denoise_mask, (batch_size, 1, frames, latent_h, latent_w)
        )
        video_timesteps_mask = mx.reshape(
            denoise_mask, (batch_size, num_tokens)
        ).astype(latents_dtype)
    else:
        video_timesteps_mask = mx.ones((batch_size, num_tokens), dtype=latents_dtype)
    if audio_state is not None:
        audio_timesteps_mask = mx.reshape(
            audio_state.denoise_mask, (audio_batch, audio_frames)
        ).astype(latents_dtype)
    else:
        audio_timesteps_mask = mx.ones((audio_batch, audio_frames), dtype=latents_dtype)

    for sigma_value, sigma_next_value in zip(sigmas[:-1], sigmas[1:]):
        denoised, audio_denoised = _predict_audio_video(
            imports=imports,
            transformer=transformer,
            latents=latents,
            positions=positions,
            text_embeddings=text_embeddings,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            negative_audio_embeddings=negative_audio_embeddings,
            sigma_value=float(sigma_value),
            video_timesteps_mask=video_timesteps_mask,
            audio_timesteps_mask=audio_timesteps_mask,
            precomputed_rope=precomputed_rope,
            precomputed_audio_rope=precomputed_audio_rope,
            state=state,
            audio_state=audio_state,
            video_cfg_scale=video_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            video_rescale_scale=video_rescale_scale,
            audio_rescale_scale=audio_rescale_scale,
        )
        if float(sigma_next_value) == 0.0:
            latents = denoised.astype(latents_dtype)
            audio_latents = audio_denoised.astype(latents_dtype)
            mx.eval(latents, audio_latents)
            break

        latents, audio_latents = _res2s_step_audio_video(
            imports=imports,
            transformer=transformer,
            latents=latents,
            positions=positions,
            text_embeddings=text_embeddings,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            negative_audio_embeddings=negative_audio_embeddings,
            sigma_value=float(sigma_value),
            sigma_next_value=float(sigma_next_value),
            video_timesteps_mask=video_timesteps_mask,
            audio_timesteps_mask=audio_timesteps_mask,
            precomputed_rope=precomputed_rope,
            precomputed_audio_rope=precomputed_audio_rope,
            state=state,
            audio_state=audio_state,
            video_cfg_scale=video_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            video_rescale_scale=video_rescale_scale,
            audio_rescale_scale=audio_rescale_scale,
        )
        mx.eval(latents, audio_latents)
    return latents, audio_latents


def _res2s_step_video_only(
    *,
    imports: _ReferenceImports,
    transformer: _VideoTransformer,
    latents: MLXArray,
    positions: MLXArray,
    text_embeddings: MLXArray,
    negative_text_embeddings: MLXArray | None,
    sigma_value: float,
    sigma_next_value: float,
    video_timesteps_mask: MLXArray,
    precomputed_rope: tuple[MLXArray, MLXArray],
    state: _LatentStateLike | None,
    video_cfg_scale: float,
    video_rescale_scale: float,
) -> MLXArray:
    denoised = _predict_video_only(
        imports=imports,
        transformer=transformer,
        latents=latents,
        positions=positions,
        text_embeddings=text_embeddings,
        negative_text_embeddings=negative_text_embeddings,
        sigma_value=sigma_value,
        video_timesteps_mask=video_timesteps_mask,
        precomputed_rope=precomputed_rope,
        state=state,
        video_cfg_scale=video_cfg_scale,
        video_rescale_scale=video_rescale_scale,
    )
    h = -math.log(sigma_next_value / sigma_value)
    a21, b1, b2 = _get_res2s_coefficients(h)
    sub_sigma = math.sqrt(sigma_value * sigma_next_value)
    x_anchor = latents.astype(mx.float32)
    eps1 = denoised.astype(mx.float32) - x_anchor
    x_mid = x_anchor + float(h * a21) * eps1
    x_mid = _inject_sde_noise(
        imports=imports,
        state=state,
        sample=x_anchor,
        denoised_sample=x_mid,
        sigma_value=sigma_value,
        sigma_next_value=sub_sigma,
    )
    if h < 0.5 and sigma_value > 0.03:
        for _ in range(_RES2S_BONGMATH_MAX_ITER):
            x_anchor = x_mid - float(h * a21) * eps1
            eps1 = denoised.astype(mx.float32) - x_anchor
    denoised2 = _predict_video_only(
        imports=imports,
        transformer=transformer,
        latents=x_mid.astype(latents.dtype),
        positions=positions,
        text_embeddings=text_embeddings,
        negative_text_embeddings=negative_text_embeddings,
        sigma_value=sub_sigma,
        video_timesteps_mask=video_timesteps_mask,
        precomputed_rope=precomputed_rope,
        state=state,
        video_cfg_scale=video_cfg_scale,
        video_rescale_scale=video_rescale_scale,
    )
    eps2 = denoised2.astype(mx.float32) - x_anchor
    x_next = x_anchor + float(h) * (float(b1) * eps1 + float(b2) * eps2)
    x_next = _inject_sde_noise(
        imports=imports,
        state=state,
        sample=x_anchor,
        denoised_sample=x_next,
        sigma_value=sigma_value,
        sigma_next_value=sigma_next_value,
    )
    return x_next.astype(latents.dtype)


def _res2s_step_audio_video(
    *,
    imports: _ReferenceImports,
    transformer: _AudioVideoTransformer,
    latents: MLXArray,
    positions: MLXArray,
    text_embeddings: MLXArray,
    audio_latents: MLXArray,
    audio_positions: MLXArray,
    audio_embeddings: MLXArray,
    negative_text_embeddings: MLXArray | None,
    negative_audio_embeddings: MLXArray | None,
    sigma_value: float,
    sigma_next_value: float,
    video_timesteps_mask: MLXArray,
    audio_timesteps_mask: MLXArray,
    precomputed_rope: tuple[MLXArray, MLXArray],
    precomputed_audio_rope: tuple[MLXArray, MLXArray],
    state: _LatentStateLike | None,
    audio_state: _LatentStateLike | None,
    video_cfg_scale: float,
    audio_cfg_scale: float,
    video_rescale_scale: float,
    audio_rescale_scale: float,
) -> tuple[MLXArray, MLXArray]:
    denoised_video, denoised_audio = _predict_audio_video(
        imports=imports,
        transformer=transformer,
        latents=latents,
        positions=positions,
        text_embeddings=text_embeddings,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_embeddings=audio_embeddings,
        negative_text_embeddings=negative_text_embeddings,
        negative_audio_embeddings=negative_audio_embeddings,
        sigma_value=sigma_value,
        video_timesteps_mask=video_timesteps_mask,
        audio_timesteps_mask=audio_timesteps_mask,
        precomputed_rope=precomputed_rope,
        precomputed_audio_rope=precomputed_audio_rope,
        state=state,
        audio_state=audio_state,
        video_cfg_scale=video_cfg_scale,
        audio_cfg_scale=audio_cfg_scale,
        video_rescale_scale=video_rescale_scale,
        audio_rescale_scale=audio_rescale_scale,
    )
    h = -math.log(sigma_next_value / sigma_value)
    a21, b1, b2 = _get_res2s_coefficients(h)
    sub_sigma = math.sqrt(sigma_value * sigma_next_value)
    x_anchor_video = latents.astype(mx.float32)
    x_anchor_audio = audio_latents.astype(mx.float32)
    eps1_video = denoised_video.astype(mx.float32) - x_anchor_video
    eps1_audio = denoised_audio.astype(mx.float32) - x_anchor_audio
    x_mid_video = x_anchor_video + float(h * a21) * eps1_video
    x_mid_audio = x_anchor_audio + float(h * a21) * eps1_audio
    x_mid_video = _inject_sde_noise(
        imports=imports,
        state=state,
        sample=x_anchor_video,
        denoised_sample=x_mid_video,
        sigma_value=sigma_value,
        sigma_next_value=sub_sigma,
    )
    x_mid_audio = _inject_sde_noise(
        imports=imports,
        state=audio_state,
        sample=x_anchor_audio,
        denoised_sample=x_mid_audio,
        sigma_value=sigma_value,
        sigma_next_value=sub_sigma,
    )
    if h < 0.5 and sigma_value > 0.03:
        for _ in range(_RES2S_BONGMATH_MAX_ITER):
            x_anchor_video = x_mid_video - float(h * a21) * eps1_video
            x_anchor_audio = x_mid_audio - float(h * a21) * eps1_audio
            eps1_video = denoised_video.astype(mx.float32) - x_anchor_video
            eps1_audio = denoised_audio.astype(mx.float32) - x_anchor_audio
    denoised_video_2, denoised_audio_2 = _predict_audio_video(
        imports=imports,
        transformer=transformer,
        latents=x_mid_video.astype(latents.dtype),
        positions=positions,
        text_embeddings=text_embeddings,
        audio_latents=x_mid_audio.astype(audio_latents.dtype),
        audio_positions=audio_positions,
        audio_embeddings=audio_embeddings,
        negative_text_embeddings=negative_text_embeddings,
        negative_audio_embeddings=negative_audio_embeddings,
        sigma_value=sub_sigma,
        video_timesteps_mask=video_timesteps_mask,
        audio_timesteps_mask=audio_timesteps_mask,
        precomputed_rope=precomputed_rope,
        precomputed_audio_rope=precomputed_audio_rope,
        state=state,
        audio_state=audio_state,
        video_cfg_scale=video_cfg_scale,
        audio_cfg_scale=audio_cfg_scale,
        video_rescale_scale=video_rescale_scale,
        audio_rescale_scale=audio_rescale_scale,
    )
    eps2_video = denoised_video_2.astype(mx.float32) - x_anchor_video
    eps2_audio = denoised_audio_2.astype(mx.float32) - x_anchor_audio
    x_next_video = x_anchor_video + float(h) * (
        float(b1) * eps1_video + float(b2) * eps2_video
    )
    x_next_audio = x_anchor_audio + float(h) * (
        float(b1) * eps1_audio + float(b2) * eps2_audio
    )
    x_next_video = _inject_sde_noise(
        imports=imports,
        state=state,
        sample=x_anchor_video,
        denoised_sample=x_next_video,
        sigma_value=sigma_value,
        sigma_next_value=sigma_next_value,
    )
    x_next_audio = _inject_sde_noise(
        imports=imports,
        state=audio_state,
        sample=x_anchor_audio,
        denoised_sample=x_next_audio,
        sigma_value=sigma_value,
        sigma_next_value=sigma_next_value,
    )
    return x_next_video.astype(latents.dtype), x_next_audio.astype(audio_latents.dtype)


def _inject_sde_noise(
    *,
    imports: _ReferenceImports,
    state: _LatentStateLike | None,
    sample: MLXArray,
    denoised_sample: MLXArray,
    sigma_value: float,
    sigma_next_value: float,
) -> MLXArray:
    if sigma_next_value <= 0.0:
        return denoised_sample.astype(sample.dtype)
    alpha_ratio, sigma_down, sigma_up = _get_sde_coefficients(
        sigma_next_value=sigma_next_value,
        sigma_up=sigma_next_value * 0.5,
    )
    if sigma_up == 0.0:
        return denoised_sample.astype(sample.dtype)
    sample_f32 = sample.astype(mx.float32)
    denoised_f32 = denoised_sample.astype(mx.float32)
    denominator = max(sigma_value - sigma_next_value, 1e-6)
    eps_next = (sample_f32 - denoised_f32) / denominator
    denoised_next = sample_f32 - sigma_value * eps_next
    x_next = alpha_ratio * (
        denoised_next + sigma_down * eps_next
    ) + sigma_up * _new_noise_like(sample)
    if state is not None:
        x_next = imports.apply_denoise_mask(
            x_next.astype(sample.dtype), state.clean_latent, state.denoise_mask
        )
    return x_next.astype(sample.dtype)


def _new_noise_like(sample: MLXArray) -> MLXArray:
    noise = mx.random.normal(sample.shape).astype(mx.float32)
    centered = noise - mx.mean(noise)
    normalized = centered / mx.maximum(
        mx.std(centered),
        mx.array(1e-6, dtype=centered.dtype),
    )
    return _channelwise_normalize(normalized).astype(sample.dtype)


def _channelwise_normalize(x: MLXArray) -> MLXArray:
    mean = mx.mean(x, axis=(-2, -1), keepdims=True)
    std = mx.std(x, axis=(-2, -1), keepdims=True)
    return (x - mean) / mx.maximum(std, mx.array(1e-6, dtype=x.dtype))


def _get_sde_coefficients(
    *,
    sigma_next_value: float,
    sigma_up: float | None = None,
    sigma_down: float | None = None,
    sigma_max: float | None = None,
) -> tuple[float, float, float]:
    if sigma_down is not None:
        alpha_ratio = (1.0 - sigma_next_value) / max(1.0 - sigma_down, 1e-6)
        sigma_up_value = math.sqrt(
            max(
                sigma_next_value**2 - (sigma_down**2) * (alpha_ratio**2),
                0.0,
            )
        )
        return alpha_ratio, sigma_down, sigma_up_value
    sigma_up_value = (
        0.0 if sigma_up is None else min(sigma_up, sigma_next_value * 0.9999)
    )
    sigma_signal = (1.0 if sigma_max is None else sigma_max) - sigma_next_value
    sigma_residual = math.sqrt(max(sigma_next_value**2 - sigma_up_value**2, 0.0))
    alpha_ratio = sigma_signal + sigma_residual
    sigma_down_value = (
        sigma_residual / alpha_ratio if abs(alpha_ratio) > 1e-6 else sigma_next_value
    )
    return alpha_ratio, sigma_down_value, sigma_up_value


def _phi(order: int, negative_h: float) -> float:
    if abs(negative_h) < 1e-10:
        return 1.0 / math.factorial(order)
    remainder = sum(negative_h**k / math.factorial(k) for k in range(order))
    return (math.exp(negative_h) - remainder) / (negative_h**order)


def _get_res2s_coefficients(
    h: float, c2: float = _RES2S_C2
) -> tuple[float, float, float]:
    a21 = c2 * _phi(1, -h * c2)
    b2 = _phi(2, -h) / c2
    b1 = _phi(1, -h) - b2
    return a21, b1, b2
