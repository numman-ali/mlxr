from __future__ import annotations

import json
import math
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from mlxr.core.runtime import TraceRecorder, mlx_memory_snapshot
from PIL import Image

from ..family_options import effective_seed
from ..generation import (
    ConditioningInput,
    GeneratedVideo,
    LoraInput,
    VideoReferenceInput,
)
from ..prompt_encoding import PromptEncodingResult
from .conditioning import (
    _half_resolution_padded_shape,
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
from .lora import read_reference_downscale_factor
from .one_stage import (
    _looks_like_distilled_checkpoint,
    _require_audio_latents,
    _require_audio_video_transformer,
    _require_video_transformer,
)
from .outputs import _decode_to_uint8_frames
from .runtime_helpers import _ensure_transformer_with_loras, _release_transformers
from .sampling import (
    _AUDIO_CFG_SCALE,
    _GUIDANCE_RESCALE_SCALE,
    _VIDEO_CFG_SCALE,
    _assert_prompt_runtime_contract,
    _denoise_distilled_audio_video,
    _denoise_distilled_video_only,
    _guided_prediction,
    _sync_optional_arrays,
)
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
    _VAEEncoder,
    _VideoDecoderLike,
    _VideoTransformer,
)


class _ICLoraHost(_RuntimeHelperHost, Protocol):
    control_variant: str | None
    conditioning_attention_strength: float

    def _imports(self) -> _ReferenceImports: ...
    def _ensure_transformer(
        self,
        imports: _ReferenceImports,
        runtime_config: _RuntimeModelConfig,
        prompt_context: PromptEncodingResult,
    ) -> _AudioVideoTransformer | _VideoTransformer: ...
    def _release_checkpoint_reader(self) -> None: ...
    def _ensure_vae_statistics(self) -> tuple[MLXArray, MLXArray]: ...
    def _ensure_upsampler(self, imports: _ReferenceImports) -> _UpsamplerLike: ...
    def _ensure_vae_decoder(self, imports: _ReferenceImports) -> _VideoDecoderLike: ...
    def _ensure_vae_encoder(self, imports: _ReferenceImports) -> _VAEEncoder: ...
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


@dataclass(frozen=True, slots=True)
class _ProbedVideo:
    width: int
    height: int
    fps: float
    num_frames: int


def generate_ic_lora(
    host: _ICLoraHost,
    *,
    prompt_context: PromptEncodingResult,
    conditioning_inputs: tuple[ConditioningInput, ...],
    video_inputs: tuple[VideoReferenceInput, ...],
    lora_inputs: tuple[LoraInput, ...],
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int | None,
    num_inference_steps: int | None,
    conditioning_attention_strength_override: float | None = None,
    guidance_scale: float | None,
) -> GeneratedVideo:
    if not _looks_like_distilled_checkpoint(host.checkpoint_path):
        raise ValueError(
            "LTX video.condition.video currently requires the distilled checkpoint"
        )
    if host.spatial_upsampler_path is None:
        raise ValueError(
            "LTX video.condition.video requires the x2 spatial upsampler component"
        )
    if len(video_inputs) != 1:
        raise ValueError(
            "LTX video.condition.video currently requires exactly one reference video"
        )
    if len(lora_inputs) != 1:
        raise ValueError(
            "LTX video.condition.video currently requires exactly one IC-LoRA input"
        )

    imports = host._imports()
    runtime_config = _runtime_model_config(host.checkpoint_path)
    _assert_prompt_runtime_contract(
        prompt_context, runtime_config, audio_required=host._audio_enabled
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
    steps = 30 if num_inference_steps is None else num_inference_steps
    if steps <= 0:
        raise ValueError("LTX video.condition.video requires num_inference_steps > 0")
    video_cfg_scale = _VIDEO_CFG_SCALE if guidance_scale is None else guidance_scale
    audio_cfg_scale = _AUDIO_CFG_SCALE if guidance_scale is None else guidance_scale
    conditioning_attention_strength = float(
        conditioning_attention_strength_override
        if conditioning_attention_strength_override is not None
        else host.conditioning_attention_strength
        if host.conditioning_attention_strength is not None
        else 1.0
    )
    if not 0.0 <= conditioning_attention_strength <= 1.0:
        raise ValueError(
            "LTX conditioning_attention_strength must be between 0.0 and 1.0"
        )

    trace_recorder = TraceRecorder(enabled=_debug_trace_enabled())
    trace_sync = _debug_trace_sync_enabled()

    stage1_padded_shape = _half_resolution_padded_shape(
        padded_shape=padded_shape,
        width=width,
        height=height,
    )
    reference_downscale_factor = _reference_downscale_factor(lora_inputs)
    if (
        stage1_padded_shape.internal_height % reference_downscale_factor != 0
        or stage1_padded_shape.internal_width % reference_downscale_factor != 0
    ):
        raise ValueError(
            "LTX reference-video conditioning requires stage-1 padded dimensions "
            "to be divisible by the IC-LoRA reference_downscale_factor"
        )

    with trace_recorder.span("ltx.prepare_conditionings", snapshot=mlx_memory_snapshot):
        conditioning_plan = host._prepare_conditionings(
            imports=imports,
            conditioning_inputs=conditioning_inputs,
            num_frames=num_frames,
            latent_frames=latent_frames,
            padded_shape=padded_shape,
            model_dtype=model_dtype,
            replace_first_frame_latent=True,
        )
    with trace_recorder.span(
        "ltx.encode_reference_video", snapshot=mlx_memory_snapshot
    ):
        vae_encoder = host._ensure_vae_encoder(imports)
        reference_latent = _encode_reference_video_latent(
            imports=imports,
            vae_encoder=vae_encoder,
            video_path=video_inputs[0].payload_path,
            width=stage1_padded_shape.internal_width // reference_downscale_factor,
            height=stage1_padded_shape.internal_height // reference_downscale_factor,
            frame_cap=num_frames,
            dtype=model_dtype,
        )
    vae_encoder = None
    host._vae_encoder = None
    mx.clear_cache()
    host._release_checkpoint_reader()

    stage1_lora_paths = tuple(item.payload_path for item in lora_inputs)
    stage1_lora_scales = tuple(float(item.strength) for item in lora_inputs)
    with trace_recorder.span("ltx.ensure_transformer", snapshot=mlx_memory_snapshot):
        stage1_transformer = _ensure_transformer_with_loras(
            host,
            imports=imports,
            runtime_config=runtime_config,
            prompt_context=prompt_context,
            lora_paths=stage1_lora_paths,
            lora_scales=stage1_lora_scales,
        )

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
                "LTX IC-LoRA audio-video path requires audio context and latents"
            )
        latents, audio_latents = _denoise_stage_1_audio_video_with_reference(
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
            reference_latent=reference_latent,
            reference_strength=video_inputs[0].strength,
            reference_downscale_factor=reference_downscale_factor,
            conditioning_attention_strength=conditioning_attention_strength,
            runtime_config=runtime_config,
            video_cfg_scale=video_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
            fps=float(fps),
        )
        mx.eval(latents, audio_latents)
    else:
        latents = _denoise_stage_1_video_only_with_reference(
            imports=imports,
            transformer=_require_video_transformer(stage1_transformer),
            latents=latents,
            positions=stage1_positions,
            text_embeddings=video_context,
            negative_text_embeddings=negative_video_context,
            sigmas=stage1_sigmas,
            state=stage1_state,
            reference_latent=reference_latent,
            reference_strength=video_inputs[0].strength,
            reference_downscale_factor=reference_downscale_factor,
            conditioning_attention_strength=conditioning_attention_strength,
            runtime_config=runtime_config,
            video_cfg_scale=video_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
            fps=float(fps),
        )
        mx.eval(latents)
    timings_ms["stage1_duration_ms"] = _elapsed_ms(stage1_started)
    # IC-LoRA stage 1 uses a reference-specialized transformer; release it before
    # stage 2 falls back to the base distilled model.
    stage1_transformer = None
    stage1_state = None
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
    stage2_transformer = host._ensure_transformer(
        imports, runtime_config, prompt_context
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
            raise RuntimeError("LTX IC-LoRA stage 2 requires audio state")
        audio_latents = _require_audio_latents(audio_latents)
        latents, audio_latents = _denoise_distilled_audio_video(
            imports=imports,
            transformer=_require_audio_video_transformer(stage2_transformer),
            latents=latents,
            positions=stage2_positions,
            text_embeddings=video_context,
            audio_latents=audio_latents,
            audio_positions=stage2_audio_positions,
            audio_embeddings=audio_context,
            negative_text_embeddings=None,
            negative_audio_embeddings=None,
            sigmas=imports.stage_2_sigmas,
            state=stage2_state,
            runtime_config=runtime_config,
            freeze_audio=False,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
        )
        mx.eval(latents, audio_latents)
    else:
        latents = _denoise_distilled_video_only(
            imports=imports,
            transformer=_require_video_transformer(stage2_transformer),
            latents=latents,
            positions=stage2_positions,
            text_embeddings=video_context,
            negative_text_embeddings=None,
            sigmas=imports.stage_2_sigmas,
            state=stage2_state,
            runtime_config=runtime_config,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
        )
        mx.eval(latents)
    timings_ms["stage2_duration_ms"] = _elapsed_ms(stage2_started)
    stage2_transformer = None
    stage2_state = None
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

    conditioning_count = len(conditioning_inputs) + len(video_inputs)
    metadata: dict[str, object] = {
        "pipeline_kind": "ic_lora" if host._audio_enabled else "ic_lora_video_only",
        "output_width": width,
        "output_height": height,
        "output_frames": num_frames,
        "padded_width": padded_shape.internal_width,
        "padded_height": padded_shape.internal_height,
        "latent_frames": latent_frames,
        "conditioning_count": conditioning_count,
        "image_conditioning_count": len(conditioning_inputs),
        "video_reference_count": len(video_inputs),
        "lora_count": len(lora_inputs),
        "control_variant": host.control_variant or "ic_lora",
        "conditioning_attention_strength": conditioning_attention_strength,
        "reference_downscale_factor": reference_downscale_factor,
        "reference_token_count": int(
            reference_latent.shape[2]
            * reference_latent.shape[3]
            * reference_latent.shape[4]
        ),
        "reference_latent_frames": int(reference_latent.shape[2]),
        "audio_conditioned": False,
        "tiling_mode": tiling_mode,
        "timings_ms": {
            **timings_ms,
            "decode_duration_ms": decode_duration_ms,
        },
        "backend": "mlxr_ltx_ic_lora"
        if host._audio_enabled
        else "mlxr_ltx_ic_lora_video_only",
    }
    if trace_recorder.enabled:
        metadata["trace"] = trace_recorder.to_metadata()
    if audio_backend is not None:
        metadata["audio_backend"] = audio_backend

    return GeneratedVideo(
        frames=np.asarray(frames_uint8, dtype=np.uint8),
        fps=fps,
        seed=effective_generation_seed,
        backend="mlxr_ltx_ic_lora"
        if host._audio_enabled
        else "mlxr_ltx_ic_lora_video_only",
        conditioning_count=conditioning_count,
        prompt_signature=_prompt_signature(prompt_context.prompt_text),
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        metadata=metadata,
    )


def _denoise_stage_1_video_only_with_reference(
    *,
    imports: _ReferenceImports,
    transformer: _VideoTransformer,
    latents: MLXArray,
    positions: MLXArray,
    text_embeddings: MLXArray,
    negative_text_embeddings: MLXArray | None,
    sigmas: tuple[float, ...],
    state: _LatentStateLike | None,
    reference_latent: MLXArray,
    reference_strength: float,
    reference_downscale_factor: int,
    conditioning_attention_strength: float,
    runtime_config: _RuntimeModelConfig,
    video_cfg_scale: float,
    audio_cfg_scale: float,
    fps: float,
    trace_recorder: TraceRecorder | None,
    trace_sync: bool,
) -> MLXArray:
    del audio_cfg_scale
    target_state = _target_token_state(
        latents=latents,
        state=state,
    )
    reference_tokens = _flatten_video_tokens(reference_latent)
    reference_positions = _reference_positions(
        imports=imports,
        reference_latent=reference_latent,
        fps=fps,
        downscale_factor=reference_downscale_factor,
    )
    reference_mask = mx.full(
        (1, int(reference_tokens.shape[1])),
        1.0 - float(reference_strength),
        dtype=latents.dtype,
    )
    full_positions = mx.concatenate([positions, reference_positions], axis=2)
    full_clean = mx.concatenate(
        [target_state.clean_tokens, reference_tokens.astype(latents.dtype)], axis=1
    )
    full_mask = mx.concatenate([target_state.denoise_mask, reference_mask], axis=1)
    full_tokens = mx.concatenate(
        [target_state.current_tokens, reference_tokens.astype(latents.dtype)], axis=1
    )
    self_attention_mask = _reference_attention_mask(
        num_noisy_tokens=int(target_state.current_tokens.shape[1]),
        num_reference_tokens=int(reference_tokens.shape[1]),
        strength=conditioning_attention_strength,
        dtype=latents.dtype,
    )
    full_tokens = _denoise_video_tokens(
        imports=imports,
        transformer=transformer,
        full_tokens=full_tokens,
        positions=full_positions,
        text_embeddings=text_embeddings,
        negative_text_embeddings=negative_text_embeddings,
        sigmas=sigmas,
        clean_tokens=full_clean,
        denoise_mask=full_mask,
        self_attention_mask=self_attention_mask,
        runtime_config=runtime_config,
        video_cfg_scale=video_cfg_scale,
        trace_recorder=trace_recorder,
        trace_sync=trace_sync,
    )
    return _reshape_video_tokens(
        full_tokens[:, : int(target_state.current_tokens.shape[1]), :],
        channels=int(latents.shape[1]),
        frames=int(latents.shape[2]),
        height=int(latents.shape[3]),
        width=int(latents.shape[4]),
    )


def _denoise_stage_1_audio_video_with_reference(
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
    reference_latent: MLXArray,
    reference_strength: float,
    reference_downscale_factor: int,
    conditioning_attention_strength: float,
    runtime_config: _RuntimeModelConfig,
    video_cfg_scale: float,
    audio_cfg_scale: float,
    fps: float,
    trace_recorder: TraceRecorder | None,
    trace_sync: bool,
) -> tuple[MLXArray, MLXArray]:
    target_state = _target_token_state(
        latents=latents,
        state=state,
    )
    reference_tokens = _flatten_video_tokens(reference_latent)
    reference_positions = _reference_positions(
        imports=imports,
        reference_latent=reference_latent,
        fps=fps,
        downscale_factor=reference_downscale_factor,
    )
    reference_mask = mx.full(
        (1, int(reference_tokens.shape[1])),
        1.0 - float(reference_strength),
        dtype=latents.dtype,
    )
    full_positions = mx.concatenate([positions, reference_positions], axis=2)
    full_clean = mx.concatenate(
        [target_state.clean_tokens, reference_tokens.astype(latents.dtype)], axis=1
    )
    full_mask = mx.concatenate([target_state.denoise_mask, reference_mask], axis=1)
    full_tokens = mx.concatenate(
        [target_state.current_tokens, reference_tokens.astype(latents.dtype)], axis=1
    )
    self_attention_mask = _reference_attention_mask(
        num_noisy_tokens=int(target_state.current_tokens.shape[1]),
        num_reference_tokens=int(reference_tokens.shape[1]),
        strength=conditioning_attention_strength,
        dtype=latents.dtype,
    )
    full_tokens, audio_latents = _denoise_audio_video_tokens(
        imports=imports,
        transformer=transformer,
        full_tokens=full_tokens,
        positions=full_positions,
        text_embeddings=text_embeddings,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_embeddings=audio_embeddings,
        negative_text_embeddings=negative_text_embeddings,
        negative_audio_embeddings=negative_audio_embeddings,
        sigmas=sigmas,
        clean_tokens=full_clean,
        denoise_mask=full_mask,
        self_attention_mask=self_attention_mask,
        runtime_config=runtime_config,
        video_cfg_scale=video_cfg_scale,
        audio_cfg_scale=audio_cfg_scale,
        trace_recorder=trace_recorder,
        trace_sync=trace_sync,
    )
    video_latents = _reshape_video_tokens(
        full_tokens[:, : int(target_state.current_tokens.shape[1]), :],
        channels=int(latents.shape[1]),
        frames=int(latents.shape[2]),
        height=int(latents.shape[3]),
        width=int(latents.shape[4]),
    )
    return video_latents, audio_latents


@dataclass(frozen=True, slots=True)
class _TargetTokenState:
    current_tokens: MLXArray
    clean_tokens: MLXArray
    denoise_mask: MLXArray


def _target_token_state(
    *,
    latents: MLXArray,
    state: _LatentStateLike | None,
) -> _TargetTokenState:
    current_tokens = _flatten_video_tokens(latents)
    if state is None:
        clean_tokens = current_tokens
        denoise_mask = mx.ones(
            (int(current_tokens.shape[0]), int(current_tokens.shape[1])),
            dtype=latents.dtype,
        )
        return _TargetTokenState(
            current_tokens=current_tokens,
            clean_tokens=clean_tokens,
            denoise_mask=denoise_mask,
        )
    height = int(latents.shape[3])
    width = int(latents.shape[4])
    clean_tokens = _flatten_video_tokens(state.clean_latent)
    denoise_mask = _flatten_video_mask(
        state.denoise_mask,
        frames=int(latents.shape[2]),
        height=height,
        width=width,
        dtype=latents.dtype,
    )
    return _TargetTokenState(
        current_tokens=current_tokens,
        clean_tokens=clean_tokens,
        denoise_mask=denoise_mask,
    )


def _denoise_video_tokens(
    *,
    imports: _ReferenceImports,
    transformer: _VideoTransformer,
    full_tokens: MLXArray,
    positions: MLXArray,
    text_embeddings: MLXArray,
    negative_text_embeddings: MLXArray | None,
    sigmas: tuple[float, ...],
    clean_tokens: MLXArray,
    denoise_mask: MLXArray,
    self_attention_mask: MLXArray | None,
    runtime_config: _RuntimeModelConfig,
    video_cfg_scale: float,
    trace_recorder: TraceRecorder | None,
    trace_sync: bool,
) -> MLXArray:
    latents_dtype = full_tokens.dtype
    cfg_enabled = negative_text_embeddings is not None
    batch_size = int(full_tokens.shape[0])
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
    total_steps = max(len(sigmas) - 1, 0)
    for step_index, (sigma_value, sigma_next_value) in enumerate(
        zip(sigmas[:-1], sigmas[1:]),
        start=1,
    ):
        step_context = (
            trace_recorder.span(
                "ltx.denoise.step",
                attributes={
                    "step_index": step_index,
                    "total_steps": total_steps,
                    "sigma": round(float(sigma_value), 6),
                    "sigma_next": round(float(sigma_next_value), 6),
                    "cfg_enabled": cfg_enabled,
                    "audio_enabled": False,
                },
                snapshot=mlx_memory_snapshot,
            )
            if trace_recorder is not None
            else nullcontext()
        )
        with step_context:
            sigma = mx.array(float(sigma_value), dtype=latents_dtype)
            sigma_next = mx.array(float(sigma_next_value), dtype=latents_dtype)
            token_timesteps = sigma * denoise_mask
            modality = _PatchedModality(
                latent=full_tokens,
                sigma=mx.full((batch_size,), float(sigma_value), dtype=latents_dtype),
                timesteps=token_timesteps,
                positions=positions,
                context=text_embeddings,
                context_mask=None,
                attention_mask=self_attention_mask,
                enabled=True,
                positional_embeddings=precomputed_rope,
            )
            velocity: MLXArray | None = None
            conditioned_context = (
                trace_recorder.span(
                    "ltx.denoise.forward.conditioned",
                    attributes={"step_index": step_index, "audio_enabled": False},
                    sync=(lambda: _sync_optional_arrays(velocity))
                    if trace_sync
                    else None,
                )
                if trace_recorder is not None
                else nullcontext()
            )
            with conditioned_context:
                velocity, _ = transformer(video=modality, audio=None)
            if velocity is None:
                raise RuntimeError(
                    "LTX IC-LoRA transformer returned empty video velocity"
                )
            denoised = imports.to_denoised(full_tokens, velocity, sigma)
            negative_denoised: MLXArray | None = None
            if negative_text_embeddings is not None:
                negative_velocity: MLXArray | None = None
                negative_modality = _PatchedModality(
                    latent=full_tokens,
                    sigma=mx.full(
                        (batch_size,), float(sigma_value), dtype=latents_dtype
                    ),
                    timesteps=token_timesteps,
                    positions=positions,
                    context=negative_text_embeddings,
                    context_mask=None,
                    attention_mask=self_attention_mask,
                    enabled=True,
                    positional_embeddings=precomputed_rope,
                )
                negative_context = (
                    trace_recorder.span(
                        "ltx.denoise.forward.negative",
                        attributes={"step_index": step_index, "audio_enabled": False},
                        sync=(lambda: _sync_optional_arrays(negative_velocity))
                        if trace_sync
                        else None,
                    )
                    if trace_recorder is not None
                    else nullcontext()
                )
                with negative_context:
                    negative_velocity, _ = transformer(
                        video=negative_modality,
                        audio=None,
                    )
                if negative_velocity is None:
                    raise RuntimeError(
                        "LTX IC-LoRA transformer returned empty negative video velocity"
                    )
                negative_denoised = imports.to_denoised(
                    full_tokens, negative_velocity, sigma
                )
            denoised = _guided_prediction(
                denoised,
                negative_denoised,
                scale=video_cfg_scale if cfg_enabled else 1.0,
                rescale_scale=_GUIDANCE_RESCALE_SCALE if cfg_enabled else 0.0,
            )
            denoised = _apply_token_denoise_mask(
                denoised=denoised,
                clean_tokens=clean_tokens,
                denoise_mask=denoise_mask,
            )
            full_tokens = _next_sigma_tokens(
                noisy=full_tokens,
                denoised=denoised,
                sigma=sigma,
                sigma_next=sigma_next,
            )
            mx.eval(full_tokens)
    return full_tokens


def _denoise_audio_video_tokens(
    *,
    imports: _ReferenceImports,
    transformer: _AudioVideoTransformer,
    full_tokens: MLXArray,
    positions: MLXArray,
    text_embeddings: MLXArray,
    audio_latents: MLXArray,
    audio_positions: MLXArray,
    audio_embeddings: MLXArray,
    negative_text_embeddings: MLXArray | None,
    negative_audio_embeddings: MLXArray | None,
    sigmas: tuple[float, ...],
    clean_tokens: MLXArray,
    denoise_mask: MLXArray,
    self_attention_mask: MLXArray | None,
    runtime_config: _RuntimeModelConfig,
    video_cfg_scale: float,
    audio_cfg_scale: float,
    trace_recorder: TraceRecorder | None,
    trace_sync: bool,
) -> tuple[MLXArray, MLXArray]:
    latents_dtype = full_tokens.dtype
    cfg_enabled = (
        negative_text_embeddings is not None and negative_audio_embeddings is not None
    )
    batch_size = int(full_tokens.shape[0])
    audio_batch, audio_channels, audio_frames, audio_bins = audio_latents.shape
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
    audio_timesteps_mask = mx.ones((audio_batch, audio_frames), dtype=latents_dtype)
    total_steps = max(len(sigmas) - 1, 0)
    for step_index, (sigma_value, sigma_next_value) in enumerate(
        zip(sigmas[:-1], sigmas[1:]),
        start=1,
    ):
        step_context = (
            trace_recorder.span(
                "ltx.denoise.step",
                attributes={
                    "step_index": step_index,
                    "total_steps": total_steps,
                    "sigma": round(float(sigma_value), 6),
                    "sigma_next": round(float(sigma_next_value), 6),
                    "audio_frames": audio_frames,
                    "cfg_enabled": cfg_enabled,
                    "freeze_audio": False,
                },
                snapshot=mlx_memory_snapshot,
            )
            if trace_recorder is not None
            else nullcontext()
        )
        with step_context:
            sigma = mx.array(float(sigma_value), dtype=latents_dtype)
            sigma_next = mx.array(float(sigma_next_value), dtype=latents_dtype)
            token_timesteps = sigma * denoise_mask
            audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
            audio_flat = mx.reshape(
                audio_flat, (audio_batch, audio_frames, audio_channels * audio_bins)
            )
            modality = _PatchedModality(
                latent=full_tokens,
                sigma=mx.full((batch_size,), float(sigma_value), dtype=latents_dtype),
                timesteps=token_timesteps,
                positions=positions,
                context=text_embeddings,
                context_mask=None,
                attention_mask=self_attention_mask,
                enabled=True,
                positional_embeddings=precomputed_rope,
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
            velocity: MLXArray | None = None
            audio_velocity: MLXArray | None = None
            conditioned_context = (
                trace_recorder.span(
                    "ltx.denoise.forward.conditioned",
                    attributes={"step_index": step_index},
                    sync=(lambda: _sync_optional_arrays(velocity, audio_velocity))
                    if trace_sync
                    else None,
                )
                if trace_recorder is not None
                else nullcontext()
            )
            with conditioned_context:
                velocity, audio_velocity = transformer(
                    video=modality,
                    audio=audio_modality,
                )
            if velocity is None or audio_velocity is None:
                raise RuntimeError(
                    "LTX IC-LoRA transformer returned empty video/audio velocities"
                )
            denoised = imports.to_denoised(full_tokens, velocity, sigma)
            audio_velocity = mx.reshape(
                audio_velocity, (audio_batch, audio_frames, audio_channels, audio_bins)
            )
            audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
            audio_denoised = imports.to_denoised(audio_latents, audio_velocity, sigma)
            negative_denoised: MLXArray | None = None
            negative_audio_denoised: MLXArray | None = None
            if (
                negative_text_embeddings is not None
                and negative_audio_embeddings is not None
            ):
                negative_velocity: MLXArray | None = None
                negative_audio_velocity: MLXArray | None = None
                negative_modality = _PatchedModality(
                    latent=full_tokens,
                    sigma=mx.full(
                        (batch_size,), float(sigma_value), dtype=latents_dtype
                    ),
                    timesteps=token_timesteps,
                    positions=positions,
                    context=negative_text_embeddings,
                    context_mask=None,
                    attention_mask=self_attention_mask,
                    enabled=True,
                    positional_embeddings=precomputed_rope,
                )
                negative_audio_modality = _PatchedModality(
                    latent=audio_flat,
                    sigma=mx.full(
                        (audio_batch,), float(sigma_value), dtype=latents_dtype
                    ),
                    timesteps=sigma * audio_timesteps_mask,
                    positions=audio_positions,
                    context=negative_audio_embeddings,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_audio_rope,
                )
                negative_context = (
                    trace_recorder.span(
                        "ltx.denoise.forward.negative",
                        attributes={"step_index": step_index},
                        sync=(
                            lambda: _sync_optional_arrays(
                                negative_velocity,
                                negative_audio_velocity,
                            )
                        )
                        if trace_sync
                        else None,
                    )
                    if trace_recorder is not None
                    else nullcontext()
                )
                with negative_context:
                    negative_velocity, negative_audio_velocity = transformer(
                        video=negative_modality,
                        audio=negative_audio_modality,
                    )
                if negative_velocity is None or negative_audio_velocity is None:
                    raise RuntimeError(
                        "LTX IC-LoRA transformer returned empty negative velocities"
                    )
                negative_denoised = imports.to_denoised(
                    full_tokens, negative_velocity, sigma
                )
                negative_audio_velocity = mx.reshape(
                    negative_audio_velocity,
                    (audio_batch, audio_frames, audio_channels, audio_bins),
                )
                negative_audio_velocity = mx.transpose(
                    negative_audio_velocity, (0, 2, 1, 3)
                )
                negative_audio_denoised = imports.to_denoised(
                    audio_latents, negative_audio_velocity, sigma
                )
            denoised = _guided_prediction(
                denoised,
                negative_denoised,
                scale=video_cfg_scale if cfg_enabled else 1.0,
                rescale_scale=_GUIDANCE_RESCALE_SCALE if cfg_enabled else 0.0,
            )
            audio_denoised = _guided_prediction(
                audio_denoised,
                negative_audio_denoised,
                scale=audio_cfg_scale if cfg_enabled else 1.0,
                rescale_scale=_GUIDANCE_RESCALE_SCALE if cfg_enabled else 0.0,
            )
            denoised = _apply_token_denoise_mask(
                denoised=denoised,
                clean_tokens=clean_tokens,
                denoise_mask=denoise_mask,
            )
            full_tokens = _next_sigma_tokens(
                noisy=full_tokens,
                denoised=denoised,
                sigma=sigma,
                sigma_next=sigma_next,
            )
            audio_latents = _next_sigma_tokens(
                noisy=audio_latents,
                denoised=audio_denoised,
                sigma=sigma,
                sigma_next=sigma_next,
            )
            mx.eval(full_tokens, audio_latents)
    return full_tokens, audio_latents


def _flatten_video_tokens(latents: MLXArray) -> MLXArray:
    batch, channels, _, _, _ = latents.shape
    return mx.transpose(mx.reshape(latents, (batch, channels, -1)), (0, 2, 1))


def _reshape_video_tokens(
    tokens: MLXArray,
    *,
    channels: int,
    frames: int,
    height: int,
    width: int,
) -> MLXArray:
    return mx.reshape(
        mx.transpose(tokens, (0, 2, 1)),
        (int(tokens.shape[0]), channels, frames, height, width),
    )


def _flatten_video_mask(
    denoise_mask: MLXArray,
    *,
    frames: int,
    height: int,
    width: int,
    dtype: mx.Dtype,
) -> MLXArray:
    expanded = mx.reshape(denoise_mask, (int(denoise_mask.shape[0]), 1, frames, 1, 1))
    expanded = mx.broadcast_to(
        expanded,
        (int(denoise_mask.shape[0]), 1, frames, height, width),
    )
    return mx.reshape(
        expanded, (int(denoise_mask.shape[0]), frames * height * width)
    ).astype(dtype)


def _reference_positions(
    *,
    imports: _ReferenceImports,
    reference_latent: MLXArray,
    fps: float,
    downscale_factor: int,
) -> MLXArray:
    positions = imports.create_position_grid(
        int(reference_latent.shape[0]),
        int(reference_latent.shape[2]),
        int(reference_latent.shape[3]),
        int(reference_latent.shape[4]),
        fps=fps,
    )
    if downscale_factor == 1:
        return positions
    spatial = positions[:, 1:3, :, :] * mx.array(
        float(downscale_factor), dtype=positions.dtype
    )
    return mx.concatenate([positions[:, 0:1, :, :], spatial], axis=1)


def _reference_attention_mask(
    *,
    num_noisy_tokens: int,
    num_reference_tokens: int,
    strength: float,
    dtype: mx.Dtype,
) -> MLXArray | None:
    if strength >= 1.0:
        return None
    noisy_block = mx.ones((1, num_noisy_tokens, num_noisy_tokens), dtype=dtype)
    reference_block = mx.ones(
        (1, num_reference_tokens, num_reference_tokens),
        dtype=dtype,
    )
    cross = mx.full((1, num_noisy_tokens, num_reference_tokens), strength, dtype=dtype)
    top = mx.concatenate([noisy_block, cross], axis=2)
    bottom = mx.concatenate(
        [mx.transpose(cross, (0, 2, 1)), reference_block],
        axis=2,
    )
    return mx.concatenate([top, bottom], axis=1)


def _apply_token_denoise_mask(
    *,
    denoised: MLXArray,
    clean_tokens: MLXArray,
    denoise_mask: MLXArray,
) -> MLXArray:
    expanded_mask = mx.expand_dims(denoise_mask.astype(denoised.dtype), axis=-1)
    one = mx.array(1.0, dtype=denoised.dtype)
    return denoised * expanded_mask + clean_tokens * (one - expanded_mask)


def _next_sigma_tokens(
    *,
    noisy: MLXArray,
    denoised: MLXArray,
    sigma: MLXArray,
    sigma_next: MLXArray,
) -> MLXArray:
    if float(sigma_next.item()) == 0.0:
        return denoised.astype(noisy.dtype)
    return (
        denoised.astype(mx.float32)
        + sigma_next.astype(mx.float32)
        * (noisy.astype(mx.float32) - denoised.astype(mx.float32))
        / sigma.astype(mx.float32)
    ).astype(noisy.dtype)


def _encode_reference_video_latent(
    *,
    imports: _ReferenceImports,
    vae_encoder: _VAEEncoder,
    video_path: Path,
    width: int,
    height: int,
    frame_cap: int,
    dtype: mx.Dtype,
) -> MLXArray:
    frames = _load_reference_video_frames(
        video_path=video_path,
        width=width,
        height=height,
        frame_cap=frame_cap,
    )
    sample = _video_frames_to_sample(frames, dtype=dtype)
    latent = vae_encoder(sample)
    mx.eval(latent)
    return latent


def _load_reference_video_frames(
    *,
    video_path: Path,
    width: int,
    height: int,
    frame_cap: int,
) -> npt.NDArray[np.float32]:
    probed = _probe_video(video_path)
    frames = _decode_video_frames(
        video_path=video_path,
        width=probed.width,
        height=probed.height,
        frame_cap=frame_cap,
    )
    fitted = _fit_reference_frame_count(frames, frame_cap=frame_cap)
    if fitted.size == 0:
        raise RuntimeError("Reference video decoded to zero usable frames")
    return _resize_and_center_crop_frames(fitted, height=height, width=width)


def _fit_reference_frame_count(
    frames: npt.NDArray[np.float32], *, frame_cap: int
) -> npt.NDArray[np.float32]:
    if frames.ndim != 4 or frames.shape[0] == 0:
        return frames
    current = int(frames.shape[0])
    if (current - 1) % 8 == 0:
        return frames
    rounded_up = 1 + math.ceil(max(current - 1, 0) / 8) * 8
    if rounded_up <= frame_cap:
        pad_count = rounded_up - current
        if pad_count <= 0:
            return frames
        last_frame = np.repeat(frames[-1:, ...], pad_count, axis=0)
        return np.concatenate([frames, last_frame], axis=0)
    rounded_down = 1 + ((current - 1) // 8) * 8
    if rounded_down >= 1:
        return frames[:rounded_down]
    return frames[:1]


def _resize_and_center_crop_frames(
    frames: npt.NDArray[np.float32], *, height: int, width: int
) -> npt.NDArray[np.float32]:
    resized: list[npt.NDArray[np.float32]] = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame * np.float32(255.0), dtype=np.uint8))
        scale = max(width / image.width, height / image.height)
        scaled_width = max(1, math.ceil(image.width * scale))
        scaled_height = max(1, math.ceil(image.height * scale))
        scaled = image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.BILINEAR,
        )
        left = max(0, (scaled_width - width) // 2)
        top = max(0, (scaled_height - height) // 2)
        cropped = scaled.crop((left, top, left + width, top + height))
        resized.append(np.asarray(cropped, dtype=np.float32) / np.float32(255.0))
    return np.stack(resized, axis=0).astype(np.float32)


def _video_frames_to_sample(
    frames: npt.NDArray[np.float32], *, dtype: mx.Dtype
) -> MLXArray:
    sample = mx.array(frames * np.float32(2.0) - np.float32(1.0), dtype=dtype)
    sample = mx.transpose(sample, (3, 0, 1, 2))
    return mx.expand_dims(sample, axis=0)


def _probe_video(video_path: Path) -> _ProbedVideo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffprobe failed to inspect IC-LoRA reference video: {stderr}"
        )
    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams", [])
    if not isinstance(streams, list) or not streams:
        raise RuntimeError(
            "IC-LoRA reference video does not contain a readable video stream"
        )
    stream = streams[0]
    fps = _fraction_to_float(
        str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    )
    if fps <= 0.0:
        raise RuntimeError("IC-LoRA reference video must report a positive frame rate")
    num_frames_text = stream.get("nb_frames")
    num_frames = int(num_frames_text) if num_frames_text not in {None, "N/A"} else 0
    return _ProbedVideo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        num_frames=num_frames,
    )


def _decode_video_frames(
    *,
    video_path: Path,
    width: int,
    height: int,
    frame_cap: int,
) -> npt.NDArray[np.float32]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-frames:v",
            str(frame_cap),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode IC-LoRA reference video: {stderr}")
    frame_area = width * height * 3
    if frame_area <= 0 or len(result.stdout) % frame_area != 0:
        raise RuntimeError(
            "IC-LoRA reference video decoded to an invalid rawvideo payload"
        )
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
    return frames.astype(np.float32) / np.float32(255.0)


def _fraction_to_float(value: str) -> float:
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    return float(numerator) / max(float(denominator), 1.0)


def _reference_downscale_factor(lora_inputs: tuple[LoraInput, ...]) -> int:
    scale = 1
    for lora_input in lora_inputs:
        metadata_scale = read_reference_downscale_factor(lora_input.payload_path)
        if metadata_scale == 1:
            continue
        if scale not in {1, metadata_scale}:
            raise ValueError(
                "LTX IC-LoRA inputs cannot mix different reference_downscale_factor values"
            )
        scale = metadata_scale
    return scale
