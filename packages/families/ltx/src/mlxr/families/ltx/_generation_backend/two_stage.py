from __future__ import annotations

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
from .sampling import (
    _assert_prompt_runtime_contract,
    _denoise_distilled_audio_video,
    _denoise_distilled_video_only,
)
from .scheduler import LTX2Scheduler
from .types import (
    MLXArray,
    _AudioVideoTransformer,
    _ConditionLike,
    _LatentStateLike,
    _PaddedShape,
    _ReferenceImports,
    _RuntimeHelperHost,
    _RuntimeModelConfig,
    _UpsamplerLike,
    _VideoDecoderLike,
    _VideoTransformer,
)


class _TwoStageHost(_RuntimeHelperHost, Protocol):
    distilled_lora_path: Path | None

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
    def _encode_audio_conditioning(
        self,
        *,
        imports: _ReferenceImports,
        audio_conditioning: AudioConditioningInput,
        audio_frames: int,
        model_dtype: mx.Dtype,
    ) -> tuple[MLXArray, npt.NDArray[np.float32], int]: ...
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


def generate_two_stage(
    host: _TwoStageHost,
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
    replace_first_frame_latent: bool = True,
) -> GeneratedVideo:
    if _looks_like_distilled_checkpoint(host.checkpoint_path):
        raise ValueError(
            "LTX two_stage requires the full dev checkpoint, not the distilled checkpoint"
        )
    if host.spatial_upsampler_path is None:
        raise ValueError(
            "LTX two_stage requires the x2 spatial upsampler component in the current artifact"
        )
    if host.distilled_lora_path is None:
        raise ValueError(
            "LTX two_stage requires the distilled LoRA component in the current artifact"
        )
    if audio_conditioning is not None:
        raise ValueError(
            "LTX two_stage currently supports text/image generation only; "
            "audio conditioning remains the separate A2Vid row"
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
        raise ValueError("LTX two_stage requires num_inference_steps > 0")
    video_cfg_scale = 3.0 if guidance_scale is None else guidance_scale
    audio_cfg_scale = 7.0 if guidance_scale is None else guidance_scale

    trace_recorder = TraceRecorder(enabled=_debug_trace_enabled())
    trace_sync = _debug_trace_sync_enabled()

    with trace_recorder.span("ltx.prepare_conditionings", snapshot=mlx_memory_snapshot):
        conditioning_plan = host._prepare_conditionings(
            imports=imports,
            conditioning_inputs=conditioning_inputs,
            num_frames=num_frames,
            latent_frames=latent_frames,
            padded_shape=padded_shape,
            model_dtype=model_dtype,
            replace_first_frame_latent=replace_first_frame_latent,
        )
    # Match upstream ordering: finish image conditioning work, then release the
    # encoder before stage-1 transformer load to avoid overlapping the two
    # heaviest stacks.
    host._vae_encoder = None
    mx.clear_cache()
    with trace_recorder.span("ltx.ensure_transformer", snapshot=mlx_memory_snapshot):
        stage1_transformer = host._ensure_transformer(
            imports, runtime_config, prompt_context
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
            raise RuntimeError("LTX two_stage audio-video path requires audio context")
        latents, audio_latents = _denoise_distilled_audio_video(
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
            runtime_config=runtime_config,
            freeze_audio=False,
            video_cfg_scale=video_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
        )
        mx.eval(latents, audio_latents)
    else:
        latents = _denoise_distilled_video_only(
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
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
        )
        mx.eval(latents)
    timings_ms["stage1_duration_ms"] = _elapsed_ms(stage1_started)
    # Keep stage residency tight: if stage 1 objects stay live here, stage 2 can
    # load on top of another 22B transformer and blow through the worker budget.
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
        lora_scales=(1.0,),
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
            raise RuntimeError("LTX two_stage audio-video path requires audio state")
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
        "pipeline_kind": "two_stage" if host._audio_enabled else "two_stage_video_only",
        "output_width": width,
        "output_height": height,
        "output_frames": num_frames,
        "padded_width": padded_shape.internal_width,
        "padded_height": padded_shape.internal_height,
        "latent_frames": latent_frames,
        "conditioning_count": len(conditioning_inputs),
        "conditioning_mode": (
            "replace_first_frame_latent"
            if replace_first_frame_latent
            else "guiding_keyframes"
        ),
        "audio_conditioned": False,
        "tiling_mode": tiling_mode,
        "timings_ms": {
            **timings_ms,
            "decode_duration_ms": decode_duration_ms,
        },
        "backend": "mlxr_ltx_two_stage"
        if host._audio_enabled
        else "mlxr_ltx_two_stage_video_only",
    }
    if trace_recorder.enabled:
        metadata["trace"] = trace_recorder.to_metadata()
    if audio_backend is not None:
        metadata["audio_backend"] = audio_backend

    return GeneratedVideo(
        frames=np.asarray(frames_uint8, dtype=np.uint8),
        fps=fps,
        seed=effective_generation_seed,
        backend="mlxr_ltx_two_stage"
        if host._audio_enabled
        else "mlxr_ltx_two_stage_video_only",
        conditioning_count=len(conditioning_inputs),
        prompt_signature=_prompt_signature(prompt_context.prompt_text),
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        metadata=metadata,
    )
