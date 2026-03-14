from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol, TypeGuard

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
from .debug import (
    _debug_stage_dump_dir,
    _debug_trace_enabled,
    _debug_trace_sync_enabled,
    _elapsed_ms,
    _emit_debug_frame_snapshot,
    _latent_stats,
)
from .outputs import _decode_to_uint8_frames
from .sampling import (
    _assert_prompt_runtime_contract,
    _denoise_distilled_audio_video,
    _denoise_distilled_video_only,
)
from .scheduler import LTX2Scheduler
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
    _RuntimeModelConfig,
    _VAEEncoder,
    _VideoDecoderLike,
    _VideoTransformer,
    _VocoderLike,
)


class _OneStageHost(Protocol):
    checkpoint_path: Path
    _audio_enabled: bool
    _reference_imports: _ReferenceImports | None
    _transformer: _AudioVideoTransformer | _VideoTransformer | None
    _vae_decoder: _VideoDecoderLike | None
    _vae_encoder: _VAEEncoder | None
    _audio_encoder: _AudioEncoderLike | None
    _audio_decoder: _AudioDecoderLike | None
    _audio_processor: _AudioProcessorLike | None
    _vocoder: _VocoderLike | None
    _audio_output_sample_rate: int | None
    _audio_backend: str | None

    def _imports(self) -> _ReferenceImports: ...
    def _ensure_transformer(
        self,
        imports: _ReferenceImports,
        runtime_config: _RuntimeModelConfig,
        prompt_context: PromptEncodingResult,
    ) -> _AudioVideoTransformer | _VideoTransformer: ...
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
    ) -> _ConditioningPlan: ...
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
    def _release_checkpoint_reader(self) -> None: ...


def generate_one_stage(
    host: _OneStageHost,
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
            "LTX one_stage requires the full dev checkpoint, not the distilled checkpoint"
        )

    imports = host._imports()
    runtime_config = _runtime_model_config(host.checkpoint_path)
    use_audio_video_denoising = host._audio_enabled and audio_conditioning is None
    _assert_prompt_runtime_contract(
        prompt_context,
        runtime_config,
        audio_required=use_audio_video_denoising,
    )

    video_context = _require_video_context(prompt_context)
    audio_context = (
        _require_audio_context(prompt_context) if use_audio_video_denoising else None
    )
    negative_guidance_active = prompt_context.negative_prompt_text is not None
    negative_video_context = (
        _optional_negative_video_context(prompt_context)
        if negative_guidance_active
        else None
    )
    negative_audio_context = (
        _optional_negative_audio_context(prompt_context)
        if use_audio_video_denoising and negative_guidance_active
        else None
    )

    padded_shape = _resolve_padded_shape(width=width, height=height)
    effective_generation_seed = effective_seed(seed=seed)
    model_dtype = _prompt_context_dtype(video_context)
    latent_frames = 1 + (num_frames - 1) // 8
    latent_height = padded_shape.internal_height // 32
    latent_width = padded_shape.internal_width // 32
    audio_frames = (
        int(imports.compute_audio_frames(num_frames, float(fps)))
        if host._audio_enabled
        else 0
    )
    steps = 30 if num_inference_steps is None else num_inference_steps
    if steps <= 0:
        raise ValueError("LTX one_stage requires num_inference_steps > 0")
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
        )
    # Match upstream lifecycle: the conditioning encoder is only needed before
    # the transformer loads, so release it before the heaviest stage.
    host._vae_encoder = None
    mx.clear_cache()
    with trace_recorder.span("ltx.ensure_transformer", snapshot=mlx_memory_snapshot):
        transformer = host._ensure_transformer(imports, runtime_config, prompt_context)

    conditioned_audio_latents: MLXArray | None = None
    conditioned_audio_waveform: npt.NDArray[np.float32] | None = None
    conditioned_audio_sample_rate: int | None = None
    if host._audio_enabled and audio_conditioning is not None:
        with trace_recorder.span(
            "ltx.encode_audio_conditioning",
            snapshot=mlx_memory_snapshot,
        ):
            (
                conditioned_audio_latents,
                conditioned_audio_waveform,
                conditioned_audio_sample_rate,
            ) = host._encode_audio_conditioning(
                imports=imports,
                audio_conditioning=audio_conditioning,
                audio_frames=audio_frames,
                model_dtype=model_dtype,
            )
    host._release_checkpoint_reader()

    mx.random.seed(effective_generation_seed)
    started = time.perf_counter()
    positions = imports.create_position_grid(
        1,
        latent_frames,
        latent_height,
        latent_width,
        fps=float(fps),
    )
    audio_positions = (
        imports.create_audio_position_grid(1, audio_frames)
        if use_audio_video_denoising
        else None
    )
    latents = mx.random.normal(
        (1, 128, latent_frames, latent_height, latent_width)
    ).astype(model_dtype)
    if not host._audio_enabled:
        audio_latents = None
    elif conditioned_audio_latents is None:
        audio_latents = mx.random.normal(
            (
                1,
                imports.audio_latent_channels,
                audio_frames,
                imports.audio_mel_bins,
            )
        ).astype(model_dtype)
    else:
        audio_latents = conditioned_audio_latents

    sigmas = LTX2Scheduler().execute(steps=steps, latent=latents)

    stage_state = None
    stage_conditionings = getattr(conditioning_plan, "stage2", ())
    if stage_conditionings:
        stage_state = host._apply_conditionings_to_stage(
            imports=imports,
            latents=latents,
            conditionings=stage_conditionings,
            sigmas=sigmas,
        )
        latents = stage_state.latent

    if use_audio_video_denoising:
        if audio_context is None or audio_positions is None or audio_latents is None:
            raise RuntimeError("LTX one_stage audio-video path requires audio context")
        latents, audio_latents = _denoise_distilled_audio_video(
            imports=imports,
            transformer=_require_audio_video_transformer(transformer),
            latents=latents,
            positions=positions,
            text_embeddings=video_context,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_context,
            negative_text_embeddings=negative_video_context,
            negative_audio_embeddings=negative_audio_context,
            sigmas=sigmas,
            state=stage_state,
            runtime_config=runtime_config,
            freeze_audio=audio_conditioning is not None,
            video_cfg_scale=video_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
        )
        mx.eval(latents, audio_latents)
    else:
        latents = _denoise_distilled_video_only(
            imports=imports,
            transformer=_require_video_transformer(transformer),
            latents=latents,
            positions=positions,
            text_embeddings=video_context,
            negative_text_embeddings=negative_video_context,
            sigmas=sigmas,
            state=stage_state,
            runtime_config=runtime_config,
            video_cfg_scale=video_cfg_scale,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
        )
        mx.eval(latents)
    denoise_duration_ms = _elapsed_ms(started)
    host._transformer = None
    del transformer
    mx.clear_cache()

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
    decode_duration_ms = _elapsed_ms(decode_started)
    frames_uint8 = _decode_to_uint8_frames(decoded_video, padded_shape=padded_shape)
    debug_dir = _debug_stage_dump_dir()
    if debug_dir is not None:
        _emit_debug_frame_snapshot(
            debug_dir=debug_dir,
            stage_name="final",
            frames_uint8=frames_uint8,
            metadata={
                "tiling_mode": tiling_mode,
                "latents": _latent_stats(latents),
                "pipeline_kind": "one_stage",
            },
        )

    audio_decode_started = time.perf_counter()
    if host._audio_enabled:
        if (
            conditioned_audio_waveform is not None
            and conditioned_audio_sample_rate is not None
        ):
            audio_waveform: npt.NDArray[np.float32] | None = conditioned_audio_waveform
            audio_sample_rate = conditioned_audio_sample_rate
            audio_backend = "input_audio_passthrough"
        else:
            audio_waveform, audio_sample_rate, audio_backend = (
                host._decode_audio_waveform(
                    imports=imports,
                    audio_latents=_require_audio_latents(audio_latents),
                )
            )
    else:
        audio_waveform = None
        audio_sample_rate = None
        audio_backend = None
    audio_decode_duration_ms = _elapsed_ms(audio_decode_started)
    mx.clear_cache()

    metadata: dict[str, object] = {
        "pipeline_kind": "one_stage" if host._audio_enabled else "one_stage_video_only",
        "denoise_duration_ms": denoise_duration_ms,
        "decode_duration_ms": decode_duration_ms,
        "audio_decode_duration_ms": audio_decode_duration_ms,
        "tiling_mode": tiling_mode,
        "num_inference_steps": steps,
        "guidance_scale": video_cfg_scale,
        "video_guidance_scale": video_cfg_scale,
        "audio_guidance_scale": audio_cfg_scale,
        "output_width": width,
        "output_height": height,
        "output_frames": num_frames,
        "internal_width": padded_shape.internal_width,
        "internal_height": padded_shape.internal_height,
        "audio_enabled": host._audio_enabled,
        "audio_present": audio_waveform is not None,
        "audio_sample_rate": audio_sample_rate,
        "audio_backend": audio_backend,
        "audio_conditioned": audio_conditioning is not None,
        "audio_video_denoising": use_audio_video_denoising,
        "negative_prompt_present": prompt_context.negative_prompt_text is not None,
        "negative_prompt_applied": negative_guidance_active,
    }
    if trace_recorder.enabled:
        metadata["trace"] = trace_recorder.to_metadata()

    return GeneratedVideo(
        frames=frames_uint8,
        fps=fps,
        seed=effective_generation_seed,
        backend="mlxr_ltx_one_stage"
        if host._audio_enabled
        else "mlxr_ltx_one_stage_video_only",
        conditioning_count=len(conditioning_inputs),
        prompt_signature=_prompt_signature(prompt_context.prompt_text),
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        metadata=metadata,
    )


def _looks_like_distilled_checkpoint(checkpoint_path: Path) -> bool:
    return "distilled" in checkpoint_path.name.lower()


def _require_audio_video_transformer(
    transformer: _AudioVideoTransformer | _VideoTransformer,
) -> _AudioVideoTransformer:
    if not _is_audio_video_transformer(transformer):
        raise RuntimeError("Expected audio-video transformer")
    return transformer


def _require_video_transformer(
    transformer: _AudioVideoTransformer | _VideoTransformer,
) -> _VideoTransformer:
    return transformer


def _require_audio_latents(audio_latents: MLXArray | None) -> MLXArray:
    if audio_latents is None:
        raise RuntimeError("Expected audio latents")
    return audio_latents


def _is_audio_video_transformer(
    transformer: _AudioVideoTransformer | _VideoTransformer,
) -> TypeGuard[_AudioVideoTransformer]:
    return hasattr(transformer, "audio_inner_dim")
