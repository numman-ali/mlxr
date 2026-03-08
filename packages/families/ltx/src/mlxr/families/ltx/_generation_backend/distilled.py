from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from mlxr.core.runtime import TraceRecorder, mlx_memory_snapshot

from ..generation import (
    AudioConditioningInput,
    ConditioningInput,
    GeneratedVideo,
    VideoGenerator,
)
from ..prompt_encoding import PromptEncodingResult
from .conditioning import (
    _effective_seed,
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
from .debug import (
    _debug_progress,
    _debug_stage_dump_dir,
    _debug_trace_enabled,
    _debug_trace_sync_enabled,
    _elapsed_ms,
    _emit_debug_frame_snapshot,
    _latent_stats,
)
from .outputs import _decode_to_uint8_frames
from .runtime_helpers import (
    _apply_conditionings_to_stage,
    _decode_audio_waveform,
    _decode_video,
    _encode_audio_conditioning,
    _ensure_audio_encoder,
    _ensure_audio_stack,
    _ensure_transformer,
    _ensure_upsampler,
    _ensure_vae_decoder,
    _ensure_vae_encoder,
    _imports,
    _prepare_conditionings,
)
from .sampling import _assert_prompt_runtime_contract, _denoise_distilled_audio_video
from .types import (
    MLXArray,
    _AudioDecoderLike,
    _AudioEncoderLike,
    _AudioProcessorLike,
    _AudioVideoTransformer,
    _ReferenceImports,
    _UpsamplerLike,
    _VAEEncoder,
    _VideoDecoderLike,
    _VocoderLike,
)


@dataclass(slots=True)
class LTXDistilledVideoGenerator(VideoGenerator):
    checkpoint_path: Path
    spatial_upsampler_path: Path
    _reference_imports: _ReferenceImports | None = None
    _transformer: _AudioVideoTransformer | None = None
    _vae_decoder: _VideoDecoderLike | None = None
    _vae_encoder: _VAEEncoder | None = None
    _upsampler: _UpsamplerLike | None = None
    _audio_encoder: _AudioEncoderLike | None = None
    _audio_decoder: _AudioDecoderLike | None = None
    _audio_processor: _AudioProcessorLike | None = None
    _vocoder: _VocoderLike | None = None
    _audio_output_sample_rate: int | None = None
    _audio_backend: str | None = None

    _imports = _imports
    _ensure_transformer = _ensure_transformer
    _ensure_vae_decoder = _ensure_vae_decoder
    _ensure_vae_encoder = _ensure_vae_encoder
    _ensure_upsampler = _ensure_upsampler
    _ensure_audio_encoder = _ensure_audio_encoder
    _encode_audio_conditioning = _encode_audio_conditioning
    _ensure_audio_stack = _ensure_audio_stack
    _decode_audio_waveform = _decode_audio_waveform
    _prepare_conditionings = _prepare_conditionings
    _apply_conditionings_to_stage = _apply_conditionings_to_stage
    _decode_video = _decode_video

    def generate(
        self,
        *,
        prompt_context: PromptEncodingResult,
        conditioning_inputs: tuple[ConditioningInput, ...],
        audio_conditioning: AudioConditioningInput | None = None,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int | None = None,
    ) -> GeneratedVideo:
        if width < 32 or height < 32:
            raise ValueError("LTX generation requires width and height >= 32")
        if num_frames < 1:
            raise ValueError("LTX generation requires at least one frame")
        if num_frames % 8 != 1:
            raise ValueError("LTX generation requires num_frames to satisfy 8n+1")
        if fps < 1:
            raise ValueError("LTX generation requires fps >= 1")

        imports = self._imports()
        runtime_config = _runtime_model_config(self.checkpoint_path)
        _assert_prompt_runtime_contract(prompt_context, runtime_config)
        video_context = _require_video_context(prompt_context)
        audio_context = _require_audio_context(prompt_context)
        negative_video_context = _optional_negative_video_context(prompt_context)
        negative_audio_context = _optional_negative_audio_context(prompt_context)
        padded_shape = _resolve_padded_shape(width=width, height=height)
        effective_seed = _effective_seed(
            prompt_context=prompt_context,
            checkpoint_path=self.checkpoint_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            seed=seed,
        )
        model_dtype = _prompt_context_dtype(video_context)
        latent_frames = 1 + (num_frames - 1) // 8
        stage1_height = padded_shape.internal_height // 2 // 32
        stage1_width = padded_shape.internal_width // 2 // 32
        stage2_height = padded_shape.internal_height // 32
        stage2_width = padded_shape.internal_width // 32
        audio_frames = int(imports.compute_audio_frames(num_frames, float(fps)))
        conditioned_audio_waveform: npt.NDArray[np.float32] | None = None
        conditioned_audio_sample_rate: int | None = None
        trace_recorder = TraceRecorder(enabled=_debug_trace_enabled())
        trace_sync = _debug_trace_sync_enabled()

        _debug_progress("ensure_transformer start")
        with trace_recorder.span(
            "ltx.ensure_transformer", snapshot=mlx_memory_snapshot
        ):
            transformer = self._ensure_transformer(
                imports, runtime_config, prompt_context
            )
        _debug_progress("ensure_transformer done")
        _debug_progress("ensure_vae_decoder start")
        with trace_recorder.span(
            "ltx.ensure_vae_decoder", snapshot=mlx_memory_snapshot
        ):
            vae_decoder = self._ensure_vae_decoder(imports)
        _debug_progress("ensure_vae_decoder done")
        _debug_progress("ensure_upsampler start")
        with trace_recorder.span("ltx.ensure_upsampler", snapshot=mlx_memory_snapshot):
            upsampler = self._ensure_upsampler(imports)
        _debug_progress("ensure_upsampler done")
        with trace_recorder.span(
            "ltx.prepare_conditionings", snapshot=mlx_memory_snapshot
        ):
            conditioning_plan = self._prepare_conditionings(
                imports=imports,
                conditioning_inputs=conditioning_inputs,
                num_frames=num_frames,
                latent_frames=latent_frames,
                padded_shape=padded_shape,
                model_dtype=model_dtype,
            )
        conditioned_audio_latents: MLXArray | None = None
        if audio_conditioning is not None:
            with trace_recorder.span(
                "ltx.encode_audio_conditioning",
                snapshot=mlx_memory_snapshot,
            ):
                (
                    conditioned_audio_latents,
                    conditioned_audio_waveform,
                    conditioned_audio_sample_rate,
                ) = self._encode_audio_conditioning(
                    imports=imports,
                    audio_conditioning=audio_conditioning,
                    audio_frames=audio_frames,
                    model_dtype=model_dtype,
                )

        mx.random.seed(effective_seed)
        timings_ms: dict[str, float] = {}

        stage1_started = time.perf_counter()
        _debug_progress("stage1 setup")
        stage1_positions = imports.create_position_grid(
            1,
            latent_frames,
            stage1_height,
            stage1_width,
            fps=float(fps),
        )
        stage1_audio_positions = imports.create_audio_position_grid(1, audio_frames)
        latents = mx.random.normal(
            (1, 128, latent_frames, stage1_height, stage1_width)
        ).astype(model_dtype)
        if conditioned_audio_latents is None:
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
        if conditioning_plan.stage1:
            stage1_state = self._apply_conditionings_to_stage(
                imports=imports,
                latents=latents,
                conditionings=conditioning_plan.stage1,
                sigmas=imports.stage_1_sigmas,
            )
            latents = stage1_state.latent
        else:
            stage1_state = None
        latents, audio_latents = _denoise_distilled_audio_video(
            imports=imports,
            transformer=transformer,
            latents=latents,
            positions=stage1_positions,
            text_embeddings=video_context,
            audio_latents=audio_latents,
            audio_positions=stage1_audio_positions,
            audio_embeddings=audio_context,
            negative_text_embeddings=negative_video_context,
            negative_audio_embeddings=negative_audio_context,
            sigmas=imports.stage_1_sigmas,
            state=stage1_state,
            runtime_config=runtime_config,
            freeze_audio=audio_conditioning is not None,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
        )
        mx.eval(latents, audio_latents)
        debug_dir = _debug_stage_dump_dir()
        if debug_dir is not None:
            _debug_progress("debug decode stage1")
            stage1_padded_shape = _half_resolution_padded_shape(
                padded_shape=padded_shape,
                width=width,
                height=height,
            )
            stage1_decoded, stage1_tiling = self._decode_video(
                imports=imports,
                vae_decoder=vae_decoder,
                latents=latents,
                padded_shape=stage1_padded_shape,
                num_frames=num_frames,
            )
            stage1_frames = _decode_to_uint8_frames(
                stage1_decoded,
                padded_shape=stage1_padded_shape,
            )
            _emit_debug_frame_snapshot(
                debug_dir=debug_dir,
                stage_name="stage1",
                frames_uint8=stage1_frames,
                metadata={
                    "tiling_mode": stage1_tiling,
                    "latents": _latent_stats(latents),
                },
            )
            mx.clear_cache()
        timings_ms["stage1_duration_ms"] = _elapsed_ms(stage1_started)

        upsample_started = time.perf_counter()
        _debug_progress("upsample start")
        latents = imports.upsample_latents(
            latents,
            upsampler,
            vae_decoder.latents_mean,
            vae_decoder.latents_std,
        )
        mx.eval(latents)
        if debug_dir is not None:
            _debug_progress("debug decode post_x2")
            post_x2_decoded, post_x2_tiling = self._decode_video(
                imports=imports,
                vae_decoder=vae_decoder,
                latents=latents,
                padded_shape=padded_shape,
                num_frames=num_frames,
            )
            post_x2_frames = _decode_to_uint8_frames(
                post_x2_decoded,
                padded_shape=padded_shape,
            )
            _emit_debug_frame_snapshot(
                debug_dir=debug_dir,
                stage_name="post_x2",
                frames_uint8=post_x2_frames,
                metadata={
                    "tiling_mode": post_x2_tiling,
                    "latents": _latent_stats(latents),
                },
            )
            mx.clear_cache()
        timings_ms["upsample_duration_ms"] = _elapsed_ms(upsample_started)

        stage2_started = time.perf_counter()
        _debug_progress("stage2 setup")
        stage2_positions = imports.create_position_grid(
            1,
            latent_frames,
            stage2_height,
            stage2_width,
            fps=float(fps),
        )
        stage2_audio_positions = imports.create_audio_position_grid(1, audio_frames)
        if conditioning_plan.stage2:
            stage2_state = self._apply_conditionings_to_stage(
                imports=imports,
                latents=latents,
                conditionings=conditioning_plan.stage2,
                sigmas=imports.stage_2_sigmas,
            )
            latents = stage2_state.latent
            noise_scale = mx.array(float(imports.stage_2_sigmas[0]), dtype=model_dtype)
            one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
            if audio_conditioning is None:
                audio_latents = (
                    mx.random.normal(audio_latents.shape).astype(model_dtype)
                    * noise_scale
                    + audio_latents * one_minus_scale
                ).astype(model_dtype)
            mx.eval(latents, audio_latents)
        else:
            stage2_state = None
            noise_scale = mx.array(float(imports.stage_2_sigmas[0]), dtype=model_dtype)
            one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
            latents = (
                mx.random.normal(latents.shape).astype(model_dtype) * noise_scale
                + latents * one_minus_scale
            ).astype(model_dtype)
            if audio_conditioning is None:
                audio_latents = (
                    mx.random.normal(audio_latents.shape).astype(model_dtype)
                    * noise_scale
                    + audio_latents * one_minus_scale
                ).astype(model_dtype)
            mx.eval(latents, audio_latents)
        latents, audio_latents = _denoise_distilled_audio_video(
            imports=imports,
            transformer=transformer,
            latents=latents,
            positions=stage2_positions,
            text_embeddings=video_context,
            audio_latents=audio_latents,
            audio_positions=stage2_audio_positions,
            audio_embeddings=audio_context,
            negative_text_embeddings=negative_video_context,
            negative_audio_embeddings=negative_audio_context,
            sigmas=imports.stage_2_sigmas,
            state=stage2_state,
            runtime_config=runtime_config,
            freeze_audio=audio_conditioning is not None,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
        )
        mx.eval(latents, audio_latents)
        timings_ms["stage2_duration_ms"] = _elapsed_ms(stage2_started)

        decode_started = time.perf_counter()
        decoded_video, tiling_mode = self._decode_video(
            imports=imports,
            vae_decoder=vae_decoder,
            latents=latents,
            padded_shape=padded_shape,
            num_frames=num_frames,
        )
        timings_ms["decode_duration_ms"] = _elapsed_ms(decode_started)
        frames_uint8 = _decode_to_uint8_frames(decoded_video, padded_shape=padded_shape)
        audio_decode_started = time.perf_counter()
        if (
            conditioned_audio_waveform is not None
            and conditioned_audio_sample_rate is not None
        ):
            audio_waveform: npt.NDArray[np.float32] | None = conditioned_audio_waveform
            audio_sample_rate = conditioned_audio_sample_rate
            audio_backend = "input_audio_passthrough"
        else:
            audio_waveform, audio_sample_rate, audio_backend = (
                self._decode_audio_waveform(
                    imports=imports,
                    audio_latents=audio_latents,
                )
            )
        timings_ms["audio_decode_duration_ms"] = _elapsed_ms(audio_decode_started)
        if debug_dir is not None:
            _emit_debug_frame_snapshot(
                debug_dir=debug_dir,
                stage_name="final",
                frames_uint8=frames_uint8,
                metadata={
                    "tiling_mode": tiling_mode,
                    "latents": _latent_stats(latents),
                },
            )
        mx.clear_cache()

        metadata: dict[str, object] = {
            "pipeline_kind": "distilled_two_stage",
            "stage1_duration_ms": timings_ms["stage1_duration_ms"],
            "upsample_duration_ms": timings_ms["upsample_duration_ms"],
            "stage2_duration_ms": timings_ms["stage2_duration_ms"],
            "decode_duration_ms": timings_ms["decode_duration_ms"],
            "audio_decode_duration_ms": timings_ms["audio_decode_duration_ms"],
            "tiling_mode": tiling_mode,
            "output_width": width,
            "output_height": height,
            "output_frames": num_frames,
            "internal_width": padded_shape.internal_width,
            "internal_height": padded_shape.internal_height,
            "audio_present": audio_waveform is not None,
            "audio_sample_rate": audio_sample_rate,
            "audio_channels": (
                int(audio_waveform.shape[1])
                if audio_waveform is not None and audio_waveform.ndim == 2
                else 1
                if audio_waveform is not None
                else 0
            ),
            "audio_backend": audio_backend,
            "audio_bwe_applied": audio_backend == "mlxr_vocoder_with_bwe",
            "audio_conditioned": audio_conditioning is not None,
            "guidance_mode": (
                "cfg"
                if prompt_context.negative_prompt_text is not None
                else "positive_only"
            ),
            "negative_prompt_present": prompt_context.negative_prompt_text is not None,
        }
        if trace_recorder.enabled:
            metadata["trace"] = trace_recorder.to_metadata()

        return GeneratedVideo(
            frames=frames_uint8,
            fps=fps,
            seed=effective_seed,
            backend="mlxr_ltx_distilled_two_stage",
            conditioning_count=len(conditioning_inputs),
            prompt_signature=_prompt_signature(prompt_context.prompt_text),
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            metadata=metadata,
        )

    def close(self) -> None:
        self._transformer = None
        self._vae_decoder = None
        self._vae_encoder = None
        self._upsampler = None
        self._audio_encoder = None
        self._audio_decoder = None
        self._audio_processor = None
        self._vocoder = None
        self._audio_output_sample_rate = None
        self._audio_backend = None
        mx.clear_cache()
