from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from mlxr.core.runtime import TraceRecorder, mlx_memory_snapshot

from ..family_options import effective_seed
from ..generation import (
    AudioConditioningInput,
    GeneratedVideo,
    RetakeOptions,
    VideoReferenceInput,
)
from ..prompt_encoding import PromptEncodingResult
from .conditioning import (
    _optional_negative_audio_context,
    _optional_negative_video_context,
    _prompt_context_dtype,
    _prompt_signature,
    _require_audio_context,
    _require_video_context,
)
from .config import _runtime_model_config
from .debug import _debug_trace_enabled, _debug_trace_sync_enabled, _elapsed_ms
from .one_stage import (
    _looks_like_distilled_checkpoint,
    _require_audio_video_transformer,
    _require_video_transformer,
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
    _AudioVideoTransformer,
    _LatentStateLike,
    _PaddedShape,
    _ReferenceImports,
    _RuntimeHelperHost,
    _RuntimeModelConfig,
    _VAEEncoder,
    _VideoDecoderLike,
    _VideoTransformer,
)

_AUDIO_HOP_LENGTH = 160
_AUDIO_DOWNSAMPLE_FACTOR = 4


class _RetakeHost(_RuntimeHelperHost, Protocol):
    def _imports(self) -> _ReferenceImports: ...

    def _ensure_transformer(
        self,
        imports: _ReferenceImports,
        runtime_config: _RuntimeModelConfig,
        prompt_context: PromptEncodingResult,
    ) -> _AudioVideoTransformer | _VideoTransformer: ...

    def _ensure_vae_encoder(self, imports: _ReferenceImports) -> _VAEEncoder: ...

    def _ensure_vae_decoder(self, imports: _ReferenceImports) -> _VideoDecoderLike: ...

    def _ensure_audio_encoder(self, imports: _ReferenceImports) -> object: ...

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

    def _encode_audio_conditioning(
        self,
        *,
        imports: _ReferenceImports,
        audio_conditioning: AudioConditioningInput,
        audio_frames: int,
        model_dtype: mx.Dtype,
    ) -> tuple[MLXArray, npt.NDArray[np.float32], int]: ...

    def _release_checkpoint_reader(self) -> None: ...


def generate_retake(
    host: _RetakeHost,
    *,
    prompt_context: PromptEncodingResult,
    video_inputs: tuple[VideoReferenceInput, ...],
    retake_options: RetakeOptions,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    seed: int | None,
) -> GeneratedVideo:
    if len(video_inputs) != 1:
        raise ValueError("LTX retake requires exactly one source video input")
    source_video = video_inputs[0]
    source_meta = _probe_video(source_video.payload_path)
    if source_meta.width % 32 != 0 or source_meta.height % 32 != 0:
        raise ValueError(
            "LTX retake requires source video width and height to be divisible by 32"
        )
    if source_meta.num_frames % 8 != 1:
        raise ValueError("LTX retake requires source video num_frames to satisfy 8n+1")
    if retake_options.end_time_seconds > source_meta.duration_seconds + 1e-6:
        raise ValueError(
            "LTX retake window_end_seconds must be within the source video duration"
        )

    imports = host._imports()
    runtime_config = _runtime_model_config(host.checkpoint_path)
    distilled = _looks_like_distilled_checkpoint(host.checkpoint_path)
    model_dtype = _prompt_context_dtype(_require_video_context(prompt_context))
    audio_required = bool(getattr(host, "_audio_enabled", True))
    _assert_prompt_runtime_contract(
        prompt_context,
        runtime_config,
        audio_required=audio_required,
    )
    video_context = _require_video_context(prompt_context)
    audio_context = _require_audio_context(prompt_context) if audio_required else None
    negative_video_context = (
        None if distilled else _optional_negative_video_context(prompt_context)
    )
    negative_audio_context = (
        None if distilled else _optional_negative_audio_context(prompt_context)
    )

    trace_recorder = TraceRecorder(enabled=_debug_trace_enabled())
    trace_sync = _debug_trace_sync_enabled()

    with trace_recorder.span(
        "ltx.retake.load_source_video", snapshot=mlx_memory_snapshot
    ):
        source_frames = _decode_video_frames(
            source_video.payload_path,
            width=source_meta.width,
            height=source_meta.height,
        )
        video_sample = _video_frames_to_sample(source_frames, dtype=model_dtype)

    num_frames = source_meta.num_frames
    fps = int(round(source_meta.fps))
    latent_frames = 1 + (num_frames - 1) // 8
    latent_height = source_meta.height // 32
    latent_width = source_meta.width // 32
    padded_shape = _PaddedShape(
        output_width=source_meta.width,
        output_height=source_meta.height,
        internal_width=source_meta.width,
        internal_height=source_meta.height,
        crop_top=0,
        crop_left=0,
    )
    video_positions = imports.create_position_grid(
        1,
        latent_frames,
        latent_height,
        latent_width,
        fps=source_meta.fps,
    )

    with trace_recorder.span(
        "ltx.retake.encode_source_video", snapshot=mlx_memory_snapshot
    ):
        video_encoder = host._ensure_vae_encoder(imports)
        source_video_latent = video_encoder(video_sample)
        mx.eval(source_video_latent)
    video_encoder = None
    host._vae_encoder = None
    mx.clear_cache()

    video_state = _retake_video_state(
        imports=imports,
        clean_latent=source_video_latent,
        start_time_seconds=retake_options.start_time_seconds,
        end_time_seconds=retake_options.end_time_seconds,
        fps=source_meta.fps,
        num_frames=num_frames,
        latent_frames=latent_frames,
        model_dtype=model_dtype,
    )

    audio_frames = int(imports.compute_audio_frames(num_frames, source_meta.fps))
    zero_audio_latents = mx.zeros(
        (1, imports.audio_latent_channels, audio_frames, imports.audio_mel_bins),
        dtype=model_dtype,
    )
    audio_latents = zero_audio_latents
    audio_state: _LatentStateLike | None = None
    preserved_audio_waveform: npt.NDArray[np.float32] | None = None
    preserved_audio_sample_rate: int | None = None
    audio_backend = "none"
    if audio_required:
        source_audio = AudioConditioningInput(
            handle_id="source-video-audio",
            payload_path=source_video.payload_path,
            start_time_seconds=0.0,
            max_duration_seconds=None,
            filename=source_video.payload_path.name,
        )
        try:
            with trace_recorder.span(
                "ltx.retake.encode_source_audio", snapshot=mlx_memory_snapshot
            ):
                (
                    encoded_audio_latents,
                    preserved_audio_waveform,
                    preserved_audio_sample_rate,
                ) = host._encode_audio_conditioning(
                    imports=imports,
                    audio_conditioning=source_audio,
                    audio_frames=audio_frames,
                    model_dtype=model_dtype,
                )
                audio_latents = encoded_audio_latents
        except RuntimeError:
            if retake_options.regenerate_audio:
                raise ValueError(
                    "LTX retake regenerate_audio requires a source video with an audio track"
                ) from None
            preserved_audio_waveform = None
            preserved_audio_sample_rate = None

        if preserved_audio_waveform is not None and retake_options.regenerate_audio:
            audio_state = _retake_audio_state(
                imports=imports,
                clean_latent=audio_latents,
                start_time_seconds=retake_options.start_time_seconds,
                end_time_seconds=retake_options.end_time_seconds,
                audio_frames=audio_frames,
                audio_sample_rate=imports.audio_sample_rate,
                model_dtype=model_dtype,
            )

    with trace_recorder.span(
        "ltx.retake.ensure_transformer", snapshot=mlx_memory_snapshot
    ):
        transformer = host._ensure_transformer(imports, runtime_config, prompt_context)
    host._release_checkpoint_reader()

    effective_generation_seed = effective_seed(seed=seed)
    mx.random.seed(effective_generation_seed)
    timings_ms: dict[str, float] = {}
    sigmas = (
        imports.stage_1_sigmas
        if distilled
        else tuple(
            float(value)
            for value in np.asarray(
                LTX2Scheduler().execute(
                    steps=40 if num_inference_steps is None else num_inference_steps
                )
            ).tolist()
        )
    )
    if len(sigmas) < 2:
        raise ValueError("LTX retake requires at least two sigma values")
    sigma0 = mx.array(float(sigmas[0]), dtype=model_dtype)
    latents = _initial_noisy_latent(video_state, sigma0)

    if audio_required:
        if retake_options.regenerate_audio and audio_state is not None:
            audio_latents = _initial_noisy_latent(audio_state, sigma0)
            freeze_audio = False
        else:
            freeze_audio = True
    else:
        freeze_audio = True

    denoise_started = time.perf_counter()
    if audio_required:
        transformer_av = _require_audio_video_transformer(transformer)
        if audio_context is None:
            raise ValueError("LTX retake requires audio prompt context on the AV path")
        audio_positions = imports.create_audio_position_grid(1, audio_frames)
        latents, audio_latents = _denoise_distilled_audio_video(
            imports=imports,
            transformer=transformer_av,
            latents=latents,
            positions=video_positions,
            text_embeddings=video_context,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_context,
            negative_text_embeddings=negative_video_context,
            negative_audio_embeddings=negative_audio_context,
            sigmas=sigmas,
            state=video_state,
            audio_state=audio_state,
            runtime_config=runtime_config,
            freeze_audio=freeze_audio,
            video_cfg_scale=3.0 if guidance_scale is None else guidance_scale,
            audio_cfg_scale=7.0 if guidance_scale is None else guidance_scale,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
        )
    else:
        transformer_video = _require_video_transformer(transformer)
        latents = _denoise_distilled_video_only(
            imports=imports,
            transformer=transformer_video,
            latents=latents,
            positions=video_positions,
            text_embeddings=video_context,
            negative_text_embeddings=negative_video_context,
            sigmas=sigmas,
            state=video_state,
            runtime_config=runtime_config,
            video_cfg_scale=3.0 if guidance_scale is None else guidance_scale,
            trace_recorder=trace_recorder,
            trace_sync=trace_sync,
        )
    timings_ms["retake_duration_ms"] = _elapsed_ms(denoise_started)

    decode_started = time.perf_counter()
    vae_decoder = host._ensure_vae_decoder(imports)
    decoded_video, tiling_mode = host._decode_video(
        imports=imports,
        vae_decoder=vae_decoder,
        latents=latents,
        padded_shape=padded_shape,
        num_frames=num_frames,
    )
    timings_ms["decode_duration_ms"] = _elapsed_ms(decode_started)

    audio_waveform = preserved_audio_waveform
    audio_sample_rate = preserved_audio_sample_rate
    if audio_required and retake_options.regenerate_audio and audio_state is not None:
        audio_decode_started = time.perf_counter()
        audio_waveform, audio_sample_rate, audio_backend = host._decode_audio_waveform(
            imports=imports,
            audio_latents=audio_latents,
        )
        timings_ms["audio_decode_duration_ms"] = _elapsed_ms(audio_decode_started)

    frames_uint8 = _decode_to_uint8_frames(
        decoded_video,
        padded_shape=padded_shape,
    )
    metadata: dict[str, object] = {
        "pipeline_kind": "retake_distilled" if distilled else "retake_full",
        "output_width": source_meta.width,
        "output_height": source_meta.height,
        "output_frames": num_frames,
        "tiling_mode": tiling_mode,
        "source_filename": source_video.filename or source_video.payload_path.name,
        "retake_window_start_seconds": retake_options.start_time_seconds,
        "retake_window_end_seconds": retake_options.end_time_seconds,
        "regenerate_video": retake_options.regenerate_video,
        "regenerate_audio": retake_options.regenerate_audio,
        "audio_present": audio_waveform is not None,
        "audio_backend": audio_backend,
    }
    metadata.update(timings_ms)
    return GeneratedVideo(
        frames=np.asarray(frames_uint8),
        fps=fps,
        seed=effective_generation_seed,
        backend="mlxr_ltx_retake",
        conditioning_count=0,
        prompt_signature=_prompt_signature(prompt_context.prompt_text),
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        metadata=metadata,
    )


class _ProbedVideo:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float,
        num_frames: int,
        duration_seconds: float,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.num_frames = num_frames
        self.duration_seconds = duration_seconds


def _probe_video(video_path: Path) -> _ProbedVideo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffprobe failed to inspect retake source video: {stderr}")
    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams", [])
    if not isinstance(streams, list) or not streams:
        raise RuntimeError(
            "Retake source video does not contain a readable video stream"
        )
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = _fraction_to_float(
        str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    )
    num_frames_text = stream.get("nb_frames")
    duration_text = stream.get("duration")
    if fps <= 0.0:
        raise RuntimeError("Retake source video must report a positive frame rate")
    if num_frames_text not in {None, "N/A"}:
        num_frames = int(num_frames_text)
    elif duration_text not in {None, "N/A"}:
        duration_seconds = float(duration_text)
        num_frames = int(round(duration_seconds * fps))
    else:
        raise RuntimeError(
            "Retake source video must report either nb_frames or duration"
        )
    duration_seconds = (
        float(duration_text) if duration_text not in {None, "N/A"} else num_frames / fps
    )
    return _ProbedVideo(
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
        duration_seconds=duration_seconds,
    )


def _decode_video_frames(
    video_path: Path, *, width: int, height: int
) -> npt.NDArray[np.float32]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
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
        raise RuntimeError(f"ffmpeg failed to decode retake source video: {stderr}")
    frame_area = width * height * 3
    if frame_area <= 0 or len(result.stdout) % frame_area != 0:
        raise RuntimeError("Retake source video decoded to an invalid rawvideo payload")
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
    return frames.astype(np.float32) / np.float32(255.0)


def _video_frames_to_sample(
    frames: npt.NDArray[np.float32], *, dtype: mx.Dtype
) -> MLXArray:
    sample = mx.array(frames * np.float32(2.0) - np.float32(1.0), dtype=dtype)
    sample = mx.transpose(sample, (3, 0, 1, 2))
    sample = mx.expand_dims(sample, axis=0)
    return sample


def _fraction_to_float(value: str) -> float:
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    return float(numerator) / max(float(denominator), 1.0)


def _retake_video_state(
    *,
    imports: _ReferenceImports,
    clean_latent: MLXArray,
    start_time_seconds: float,
    end_time_seconds: float,
    fps: float,
    num_frames: int,
    latent_frames: int,
    model_dtype: mx.Dtype,
) -> _LatentStateLike:
    mask = mx.zeros((1, 1, latent_frames, 1, 1), dtype=model_dtype)
    start_frame = max(0, int(math.floor(start_time_seconds * fps)))
    end_frame = min(num_frames, int(math.ceil(end_time_seconds * fps)))
    if end_frame <= start_frame:
        raise ValueError("Retake time window did not map to any video frames")
    start_latent = min(latent_frames - 1, max(0, start_frame // 8))
    end_latent = min(latent_frames, max(start_latent + 1, ((end_frame - 1) // 8) + 1))
    mask[:, :, start_latent:end_latent, :, :] = 1.0
    return imports.latent_state_class(
        latent=clean_latent,
        clean_latent=clean_latent,
        denoise_mask=mask,
    )


def _retake_audio_state(
    *,
    imports: _ReferenceImports,
    clean_latent: MLXArray,
    start_time_seconds: float,
    end_time_seconds: float,
    audio_frames: int,
    audio_sample_rate: int,
    model_dtype: mx.Dtype,
) -> _LatentStateLike:
    latents_per_second = (
        audio_sample_rate / _AUDIO_HOP_LENGTH / _AUDIO_DOWNSAMPLE_FACTOR
    )
    start_index = max(0, int(math.floor(start_time_seconds * latents_per_second)))
    end_index = min(audio_frames, int(math.ceil(end_time_seconds * latents_per_second)))
    if end_index <= start_index:
        raise ValueError("Retake time window did not map to any audio latents")
    mask = mx.zeros((1, 1, audio_frames, 1), dtype=model_dtype)
    mask[:, :, start_index:end_index, :] = 1.0
    return imports.latent_state_class(
        latent=clean_latent,
        clean_latent=clean_latent,
        denoise_mask=mask,
    )


def _initial_noisy_latent(state: _LatentStateLike, sigma: MLXArray) -> MLXArray:
    noise = mx.random.normal(state.clean_latent.shape).astype(state.clean_latent.dtype)
    scaled_mask = state.denoise_mask * sigma
    one = mx.array(1.0, dtype=state.clean_latent.dtype)
    noisy = noise * scaled_mask + state.clean_latent * (one - scaled_mask)
    mx.eval(noisy)
    return noisy
