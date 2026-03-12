from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .prompt_encoding import PromptEncodingResult


@dataclass(frozen=True, slots=True)
class ConditioningInput:
    handle_id: str
    payload_path: Path
    frame_index: int
    strength: float
    media_type: str | None = None
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class AudioConditioningInput:
    handle_id: str
    payload_path: Path
    start_time_seconds: float = 0.0
    max_duration_seconds: float | None = None
    media_type: str | None = None
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class VideoReferenceInput:
    handle_id: str
    payload_path: Path
    strength: float = 1.0
    media_type: str | None = None
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class LoraInput:
    handle_id: str
    payload_path: Path
    strength: float = 1.0
    media_type: str | None = None
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class RetakeOptions:
    start_time_seconds: float
    end_time_seconds: float
    regenerate_video: bool = True
    regenerate_audio: bool = True


@dataclass(slots=True)
class GeneratedVideo:
    frames: npt.NDArray[np.uint8]
    fps: int
    seed: int
    backend: str
    conditioning_count: int
    prompt_signature: str
    audio_waveform: npt.NDArray[np.float32] | None = None
    audio_sample_rate: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class VideoGenerator(Protocol):
    def generate(
        self,
        *,
        prompt_context: PromptEncodingResult,
        task: str = "video.generate",
        conditioning_inputs: tuple[ConditioningInput, ...],
        video_inputs: tuple[VideoReferenceInput, ...] = (),
        lora_inputs: tuple[LoraInput, ...] = (),
        audio_conditioning: AudioConditioningInput | None = None,
        retake_options: RetakeOptions | None = None,
        control_variant: str | None = None,
        conditioning_attention_strength: float | None = None,
        pipeline_variant: str = "distilled_two_stage",
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int | None = None,
    ) -> GeneratedVideo: ...

    def close(self) -> None: ...


def create_video_generator(
    checkpoint_path: Path,
    spatial_upsampler_path: Path | None,
    distilled_lora_path: Path | None,
    *,
    audio_enabled: bool = True,
) -> VideoGenerator:
    from ._generation_backend import (
        create_video_generator as create_backend_video_generator,
    )

    return create_backend_video_generator(
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
        distilled_lora_path=distilled_lora_path,
        audio_enabled=audio_enabled,
    )


def encode_mp4_video(video: GeneratedVideo, output_path: Path) -> None:
    backend_module = import_module("mlxr.families.ltx._generation_backend")
    encode_backend_mp4_video = backend_module.encode_mp4_video
    encode_backend_mp4_video(video=video, output_path=output_path)


def encode_wav_audio(video: GeneratedVideo, output_path: Path) -> None:
    backend_module = import_module("mlxr.families.ltx._generation_backend")
    encode_backend_wav_audio = backend_module.encode_wav_audio
    encode_backend_wav_audio(video=video, output_path=output_path)
