from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(slots=True)
class GeneratedVideo:
    frames: npt.NDArray[np.uint8]
    fps: int
    seed: int
    backend: str
    conditioning_count: int
    prompt_signature: str


class VideoGenerator(Protocol):
    def generate(
        self,
        *,
        prompt_context: PromptEncodingResult,
        conditioning_inputs: tuple[ConditioningInput, ...],
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int | None = None,
    ) -> GeneratedVideo: ...

    def close(self) -> None: ...


def create_video_generator(
    checkpoint_path: Path,
    spatial_upsampler_path: Path,
) -> VideoGenerator:
    from ._generation_backend import (
        create_video_generator as create_backend_video_generator,
    )

    return create_backend_video_generator(
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
    )


def encode_mp4_video(video: GeneratedVideo, output_path: Path) -> None:
    from ._generation_backend import encode_mp4_video as encode_backend_mp4_video

    encode_backend_mp4_video(video=video, output_path=output_path)
