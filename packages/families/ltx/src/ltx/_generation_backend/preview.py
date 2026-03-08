from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..generation import (
    AudioConditioningInput,
    ConditioningInput,
    GeneratedVideo,
    VideoGenerator,
)
from ..prompt_encoding import PromptEncodingResult
from .conditioning import (
    _apply_conditioning,
    _base_frames,
    _effective_seed,
    _prompt_signature,
)


@dataclass(slots=True)
class LTXPreviewVideoGenerator(VideoGenerator):
    checkpoint_path: Path
    spatial_upsampler_path: Path

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
            raise ValueError("LTX preview generation requires width and height >= 32")
        if num_frames < 1:
            raise ValueError("LTX preview generation requires at least one frame")
        if fps < 1:
            raise ValueError("LTX preview generation requires fps >= 1")
        effective_seed = _effective_seed(
            prompt_context=prompt_context,
            checkpoint_path=self.checkpoint_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            seed=seed,
        )
        mx.random.seed(effective_seed)
        frames = _base_frames(
            width=width,
            height=height,
            num_frames=num_frames,
            prompt_context=prompt_context,
            seed=effective_seed,
        )
        for conditioning_input in conditioning_inputs:
            frames = _apply_conditioning(
                frames=frames,
                conditioning_input=conditioning_input,
                width=width,
                height=height,
                num_frames=num_frames,
            )
        del audio_conditioning
        frames_uint8 = np.asarray((mx.clip(frames, 0.0, 1.0) * 255.0).astype(mx.uint8))
        return GeneratedVideo(
            frames=frames_uint8,
            fps=fps,
            seed=effective_seed,
            backend="mlx_prompt_conditioned_preview",
            conditioning_count=len(conditioning_inputs),
            prompt_signature=_prompt_signature(prompt_context.prompt_text),
            metadata={
                "pipeline_kind": "preview",
                "output_width": width,
                "output_height": height,
                "output_frames": num_frames,
                "tiling_mode": "none",
                "audio_present": False,
            },
        )

    def close(self) -> None:
        return None
