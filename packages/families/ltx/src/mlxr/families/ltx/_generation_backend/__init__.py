from __future__ import annotations

from pathlib import Path

from ..generation import VideoGenerator
from .conditioning import _attention_mask
from .config import (
    _assert_prompt_runtime_contract,
    _runtime_model_config,
    _runtime_vocoder_config,
    _validate_reference_backend_compatibility,
    can_use_reference_backend,
)
from .distilled import LTXDistilledVideoGenerator
from .outputs import encode_mp4_video, encode_wav_audio
from .preview import LTXPreviewVideoGenerator
from .types import _RuntimeModelConfig


def create_video_generator(
    *,
    checkpoint_path: Path,
    spatial_upsampler_path: Path,
    audio_enabled: bool = True,
) -> VideoGenerator:
    if can_use_reference_backend(
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
    ):
        return LTXDistilledVideoGenerator(
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            _audio_enabled=audio_enabled,
        )
    return LTXPreviewVideoGenerator(
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
    )


__all__ = [
    "LTXDistilledVideoGenerator",
    "LTXPreviewVideoGenerator",
    "_RuntimeModelConfig",
    "_attention_mask",
    "_assert_prompt_runtime_contract",
    "_runtime_model_config",
    "_runtime_vocoder_config",
    "_validate_reference_backend_compatibility",
    "can_use_reference_backend",
    "create_video_generator",
    "encode_mp4_video",
    "encode_wav_audio",
]
