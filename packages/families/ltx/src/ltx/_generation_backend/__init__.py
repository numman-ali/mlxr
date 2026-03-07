# mypy: ignore-errors
from __future__ import annotations

from .conditioning import _attention_mask
from .config import (
    _assert_prompt_runtime_contract,
    _runtime_model_config,
    _runtime_vocoder_config,
    _RuntimeModelConfig,
    _validate_reference_backend_compatibility,
    can_use_reference_backend,
)
from .distilled import LTXDistilledVideoGenerator
from .outputs import encode_mp4_video, encode_wav_audio
from .preview import LTXPreviewVideoGenerator


def create_video_generator(*, checkpoint_path, spatial_upsampler_path):
    if can_use_reference_backend(
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
    ):
        return LTXDistilledVideoGenerator(
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
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
