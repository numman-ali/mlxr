from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mlx.core as mx

from ... import _nn_compat as nn


class AudioCausalityAxis(Enum):
    NONE = "none"
    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_COMPATIBILITY = "width_compatibility"

    @classmethod
    def from_config_value(cls, value: str) -> "AudioCausalityAxis":
        normalized = value.strip().lower().replace("-", "_")
        if normalized == "none":
            return cls.NONE
        if normalized == "width":
            return cls.WIDTH
        if normalized == "height":
            return cls.HEIGHT
        if normalized == "width_compatibility":
            return cls.WIDTH_COMPATIBILITY
        raise ValueError(f"Unsupported audio causality axis {value!r}")


class AudioNormKind(Enum):
    GROUP = "group"
    PIXEL = "pixel"

    @classmethod
    def from_config_value(cls, value: str) -> "AudioNormKind":
        normalized = value.strip().lower()
        if normalized == "group":
            return cls.GROUP
        if normalized == "pixel":
            return cls.PIXEL
        raise ValueError(f"Unsupported audio norm kind {value!r}")


@dataclass(frozen=True, slots=True)
class AudioLatentShape:
    batch: int
    channels: int
    frames: int
    mel_bins: int


class PerChannelStatistics(nn.Module):
    _std_of_means: mx.array
    _mean_of_means: mx.array

    def __init__(self, width: int) -> None:
        super().__init__()
        self._std_of_means = mx.ones((width,), dtype=mx.float32)
        self._mean_of_means = mx.zeros((width,), dtype=mx.float32)

    def normalize(self, x: mx.array) -> mx.array:
        std = self._std_of_means.astype(x.dtype)
        mean = self._mean_of_means.astype(x.dtype)
        return (x - mean) / std

    def un_normalize(self, x: mx.array) -> mx.array:
        std = self._std_of_means.astype(x.dtype)
        mean = self._mean_of_means.astype(x.dtype)
        return (x * std) + mean


@dataclass(frozen=True, slots=True)
class AudioPatchifier:
    patch_size: int = 1
    sample_rate: int = 16000
    hop_length: int = 160
    audio_latent_downsample_factor: int = 4
    is_causal: bool = True
    shift: int = 0

    def patchify(self, audio_latents: mx.array) -> mx.array:
        if audio_latents.ndim != 4:
            raise ValueError(
                "AudioPatchifier.patchify expects [batch, channels, frames, mel_bins]"
            )
        batch, channels, frames, mel_bins = (int(size) for size in audio_latents.shape)
        transposed = mx.transpose(audio_latents, (0, 2, 1, 3))
        return transposed.reshape(batch, frames, channels * mel_bins)

    def unpatchify(
        self,
        audio_latents: mx.array,
        output_shape: AudioLatentShape,
    ) -> mx.array:
        if audio_latents.ndim != 3:
            raise ValueError(
                "AudioPatchifier.unpatchify expects [batch, frames, channels_x_mel_bins]"
            )
        batch, frames, width = (int(size) for size in audio_latents.shape)
        expected_width = output_shape.channels * output_shape.mel_bins
        if batch != output_shape.batch or frames != output_shape.frames:
            raise ValueError(
                "AudioPatchifier.unpatchify input does not match the target batch/frame shape"
            )
        if width != expected_width:
            raise ValueError(
                f"AudioPatchifier expected flattened width {expected_width}, got {width}"
            )
        restored = audio_latents.reshape(
            output_shape.batch,
            output_shape.frames,
            output_shape.channels,
            output_shape.mel_bins,
        )
        return mx.transpose(restored, (0, 2, 1, 3))

    def latent_time_bounds_seconds(self, latent_index: int) -> tuple[float, float]:
        start_index = latent_index + self.shift
        end_index = start_index + 1
        start_frames = start_index * self.audio_latent_downsample_factor
        end_frames = end_index * self.audio_latent_downsample_factor
        if self.is_causal:
            start_frames = max(
                start_frames + 1 - self.audio_latent_downsample_factor,
                0,
            )
            end_frames = max(
                end_frames + 1 - self.audio_latent_downsample_factor,
                0,
            )
        scale = self.hop_length / float(self.sample_rate)
        return start_frames * scale, end_frames * scale
