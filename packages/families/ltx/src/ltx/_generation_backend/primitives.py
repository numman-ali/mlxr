from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .types import MLXArray, _ConditionLike, _LatentStateLike

STAGE_1_SIGMAS: tuple[float, ...] = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)

STAGE_2_SIGMAS: tuple[float, ...] = (
    0.909375,
    0.725,
    0.421875,
    0.0,
)

_AUDIO_LATENT_SAMPLE_RATE = 16000
_AUDIO_HOP_LENGTH = 160
_AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
_AUDIO_LATENTS_PER_SECOND = (
    _AUDIO_LATENT_SAMPLE_RATE / _AUDIO_HOP_LENGTH / _AUDIO_LATENT_DOWNSAMPLE_FACTOR
)


@dataclass(frozen=True, slots=True)
class VideoConditionByLatentIndex:
    latent: MLXArray
    frame_idx: int = 0
    strength: float = 1.0


@dataclass(slots=True)
class LatentState:
    latent: MLXArray
    clean_latent: MLXArray
    denoise_mask: MLXArray

    def clone(self) -> LatentState:
        return LatentState(
            latent=self.latent,
            clean_latent=self.clean_latent,
            denoise_mask=self.denoise_mask,
        )


def apply_conditioning(
    state: _LatentStateLike,
    conditionings: list[_ConditionLike],
) -> _LatentStateLike:
    state = state.clone()
    dtype = state.latent.dtype
    batch, channels, frames, height, width = state.latent.shape

    for cond in conditionings:
        cond_latent = cond.latent
        frame_idx = cond.frame_idx
        strength = cond.strength

        _, cond_channels, cond_frames, cond_height, cond_width = cond_latent.shape
        if (cond_channels, cond_height, cond_width) != (channels, height, width):
            raise ValueError(
                "Conditioning latent spatial shape "
                f"({cond_channels}, {cond_height}, {cond_width}) does not match "
                f"target shape ({channels}, {height}, {width})"
            )
        if frame_idx >= frames:
            raise ValueError(
                f"Frame index {frame_idx} is out of bounds for latent with {frames} frames"
            )

        end_idx = min(frame_idx + cond_frames, frames)
        latent_parts: list[MLXArray] = []
        clean_parts: list[MLXArray] = []
        mask_parts: list[MLXArray] = []

        for index in range(frames):
            if frame_idx <= index < end_idx:
                cond_idx = index - frame_idx
                latent_parts.append(cond_latent[:, :, cond_idx : cond_idx + 1])
                clean_parts.append(cond_latent[:, :, cond_idx : cond_idx + 1])
                mask_parts.append(
                    mx.full((batch, 1, 1, 1, 1), 1.0 - strength, dtype=dtype)
                )
            else:
                latent_parts.append(state.latent[:, :, index : index + 1])
                clean_parts.append(state.clean_latent[:, :, index : index + 1])
                mask_parts.append(state.denoise_mask[:, :, index : index + 1])

        state.latent = mx.concatenate(latent_parts, axis=2)
        state.clean_latent = mx.concatenate(clean_parts, axis=2)
        state.denoise_mask = mx.concatenate(mask_parts, axis=2)

    return state


def apply_denoise_mask(
    denoised: MLXArray,
    clean: MLXArray,
    denoise_mask: MLXArray,
) -> MLXArray:
    one = mx.array(1.0, dtype=denoised.dtype)
    return denoised * denoise_mask + clean * (one - denoise_mask)


def create_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> MLXArray:
    t_coords = np.arange(0, num_frames, dtype=np.int32)
    h_coords = np.arange(0, height, dtype=np.int32)
    w_coords = np.arange(0, width, dtype=np.int32)

    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)
    patch_delta = np.array([1, 1, 1], dtype=np.int32).reshape(3, 1, 1, 1)
    patch_ends = patch_starts + patch_delta

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    scale_factors = np.array(
        [temporal_scale, spatial_scale, spatial_scale], dtype=np.float32
    ).reshape(1, 3, 1, 1)
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    if causal_fix:
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale,
            a_min=0,
            a_max=None,
        )

    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps
    return mx.array(pixel_coords, dtype=mx.float32)


def create_audio_position_grid(
    batch_size: int,
    audio_frames: int,
    *,
    sample_rate: int = _AUDIO_LATENT_SAMPLE_RATE,
    hop_length: int = _AUDIO_HOP_LENGTH,
    downsample_factor: int = _AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    is_causal: bool = True,
) -> MLXArray:
    latent_frame = np.arange(0, audio_frames, dtype=np.float32)
    mel_frame = latent_frame * downsample_factor
    if is_causal:
        mel_frame = np.clip(mel_frame + 1 - downsample_factor, 0, None)
    start_times = mel_frame * hop_length / sample_rate

    next_frame = np.arange(1, audio_frames + 1, dtype=np.float32)
    next_mel = next_frame * downsample_factor
    if is_causal:
        next_mel = np.clip(next_mel + 1 - downsample_factor, 0, None)
    end_times = next_mel * hop_length / sample_rate

    positions = np.stack([start_times, end_times], axis=-1)
    positions = positions[np.newaxis, np.newaxis, :, :]
    positions = np.tile(positions, (batch_size, 1, 1, 1))
    return mx.array(positions, dtype=mx.float32)


def compute_audio_frames(num_video_frames: int, fps: float) -> int:
    duration = num_video_frames / fps
    return round(duration * _AUDIO_LATENTS_PER_SECOND)


def to_denoised(
    noisy: MLXArray,
    velocity: MLXArray,
    sigma: MLXArray | float,
) -> MLXArray:
    original_dtype = noisy.dtype
    noisy_f32 = noisy.astype(mx.float32)
    velocity_f32 = velocity.astype(mx.float32)

    if isinstance(sigma, (int, float)):
        sigma_f32 = mx.array(float(sigma), dtype=mx.float32)
    else:
        sigma_f32 = sigma.astype(mx.float32)
        while sigma_f32.ndim < velocity_f32.ndim:
            sigma_f32 = mx.expand_dims(sigma_f32, axis=-1)

    result = noisy_f32 - sigma_f32 * velocity_f32
    return result.astype(original_dtype)


def load_image(
    image_path: str | Path,
    *,
    height: int | None = None,
    width: int | None = None,
    dtype: mx.Dtype = mx.float32,
) -> MLXArray:
    image = Image.open(image_path).convert("RGB")
    if height is not None and width is not None:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    elif height is not None or width is not None:
        original_width, original_height = image.size
        if height is not None:
            scale = height / original_height
            new_width = (int(original_width * scale) // 32) * 32
            image = image.resize((new_width, height), Image.Resampling.LANCZOS)
        else:
            assert width is not None
            scale = width / original_width
            new_height = (int(original_height * scale) // 32) * 32
            image = image.resize((width, new_height), Image.Resampling.LANCZOS)
    else:
        original_width, original_height = image.size
        new_width = (original_width // 32) * 32
        new_height = (original_height // 32) * 32
        if new_width != original_width or new_height != original_height:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    image_np = np.asarray(image).astype(np.float32) / np.float32(255.0)
    return mx.array(image_np, dtype=dtype)


def prepare_image_for_encoding(
    image: MLXArray,
    target_height: int,
    target_width: int,
    *,
    dtype: mx.Dtype = mx.float32,
) -> MLXArray:
    height, width = image.shape[:2]
    if height != target_height or width != target_width:
        image_np = np.array(image)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        resized = Image.fromarray(image_np).resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        image = mx.array(np.asarray(resized).astype(np.float32) / np.float32(255.0))

    image = image * 2.0 - 1.0
    image = mx.transpose(image, (2, 0, 1))
    image = mx.expand_dims(image, axis=0)
    image = mx.expand_dims(image, axis=2)
    return image.astype(dtype)
