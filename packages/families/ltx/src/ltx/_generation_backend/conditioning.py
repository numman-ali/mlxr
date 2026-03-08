from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import TypeGuard

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from PIL import Image

from ..generation import ConditioningInput
from ..prompt_encoding import PromptEncodingResult
from .types import MLXArray, _PaddedShape, _ReferenceImports, _VAEEncoder


def _decode_conditioning_audio_file(
    audio_path: Path,
    *,
    sample_rate: int,
    start_time_seconds: float,
    max_duration_seconds: float | None,
) -> tuple[npt.NDArray[np.float32], int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
    ]
    if start_time_seconds > 0.0:
        command.extend(["-ss", str(start_time_seconds)])
    command.extend(["-i", str(audio_path)])
    if max_duration_seconds is not None:
        command.extend(["-t", str(max_duration_seconds)])
    command.extend(
        [
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode conditioning audio: {stderr}")
    waveform = np.frombuffer(result.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise RuntimeError(
            f"Conditioning audio '{audio_path}' decoded to an empty waveform"
        )
    channels = 2
    usable = waveform.size - (waveform.size % channels)
    if usable == 0:
        raise RuntimeError(
            f"Conditioning audio '{audio_path}' did not decode to stereo PCM frames"
        )
    waveform = waveform[:usable].reshape(-1, channels)
    return waveform.astype(np.float32), sample_rate


def _fit_audio_latents(audio_latents: MLXArray, *, target_frames: int) -> MLXArray:
    current_frames = int(audio_latents.shape[2])
    if current_frames == target_frames:
        return audio_latents
    if current_frames > target_frames:
        return audio_latents[:, :, :target_frames, :]
    pad_frames = target_frames - current_frames
    padding = mx.zeros(
        (
            int(audio_latents.shape[0]),
            int(audio_latents.shape[1]),
            pad_frames,
            int(audio_latents.shape[3]),
        ),
        dtype=audio_latents.dtype,
    )
    return mx.concatenate([audio_latents, padding], axis=2)


def _emit_debug_frame_snapshot(
    *,
    debug_dir: Path,
    stage_name: str,
    frames_uint8: npt.NDArray[np.uint8],
    metadata: dict[str, object],
) -> None:
    Image.fromarray(frames_uint8[0]).save(debug_dir / f"{stage_name}_frame_0001.png")
    (debug_dir / f"{stage_name}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _encode_conditioning_latent(
    *,
    imports: _ReferenceImports,
    vae_encoder: _VAEEncoder,
    payload_path: Path,
    width: int,
    height: int,
    dtype: mx.Dtype,
) -> MLXArray:
    image = imports.load_image(
        str(payload_path), height=height, width=width, dtype=dtype
    )
    encoded = imports.prepare_image_for_encoding(image, height, width, dtype=dtype)
    latent = vae_encoder(encoded)
    mx.eval(latent)
    return latent


def _resolve_padded_shape(
    *, width: int, height: int, divisor: int = 64
) -> _PaddedShape:
    pad_w = (divisor - (width % divisor)) % divisor
    pad_h = (divisor - (height % divisor)) % divisor
    return _PaddedShape(
        output_width=width,
        output_height=height,
        internal_width=width + pad_w,
        internal_height=height + pad_h,
        crop_top=pad_h // 2,
        crop_left=pad_w // 2,
    )


def _half_resolution_padded_shape(
    *,
    padded_shape: _PaddedShape,
    width: int,
    height: int,
) -> _PaddedShape:
    output_width = max(1, width // 2)
    output_height = max(1, height // 2)
    internal_width = max(1, padded_shape.internal_width // 2)
    internal_height = max(1, padded_shape.internal_height // 2)
    return _PaddedShape(
        output_width=output_width,
        output_height=output_height,
        internal_width=internal_width,
        internal_height=internal_height,
        crop_top=padded_shape.crop_top // 2,
        crop_left=padded_shape.crop_left // 2,
    )


def _resolve_latent_frame_index(
    *, frame_index: int, num_frames: int, latent_frames: int
) -> int:
    if frame_index < latent_frames:
        return frame_index
    if num_frames <= 1 or latent_frames <= 1:
        return 0
    scaled = int((frame_index / (num_frames - 1) * (latent_frames - 1)) + 0.5)
    return int(max(0, min(latent_frames - 1, scaled)))


def _require_video_context(prompt_context: PromptEncodingResult) -> MLXArray:
    if not _looks_like_mlx_array(prompt_context.video_context):
        raise ValueError("LTX real generation requires an MLX video prompt context")
    return prompt_context.video_context


def _require_audio_context(prompt_context: PromptEncodingResult) -> MLXArray:
    if not _looks_like_mlx_array(prompt_context.audio_context):
        raise ValueError("LTX real generation requires an MLX audio prompt context")
    return prompt_context.audio_context


def _optional_negative_video_context(
    prompt_context: PromptEncodingResult,
) -> MLXArray | None:
    if not _looks_like_mlx_array(prompt_context.negative_video_context):
        return None
    return prompt_context.negative_video_context


def _optional_negative_audio_context(
    prompt_context: PromptEncodingResult,
) -> MLXArray | None:
    if not _looks_like_mlx_array(prompt_context.negative_audio_context):
        return None
    return prompt_context.negative_audio_context


def _attention_mask(prompt_context: PromptEncodingResult) -> MLXArray | None:
    if not _looks_like_mlx_array(prompt_context.attention_mask):
        return None
    if int(mx.sum(prompt_context.attention_mask).item()) == int(
        prompt_context.attention_mask.size
    ):
        return None
    return prompt_context.attention_mask


def _context_width(video_context: object) -> int:
    if not _looks_like_mlx_array(video_context):
        raise ValueError("LTX real generation requires an MLX video prompt context")
    return int(video_context.shape[-1])


def _looks_like_mlx_array(value: object) -> TypeGuard[MLXArray]:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _prompt_context_dtype(context: MLXArray) -> mx.Dtype:
    return getattr(context, "dtype", mx.bfloat16)


def _effective_seed(
    *,
    prompt_context: PromptEncodingResult,
    checkpoint_path: Path,
    spatial_upsampler_path: Path,
    seed: int | None,
) -> int:
    payload = "|".join(
        (
            prompt_context.prompt_text,
            str(prompt_context.token_count),
            str(prompt_context.sequence_length),
            checkpoint_path.name,
            str(checkpoint_path.stat().st_size),
            spatial_upsampler_path.name,
            str(spatial_upsampler_path.stat().st_size),
            str(seed if seed is not None else "auto"),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _prompt_signature(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]


def _base_frames(
    *,
    width: int,
    height: int,
    num_frames: int,
    prompt_context: PromptEncodingResult,
    seed: int,
) -> mx.array:
    prompt_scale = max(prompt_context.token_count, 1) / max(
        prompt_context.sequence_length, 1
    )
    seed_scale = ((seed % 997) + 1) / 997.0
    width_grid = mx.linspace(0.0, 1.0, width).reshape(1, 1, width, 1)
    height_grid = mx.linspace(0.0, 1.0, height).reshape(1, height, 1, 1)
    frame_grid = mx.linspace(0.0, 1.0, num_frames).reshape(num_frames, 1, 1, 1)
    tau = math.tau
    red = 0.52 + 0.48 * mx.sin(
        tau
        * (
            width_grid * (2.8 + prompt_scale * 4.2)
            + height_grid * 0.7
            + frame_grid * (1.6 + seed_scale * 2.1)
        )
    )
    green = 0.48 + 0.45 * mx.cos(
        tau
        * (
            width_grid * 0.9
            + height_grid * (2.4 + prompt_scale * 3.6)
            + frame_grid * (1.2 + seed_scale * 1.4)
        )
    )
    blue = 0.51 + 0.44 * mx.sin(
        tau
        * (
            width_grid * (1.2 + seed_scale * 2.7)
            + height_grid * (1.5 + prompt_scale * 2.0)
            + frame_grid * 0.75
        )
    )
    base = mx.concatenate([red, green, blue], axis=-1)
    noise = mx.random.uniform(shape=base.shape, low=-0.08, high=0.08)
    return base + noise


def _apply_conditioning(
    *,
    frames: mx.array,
    conditioning_input: ConditioningInput,
    width: int,
    height: int,
    num_frames: int,
) -> mx.array:
    image = _decode_conditioning_image(
        payload_path=conditioning_input.payload_path,
        width=width,
        height=height,
    )
    weights = np.zeros((num_frames, 1, 1, 1), dtype=np.float32)
    for offset, factor in ((0, 1.0), (-1, 0.35), (1, 0.35)):
        target_index = conditioning_input.frame_index + offset
        if 0 <= target_index < num_frames:
            blended_strength = float(conditioning_input.strength) * factor
            weights[target_index, 0, 0, 0] = max(
                weights[target_index, 0, 0, 0], blended_strength
            )
    image_tensor = mx.array(image).reshape(1, height, width, 3)
    weight_tensor = mx.array(weights)
    return frames * (1.0 - weight_tensor) + image_tensor * weight_tensor


def _decode_conditioning_image(
    *, payload_path: Path, width: int, height: int
) -> npt.NDArray[np.float32]:
    with Image.open(payload_path) as image:
        rgb = image.convert("RGB")
        resized = rgb.resize((width, height), resample=Image.Resampling.BICUBIC)
        return np.asarray(resized, dtype=np.float32) / np.float32(255.0)
