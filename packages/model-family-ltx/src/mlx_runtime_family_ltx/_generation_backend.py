from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from PIL import Image

from .generation import ConditioningInput, GeneratedVideo, VideoGenerator
from .prompt_encoding import PromptEncodingResult

FloatImage = npt.NDArray[np.float32]


@dataclass(slots=True)
class LTXPreviewVideoGenerator(VideoGenerator):
    checkpoint_path: Path
    spatial_upsampler_path: Path

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

        frames_uint8 = np.asarray((mx.clip(frames, 0.0, 1.0) * 255.0).astype(mx.uint8))
        return GeneratedVideo(
            frames=frames_uint8,
            fps=fps,
            seed=effective_seed,
            backend="mlx_prompt_conditioned_preview",
            conditioning_count=len(conditioning_inputs),
            prompt_signature=_prompt_signature(prompt_context.prompt_text),
        )

    def close(self) -> None:
        return None


def create_video_generator(
    checkpoint_path: Path,
    spatial_upsampler_path: Path,
) -> VideoGenerator:
    return LTXPreviewVideoGenerator(
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
    )


def encode_mp4_video(*, video: GeneratedVideo, output_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("Current LTX mp4 output requires an 'ffmpeg' binary on PATH")
    if video.frames.ndim != 4 or video.frames.shape[-1] != 3:
        raise ValueError(
            "Generated video frames must have shape [frames, height, width, 3]"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames, height, width, _ = video.frames.shape
    if num_frames < 1:
        raise ValueError("Generated video must contain at least one frame")

    result = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(video.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "mpeg4",
            "-q:v",
            "4",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        input=video.frames.tobytes(order="C"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to encode mp4 output: {stderr}")


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
) -> FloatImage:
    with Image.open(payload_path) as image:
        rgb = image.convert("RGB")
        resized = rgb.resize((width, height), resample=Image.Resampling.BICUBIC)
        return np.asarray(resized, dtype=np.float32) / np.float32(255.0)


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
