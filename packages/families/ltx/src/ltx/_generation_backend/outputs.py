from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from ..generation import GeneratedVideo
from .types import _PaddedShape


def encode_mp4_video(*, video: GeneratedVideo, output_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("Current LTX mp4 output requires an 'ffmpeg' binary on PATH")
    if video.audio_waveform is not None and video.audio_sample_rate is None:
        raise ValueError("Generated audio requires audio_sample_rate metadata")
    if video.frames.ndim != 4 or video.frames.shape[-1] != 3:
        raise ValueError(
            "Generated video frames must have shape [frames, height, width, 3]"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if video.audio_waveform is None:
        _encode_video_only_mp4(
            video=video, output_path=output_path, ffmpeg_path=ffmpeg_path
        )
        return

    with tempfile.TemporaryDirectory(prefix="mlxr-ltx-audio-export-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        video_only_path = tmp_root / "video-only.mp4"
        audio_path = tmp_root / "audio.wav"
        _encode_video_only_mp4(
            video=video,
            output_path=video_only_path,
            ffmpeg_path=ffmpeg_path,
        )
        encode_wav_audio(video=video, output_path=audio_path)
        _mux_mp4_with_audio(
            ffmpeg_path=ffmpeg_path,
            video_path=video_only_path,
            audio_path=audio_path,
            output_path=output_path,
            audio_sample_rate=video.audio_sample_rate,
        )


def encode_wav_audio(*, video: GeneratedVideo, output_path: Path) -> None:
    if video.audio_waveform is None or video.audio_sample_rate is None:
        raise ValueError("Generated video does not contain decodable audio output")

    audio = _normalized_audio_waveform(video.audio_waveform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    audio_int16 = (audio * 32767.0).astype(np.int16)

    import wave

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(video.audio_sample_rate))
        wav_file.writeframes(audio_int16.tobytes(order="C"))


def _encode_video_only_mp4(
    *,
    video: GeneratedVideo,
    output_path: Path,
    ffmpeg_path: str,
) -> None:
    num_frames, height, width, _ = video.frames.shape
    if num_frames < 1:
        raise ValueError("Generated video must contain at least one frame")
    encoder_args = _preferred_h264_encoder_args(ffmpeg_path)

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
            *encoder_args,
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        input=video.frames.tobytes(order="C"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to encode mp4 output: {stderr}")


@cache
def _available_ffmpeg_encoders(ffmpeg_path: str) -> frozenset[str]:
    result = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-encoders"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"ffmpeg failed to list encoders: {stderr}")

    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return frozenset(encoders)


def _preferred_h264_encoder_args(ffmpeg_path: str) -> list[str]:
    encoders = _available_ffmpeg_encoders(ffmpeg_path)
    if "h264_videotoolbox" in encoders:
        return [
            "-c:v",
            "h264_videotoolbox",
            "-allow_sw",
            "1",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "avc1",
        ]
    if "libx264" in encoders:
        return [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "avc1",
        ]
    raise RuntimeError(
        "Current LTX mp4 output requires an H.264 ffmpeg encoder "
        "('h264_videotoolbox' or 'libx264')"
    )


def _mux_mp4_with_audio(
    *,
    ffmpeg_path: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_sample_rate: int | None,
) -> None:
    if audio_sample_rate is None:
        raise ValueError("Generated audio requires audio_sample_rate metadata")

    result = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-ar",
            str(audio_sample_rate),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to mux mp4 audio output: {stderr}")


def _decode_to_uint8_frames(
    decoded_video: mx.array,
    *,
    padded_shape: _PaddedShape,
) -> npt.NDArray[np.uint8]:
    video = mx.squeeze(decoded_video, axis=0)
    video = mx.transpose(video, (1, 2, 3, 0))
    video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
    video = (video * 255.0).astype(mx.uint8)
    frames = np.asarray(video)
    if (
        padded_shape.internal_width != padded_shape.output_width
        or padded_shape.internal_height != padded_shape.output_height
    ):
        top = padded_shape.crop_top
        left = padded_shape.crop_left
        frames = frames[
            :,
            top : top + padded_shape.output_height,
            left : left + padded_shape.output_width,
            :,
        ]
    return frames


def _audio_waveform_to_numpy(audio_waveform: object) -> npt.NDArray[np.float32]:
    waveform = np.asarray(audio_waveform, dtype=np.float32)
    if (
        waveform.ndim == 2
        and waveform.shape[0] in {1, 2}
        and waveform.shape[1] > waveform.shape[0]
    ):
        waveform = np.transpose(waveform, (1, 0))
    if waveform.ndim not in {1, 2}:
        raise ValueError(
            "Decoded LTX audio must be mono [samples] or stereo [samples, channels]"
        )
    return _normalized_audio_waveform(waveform)


def _normalized_audio_waveform(
    waveform: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    audio = np.asarray(waveform, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32, copy=False)
