from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from mlxr.core.runtime import RuntimeHome


def _media_type_for_path(path: Path, kind: str) -> str:
    suffix = path.suffix.lower()
    if kind == "image":
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".ppm": "image/x-portable-pixmap",
            ".webp": "image/webp",
        }.get(suffix, "application/octet-stream")
    if kind == "audio":
        return {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }.get(suffix, "application/octet-stream")
    if kind == "video":
        return {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".m4v": "video/x-m4v",
            ".webm": "video/webm",
            ".avi": "video/x-msvideo",
        }.get(suffix, "application/octet-stream")
    if kind == "lora":
        return {
            ".safetensors": "application/x-safetensors",
        }.get(suffix, "application/octet-stream")
    return "application/octet-stream"


def _default_uds_path() -> Path:
    raw = os.environ.get("MLX_RUNTIME_UDS_PATH")
    if raw:
        return Path(raw)
    runtime_home = RuntimeHome.from_env()
    return runtime_home.temp_dir / "control-plane.sock"


def _file_chunks(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk
