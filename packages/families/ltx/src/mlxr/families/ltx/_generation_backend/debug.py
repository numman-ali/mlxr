from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from PIL import Image

from .types import MLXArray, _PaddedShape, _ReferenceImports, _VAEEncoder


def _debug_stage_dump_dir() -> Path | None:
    raw = os.environ.get("MLXR_LTX_DEBUG_STAGE_DUMPS_DIR")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def _debug_progress_enabled() -> bool:
    return os.environ.get("MLXR_LTX_DEBUG_PROGRESS") == "1"


def _debug_progress(message: str) -> None:
    if _debug_progress_enabled():
        active_gb = mx.get_active_memory() / (1024**3)
        peak_gb = mx.get_peak_memory() / (1024**3)
        cache_gb = mx.get_cache_memory() / (1024**3)
        print(
            (
                f"[ltx] {message} "
                f"active_gb={active_gb:.2f} "
                f"peak_gb={peak_gb:.2f} "
                f"cache_gb={cache_gb:.2f}"
            ),
            flush=True,
        )


def _debug_trace_enabled() -> bool:
    return os.environ.get("MLXR_LTX_DEBUG_TRACE") == "1"


def _debug_trace_sync_enabled() -> bool:
    return os.environ.get("MLXR_LTX_DEBUG_TRACE_SYNC") == "1"


def _latent_stats(latents: MLXArray) -> dict[str, object]:
    latents_f32 = latents.astype(mx.float32)
    return {
        "shape": [int(size) for size in latents.shape],
        "mean": float(mx.mean(latents_f32).item()),
        "std": float(mx.std(latents_f32).item()),
        "min": float(mx.min(latents_f32).item()),
        "max": float(mx.max(latents_f32).item()),
    }


def _looks_like_metal_oom(exc: BaseException) -> bool:
    message = str(exc)
    needles = (
        "out of memory",
        "Out of memory",
        "OOM",
        "failed to allocate",
        "kIOGPU",
        "Command buffer execution failed",
        "Invalid Resource",
        "MTLCommandBufferError",
        "[METAL]",
    )
    return any(needle in message for needle in needles)
