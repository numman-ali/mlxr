from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Protocol

import mlx.core as mx
from safetensors import safe_open

from .types import MLXArray


class _SafeOpenHandle(Protocol):
    def __enter__(self) -> "_SafeOpenHandle": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None: ...
    def metadata(self) -> dict[str, str] | None: ...


def apply_lora_deltas(
    base_weights: dict[str, MLXArray],
    *,
    lora_paths: tuple[Path, ...],
    lora_scales: tuple[float, ...],
) -> dict[str, MLXArray]:
    if len(lora_paths) != len(lora_scales):
        raise ValueError("LTX LoRA paths and scales must have matching lengths")
    if not lora_paths:
        return dict(base_weights)

    updated = dict(base_weights)
    expected_shapes = {
        name: tuple(int(dimension) for dimension in value.shape)
        for name, value in updated.items()
    }
    for lora_path, scale in zip(lora_paths, lora_scales, strict=True):
        _apply_single_lora(
            updated,
            expected_shapes=expected_shapes,
            lora_path=lora_path,
            scale=scale,
        )
    return updated


def reference_downscale_factor_for_loras(
    lora_paths: tuple[Path, ...],
) -> int:
    scale = 1
    for lora_path in lora_paths:
        current = _reference_downscale_factor(lora_path)
        if current == 1:
            continue
        if scale not in {1, current}:
            raise ValueError(
                "LTX IC-LoRA inputs cannot mix different reference_downscale_factor "
                f"values; already have {scale}, but '{lora_path}' specifies {current}"
            )
        scale = current
    return scale


def _apply_single_lora(
    base_weights: dict[str, MLXArray],
    *,
    expected_shapes: dict[str, tuple[int, ...]],
    lora_path: Path,
    scale: float,
) -> None:
    loaded = mx.load(str(lora_path))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Expected tensor mapping when loading '{lora_path}'")

    metadata = _metadata(lora_path)
    global_alpha = _optional_float(metadata.get("lora_alpha"))
    global_rank = _optional_float(metadata.get("lora_rank"))

    grouped: dict[str, dict[str, MLXArray]] = defaultdict(dict)
    for raw_name, value in loaded.items():
        name = _alias_lora_name(raw_name)
        if name.endswith(".lora_A.weight"):
            grouped[name[: -len(".lora_A.weight")]]["down"] = value
        elif name.endswith(".lora_B.weight"):
            grouped[name[: -len(".lora_B.weight")]]["up"] = value
        elif name.endswith(".alpha"):
            grouped[name[: -len(".alpha")]]["alpha"] = value

    if not grouped:
        raise ValueError(f"LTX LoRA file '{lora_path}' contains no LoRA tensors")

    missing_targets = [
        target
        for target in grouped
        if _weight_name_for_target(target) not in base_weights
    ]
    if missing_targets:
        sample = ", ".join(sorted(missing_targets)[:8])
        raise ValueError(
            f"LTX LoRA target weights were not found in the owned transformer: {sample}"
        )

    for target, tensors in grouped.items():
        up = tensors.get("up")
        down = tensors.get("down")
        alpha = tensors.get("alpha")
        if up is None or down is None:
            raise ValueError(
                f"LTX LoRA target '{target}' is missing lora_A or lora_B weights"
            )
        rank = int(down.shape[0])
        if rank <= 0:
            raise ValueError(f"LTX LoRA target '{target}' has invalid rank")
        alpha_value = (
            float(alpha.item())
            if alpha is not None
            else global_alpha
            if global_alpha is not None
            else global_rank
            if global_rank is not None
            else float(rank)
        )
        delta = mx.matmul(up.astype(mx.float32), down.astype(mx.float32))
        delta = delta * mx.array((alpha_value / rank) * scale, dtype=mx.float32)
        weight_name = _weight_name_for_target(target)
        expected_shape = expected_shapes.get(weight_name)
        if (
            expected_shape is not None
            and tuple(int(dimension) for dimension in delta.shape) != expected_shape
        ):
            raise ValueError(
                "LTX LoRA delta shape mismatch for "
                f"'{weight_name}': got {tuple(int(dimension) for dimension in delta.shape)}, "
                f"expected {expected_shape}"
            )
        base = base_weights[weight_name]
        base_weights[weight_name] = (base.astype(mx.float32) + delta).astype(base.dtype)


def _alias_lora_name(name: str) -> str:
    if name.startswith("diffusion_model."):
        return f"model.{name}"
    return name


def _weight_name_for_target(target: str) -> str:
    return f"{target}.weight"


def _metadata(lora_path: Path) -> dict[str, str]:
    with _safe_open_numpy(lora_path) as handle:
        return handle.metadata() or {}


def read_reference_downscale_factor(lora_path: Path) -> int:
    metadata = _metadata(lora_path)
    raw_value = metadata.get("reference_downscale_factor")
    if raw_value is None:
        return 1
    try:
        scale = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "LTX IC-LoRA metadata reference_downscale_factor must be an integer"
        ) from exc
    if scale < 1:
        raise ValueError("LTX IC-LoRA metadata reference_downscale_factor must be >= 1")
    return scale


def _safe_open_numpy(path: Path) -> _SafeOpenHandle:
    return safe_open(str(path), framework="numpy")  # type: ignore[no-untyped-call]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return None


def _reference_downscale_factor(lora_path: Path) -> int:
    metadata = _metadata(lora_path)
    raw_value = metadata.get("reference_downscale_factor", "1")
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "LTX LoRA metadata reference_downscale_factor must be an integer when present"
        ) from exc
