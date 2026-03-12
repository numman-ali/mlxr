from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import mlx.core as mx

_LORA_ALIAS_MAP = {
    "diffusion_model.": "",
    ".img_mod.1.": ".img_mod.",
    ".txt_mod.1.": ".txt_mod.",
    ".img_mlp.net.0.proj.": ".img_mlp.proj_in.",
    ".img_mlp.net.2.": ".img_mlp.proj_out.",
    ".txt_mlp.net.0.proj.": ".txt_mlp.proj_in.",
    ".txt_mlp.net.2.": ".txt_mlp.proj_out.",
    ".to_out.0.": ".to_out.",
}


def apply_lora_deltas(
    base_weights: list[tuple[str, mx.array]],
    *,
    lora_paths: tuple[Path, ...],
    lora_scales: tuple[float, ...],
    expected_shapes: dict[str, tuple[int, ...]],
) -> list[tuple[str, mx.array]]:
    if len(lora_paths) != len(lora_scales):
        raise ValueError("Qwen-Image LoRA paths and scales must have matching lengths")
    if not lora_paths:
        return list(base_weights)

    updated = {name: value for name, value in base_weights}
    for lora_path, scale in zip(lora_paths, lora_scales, strict=True):
        _apply_single_lora(
            updated,
            lora_path=lora_path,
            scale=scale,
            expected_shapes=expected_shapes,
        )
    return list(updated.items())


def _apply_single_lora(
    base_weights: dict[str, mx.array],
    *,
    lora_path: Path,
    scale: float,
    expected_shapes: dict[str, tuple[int, ...]],
) -> None:
    loaded = mx.load(str(lora_path))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Expected tensor mapping when loading '{lora_path}'")

    grouped: dict[str, dict[str, mx.array]] = defaultdict(dict)
    for raw_name, value in loaded.items():
        name = _alias_lora_name(raw_name)
        if name.endswith(".lora_up.weight"):
            grouped[name[: -len(".lora_up.weight")]]["up"] = value
        elif name.endswith(".lora_down.weight"):
            grouped[name[: -len(".lora_down.weight")]]["down"] = value
        elif name.endswith(".lora_A.weight"):
            # Some Qwen LoRA repos publish diffusers-style A/B keys instead of
            # the more explicit down/up names used by LightX2V.
            grouped[name[: -len(".lora_A.weight")]]["down"] = value
        elif name.endswith(".lora_B.weight"):
            grouped[name[: -len(".lora_B.weight")]]["up"] = value
        elif name.endswith(".alpha"):
            grouped[name[: -len(".alpha")]]["alpha"] = value

    if not grouped:
        raise ValueError(f"Qwen-Image LoRA file '{lora_path}' contains no LoRA tensors")

    missing_targets = [
        target for target in grouped if f"{target}.weight" not in base_weights
    ]
    if missing_targets:
        missing = ", ".join(sorted(missing_targets)[:8])
        raise ValueError(
            "Qwen-Image LoRA target weights were not found in the owned transformer: "
            f"{missing}"
        )

    for target, tensors in grouped.items():
        up = tensors.get("up")
        down = tensors.get("down")
        alpha = tensors.get("alpha")
        if up is None or down is None:
            raise ValueError(
                f"Qwen-Image LoRA target '{target}' is missing lora_up or lora_down weights"
            )
        rank = int(down.shape[0])
        if rank <= 0:
            raise ValueError(f"Qwen-Image LoRA target '{target}' has invalid rank")
        alpha_value = float(alpha.item()) if alpha is not None else float(rank)
        delta = mx.matmul(up.astype(mx.float32), down.astype(mx.float32))
        delta = delta * mx.array((alpha_value / rank) * scale, dtype=mx.float32)
        weight_name = f"{target}.weight"
        base = base_weights[weight_name]
        expected_shape = expected_shapes.get(weight_name)
        if (
            expected_shape is not None
            and tuple(int(dimension) for dimension in delta.shape) != expected_shape
        ):
            raise ValueError(
                "Qwen-Image LoRA delta shape mismatch for "
                f"'{weight_name}': got {tuple(int(d) for d in delta.shape)}, "
                f"expected {expected_shape}"
            )
        base_weights[weight_name] = (base.astype(mx.float32) + delta).astype(base.dtype)


def _alias_lora_name(name: str) -> str:
    aliased = name
    for old, new in _LORA_ALIAS_MAP.items():
        aliased = aliased.replace(old, new)
    return aliased
