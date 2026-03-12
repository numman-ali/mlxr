from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import mlx.core as mx
from mlx.utils import tree_flatten

from .autoencoder import QwenImageAutoencoderDecoder
from .config import SchedulerConfig
from .lora import apply_lora_deltas
from .scheduler import FlowMatchEulerDiscreteScheduler
from .transformer import QwenImageTransformer2DModel


class _HasParameters(Protocol):
    def parameters(self) -> object: ...


def load_local_transformer(
    component_path: Path,
    *,
    lora_paths: tuple[Path, ...] = (),
    lora_scales: tuple[float, ...] = (),
) -> QwenImageTransformer2DModel:
    from .config import QwenImageTransformerConfig

    config = QwenImageTransformerConfig.from_path(component_path / "config.json")
    model = QwenImageTransformer2DModel(config)
    weights = _align_weight_layouts(
        _load_component_weights(
            component_path,
            alias_map={
                "time_text_embed.timestep_embedder.": "time_text_embed.",
                ".img_mod.1.": ".img_mod.",
                ".txt_mod.1.": ".txt_mod.",
                ".img_mlp.net.0.proj.": ".img_mlp.proj_in.",
                ".img_mlp.net.2.": ".img_mlp.proj_out.",
                ".txt_mlp.net.0.proj.": ".txt_mlp.proj_in.",
                ".txt_mlp.net.2.": ".txt_mlp.proj_out.",
                ".to_out.0.": ".to_out.",
            },
        ),
        expected_shapes=_parameter_shapes(model),
    )
    weights = apply_lora_deltas(
        weights,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        expected_shapes=_parameter_shapes(model),
    )
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def load_local_autoencoder(component_path: Path) -> QwenImageAutoencoderDecoder:
    from .config import AutoencoderConfig

    config = AutoencoderConfig.from_path(component_path / "config.json")
    model = QwenImageAutoencoderDecoder(config)
    weights = _align_weight_layouts(
        _load_component_weights(
            component_path,
            alias_map={
                ".resample.1.": ".resample_conv.",
            },
            include_prefixes=(
                "encoder.",
                "quant_conv.",
                "post_quant_conv.",
                "decoder.",
            ),
        ),
        expected_shapes=_parameter_shapes(model),
    )
    model.load_weights(weights, strict=True)
    model.enable_tiling()
    mx.eval(model.parameters())
    return model


def load_local_scheduler(
    component_path: Path,
    *,
    scheduler_preset: str = "default",
) -> FlowMatchEulerDiscreteScheduler:
    config = SchedulerConfig.from_path(component_path / "scheduler_config.json")
    if scheduler_preset == "lightning":
        config = config.with_overrides(
            base_shift=1.0986122886681098,
            max_shift=1.0986122886681098,
            shift_terminal=None,
            preserve_shift_terminal=False,
        )
    elif scheduler_preset == "turbo_wuli":
        config = config.with_overrides(
            base_shift=0.9162907318741551,
            max_shift=0.9162907318741551,
            shift_terminal=None,
            preserve_shift_terminal=False,
        )
    elif scheduler_preset != "default":
        raise ValueError(
            "Qwen-Image scheduler_preset must be 'default', 'lightning', or 'turbo_wuli'"
        )
    return FlowMatchEulerDiscreteScheduler(config)


def weight_files(component_path: Path) -> tuple[Path, ...]:
    single_file = component_path / "diffusion_pytorch_model.safetensors"
    if single_file.exists():
        return (single_file,)
    single_text_file = component_path / "model.safetensors"
    if single_text_file.exists():
        return (single_text_file,)
    index_file = component_path / "model.safetensors.index.json"
    if index_file.exists():
        raw = json.loads(index_file.read_text("utf-8"))
        weight_map = raw.get("weight_map", {})
        files = sorted({component_path / filename for filename in weight_map.values()})
        return tuple(files)
    diffusion_index = component_path / "diffusion_pytorch_model.safetensors.index.json"
    if diffusion_index.exists():
        raw = json.loads(diffusion_index.read_text("utf-8"))
        weight_map = raw.get("weight_map", {})
        files = sorted({component_path / filename for filename in weight_map.values()})
        return tuple(files)
    raise FileNotFoundError(f"No Qwen-Image weights found under {component_path}")


def _load_component_weights(
    component_path: Path,
    *,
    alias_map: dict[str, str] | None = None,
    include_prefixes: tuple[str, ...] | None = None,
) -> list[tuple[str, mx.array]]:
    weights: list[tuple[str, mx.array]] = []
    for weight_file in weight_files(component_path):
        loaded = mx.load(str(weight_file))
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Expected tensor mapping when loading '{weight_file}'")
        for name, value in loaded.items():
            if include_prefixes is not None and not name.startswith(include_prefixes):
                continue
            aliased_name = name
            if alias_map is not None:
                for old, new in alias_map.items():
                    aliased_name = aliased_name.replace(old, new)
            weights.append((aliased_name, value))
    return weights


def _parameter_shapes(model: _HasParameters) -> dict[str, tuple[int, ...]]:
    flattened = tree_flatten(model.parameters(), destination={})
    if not isinstance(flattened, dict):
        raise RuntimeError("Expected MLX parameter tree to flatten into a mapping")
    return {
        name: tuple(int(dimension) for dimension in value.shape)
        for name, value in flattened.items()
    }


def _align_weight_layouts(
    weights: list[tuple[str, mx.array]],
    *,
    expected_shapes: dict[str, tuple[int, ...]],
) -> list[tuple[str, mx.array]]:
    aligned: list[tuple[str, mx.array]] = []
    for name, value in weights:
        expected_shape = expected_shapes.get(name)
        if (
            expected_shape is not None
            and tuple(int(dimension) for dimension in value.shape) != expected_shape
        ):
            value = _transpose_if_matching(value, expected_shape)
        aligned.append((name, value))
    return aligned


def _transpose_if_matching(
    value: mx.array, expected_shape: tuple[int, ...]
) -> mx.array:
    squeezed = mx.squeeze(value)
    if tuple(int(dimension) for dimension in squeezed.shape) == expected_shape:
        return squeezed
    if value.ndim == 4:
        transposed = value.transpose(0, 2, 3, 1)
        if tuple(int(dimension) for dimension in transposed.shape) == expected_shape:
            return transposed
    if value.ndim == 5:
        transposed = value.transpose(0, 2, 3, 4, 1)
        if tuple(int(dimension) for dimension in transposed.shape) == expected_shape:
            return transposed
    return value
