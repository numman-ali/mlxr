from __future__ import annotations

from pathlib import Path
from typing import Protocol

import mlx.core as mx
from mlx.utils import tree_flatten

from .autoencoder import AutoencoderKLFlux2
from .config import AutoencoderConfig, Flux2TransformerConfig, SchedulerConfig
from .scheduler import FlowMatchEulerDiscreteScheduler
from .transformer import Flux2Transformer2DModel


class _HasParameters(Protocol):
    def parameters(self) -> object: ...


def load_local_flux2_transformer(component_path: Path) -> Flux2Transformer2DModel:
    config = Flux2TransformerConfig.from_path(component_path / "config.json")
    model = Flux2Transformer2DModel(config)
    weights = _load_component_weights(
        component_path,
        alias_map={
            ".to_out.0.": ".to_out.",
        },
    )
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def load_local_autoencoder(component_path: Path) -> AutoencoderKLFlux2:
    config = AutoencoderConfig.from_path(component_path / "config.json")
    model = AutoencoderKLFlux2(config)
    weights = _align_autoencoder_weight_layouts(
        _load_component_weights(
            component_path,
            alias_map={
                ".to_out.0.": ".to_out.",
            },
        ),
        expected_shapes=_parameter_shapes(model),
    )
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def load_local_scheduler(component_path: Path) -> FlowMatchEulerDiscreteScheduler:
    config = SchedulerConfig.from_path(component_path / "scheduler_config.json")
    return FlowMatchEulerDiscreteScheduler(config)


def _load_component_weights(
    component_path: Path,
    *,
    alias_map: dict[str, str] | None = None,
) -> list[tuple[str, mx.array]]:
    weights: list[tuple[str, mx.array]] = []
    for weight_file in _weight_files(component_path):
        loaded = mx.load(str(weight_file))
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Expected tensor mapping when loading '{weight_file}'")
        for name, value in loaded.items():
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


def _align_autoencoder_weight_layouts(
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
            and value.ndim == 4
            and tuple(int(dimension) for dimension in value.transpose(0, 2, 3, 1).shape)
            == expected_shape
        ):
            value = value.transpose(0, 2, 3, 1)
        aligned.append((name, value))
    return aligned


def _weight_files(component_path: Path) -> tuple[Path, ...]:
    single_file = component_path / "diffusion_pytorch_model.safetensors"
    if single_file.exists():
        return (single_file,)
    single_text_file = component_path / "model.safetensors"
    if single_text_file.exists():
        return (single_text_file,)
    index_file = component_path / "model.safetensors.index.json"
    if index_file.exists():
        import json

        raw = json.loads(index_file.read_text("utf-8"))
        weight_map = raw.get("weight_map", {})
        files = sorted({component_path / filename for filename in weight_map.values()})
        return tuple(files)
    diffusion_index = component_path / "diffusion_pytorch_model.safetensors.index.json"
    if diffusion_index.exists():
        import json

        raw = json.loads(diffusion_index.read_text("utf-8"))
        weight_map = raw.get("weight_map", {})
        files = sorted({component_path / filename for filename in weight_map.values()})
        return tuple(files)
    raise FileNotFoundError(f"No FLUX.2 weights found under {component_path}")
