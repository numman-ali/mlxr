from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .prompt_encoding import PromptEncodingResult


@dataclass(slots=True)
class GeneratedImage:
    pixels: npt.NDArray[np.uint8]
    seed: int
    backend: str
    prompt_signature: str
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class ImageGenerator(Protocol):
    def generate(
        self,
        *,
        prompt_context: PromptEncodingResult,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        cfg_normalization: float = 0.0,
        cfg_truncation: float = 1.0,
        seed: int | None = None,
    ) -> GeneratedImage: ...

    def close(self) -> None: ...


def create_image_generator(
    *,
    transformer_path: Path,
    vae_path: Path,
    scheduler_path: Path,
) -> ImageGenerator:
    from ._generation_backend import (
        create_image_generator as create_backend_image_generator,
    )

    return create_backend_image_generator(
        transformer_path=transformer_path,
        vae_path=vae_path,
        scheduler_path=scheduler_path,
    )


def encode_png_image(image: GeneratedImage, output_path: Path) -> None:
    backend_module = import_module("mlxr.families.z_image._generation_backend")
    encode_backend_png_image = backend_module.encode_png_image
    encode_backend_png_image(image=image, output_path=output_path)


def encode_jpg_image(image: GeneratedImage, output_path: Path) -> None:
    backend_module = import_module("mlxr.families.z_image._generation_backend")
    encode_backend_jpg_image = backend_module.encode_jpg_image
    encode_backend_jpg_image(image=image, output_path=output_path)
