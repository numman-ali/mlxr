from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from PIL import Image


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
        prompt: str,
        task: str,
        width: int | None,
        height: int | None,
        num_inference_steps: int | None,
        guidance_scale: float | None,
        negative_prompt: str | None,
        seed: int | None,
        image_paths: tuple[Path, ...],
    ) -> GeneratedImage: ...

    def close(self) -> None: ...


def create_image_generator(
    *,
    variant: str,
    model_root: Path,
    task: str,
    quantize_bits: int | None,
    lora_paths: tuple[Path, ...],
    lora_scales: tuple[float, ...],
) -> ImageGenerator:
    from ._generation_backend import (
        create_image_generator as create_backend_image_generator,
    )

    return create_backend_image_generator(
        variant=variant,
        model_root=model_root,
        task=task,
        quantize_bits=quantize_bits,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )


def encode_png_image(image: GeneratedImage, output_path: Path) -> None:
    Image.fromarray(image.pixels, mode="RGB").save(output_path, format="PNG")


def encode_jpg_image(image: GeneratedImage, output_path: Path) -> None:
    Image.fromarray(image.pixels, mode="RGB").save(
        output_path,
        format="JPEG",
        quality=95,
    )
