from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from PIL.Image import Resampling

from ..generation import GeneratedImage, ImageGenerator
from ..prompt_encoding import PromptEncodingResult
from .autoencoder import QwenImageAutoencoderDecoder
from .config import AutoencoderConfig, QwenImageTransformerConfig
from .loading import (
    load_local_autoencoder,
    load_local_scheduler,
    load_local_transformer,
)
from .sampling import (
    denormalize_latents,
    normalize_latents,
    pack_latents,
    unpack_latents,
)
from .scheduler import FlowMatchEulerDiscreteScheduler
from .transformer import QwenImageTransformer2DModel

_DEFAULT_SIZE = 1024
_DEFAULT_STEPS = 50
_DEFAULT_TRUE_CFG = 4.0
_LIGHTNING_DEFAULT_STEPS = 4
_LIGHTNING_DEFAULT_TRUE_CFG = 1.0
_VAE_IMAGE_AREA = 1024 * 1024


@dataclass(slots=True)
class GenerationTrace:
    sample: mx.array
    unpacked_latents: mx.array
    denormalized_latents: mx.array
    decoded: mx.array
    metadata: dict[str, object]
    seed: int
    prompt_signature: str


@dataclass(slots=True)
class _RuntimeImageGenerator(ImageGenerator):
    variant: str
    model_root: Path
    task: str
    scheduler_preset: str
    _transformer: QwenImageTransformer2DModel
    _transformer_config: QwenImageTransformerConfig
    _autoencoder: QwenImageAutoencoderDecoder
    _autoencoder_config: AutoencoderConfig
    _scheduler: FlowMatchEulerDiscreteScheduler

    def generate(
        self,
        *,
        prompt_context: PromptEncodingResult,
        task: str,
        width: int | None,
        height: int | None,
        num_inference_steps: int | None,
        guidance_scale: float | None,
        seed: int | None,
        image_paths: tuple[Path, ...],
    ) -> GeneratedImage:
        trace = self.debug_generate(
            prompt_context=prompt_context,
            task=task,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            image_paths=image_paths,
        )
        frame = trace.decoded[0, :, 0, :, :].transpose(1, 2, 0)
        pixels = mx.clip((frame + 1.0) * 127.5, 0.0, 255.0).astype(mx.uint8)
        mx.eval(pixels)
        return GeneratedImage(
            pixels=np.asarray(pixels),
            seed=trace.seed,
            backend="native_mlx_qwen_image",
            prompt_signature=trace.prompt_signature,
            metadata=trace.metadata,
        )

    def debug_generate(
        self,
        *,
        prompt_context: PromptEncodingResult,
        task: str,
        width: int | None,
        height: int | None,
        num_inference_steps: int | None,
        guidance_scale: float | None,
        seed: int | None,
        image_paths: tuple[Path, ...],
    ) -> GenerationTrace:
        if task not in {"image.generate", "image.edit"}:
            raise ValueError(f"Unsupported Qwen-Image task '{task}'")
        if task == "image.generate" and image_paths:
            raise ValueError("Qwen-Image generate does not accept image_paths")
        if task == "image.edit" and not image_paths:
            raise ValueError("Qwen-Image image.edit requires at least one image")

        resolved_width = _resolved_dimension(
            width,
            multiple_of=self._autoencoder_config.pixel_multiple,
        )
        resolved_height = _resolved_dimension(
            height,
            multiple_of=self._autoencoder_config.pixel_multiple,
        )
        steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else (
                _LIGHTNING_DEFAULT_STEPS
                if self.scheduler_preset == "lightning"
                else _DEFAULT_STEPS
            )
        )
        if steps <= 0:
            raise ValueError("Qwen-Image num_inference_steps must be positive")
        true_cfg_scale = float(
            guidance_scale
            if guidance_scale is not None
            else (
                _LIGHTNING_DEFAULT_TRUE_CFG
                if self.scheduler_preset == "lightning"
                else _DEFAULT_TRUE_CFG
            )
        )
        if true_cfg_scale < 0.0:
            raise ValueError("Qwen-Image guidance_scale must be non-negative")
        resolved_seed = (
            int(seed)
            if seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0])
        )
        mx.random.seed(resolved_seed)

        prompt_embeddings = _as_mx_array(
            prompt_context.prompt_embeddings, dtype=mx.bfloat16
        )
        prompt_mask = _as_mx_array(prompt_context.prompt_attention_mask, dtype=mx.int32)
        negative_prompt_embeddings = _optional_mx_array(
            prompt_context.negative_prompt_embeddings, dtype=mx.bfloat16
        )
        negative_prompt_mask = _optional_mx_array(
            prompt_context.negative_prompt_attention_mask, dtype=mx.int32
        )
        do_true_cfg = (
            true_cfg_scale > 1.0
            and negative_prompt_embeddings is not None
            and negative_prompt_mask is not None
        )

        sample = self._run_denoise(
            task=task,
            image_paths=image_paths,
            prompt_embeddings=prompt_embeddings,
            prompt_mask=prompt_mask,
            negative_prompt_embeddings=negative_prompt_embeddings,
            negative_prompt_mask=negative_prompt_mask,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=true_cfg_scale,
            resolved_width=resolved_width,
            resolved_height=resolved_height,
            steps=steps,
        )

        unpacked = unpack_latents(
            sample,
            height=resolved_height,
            width=resolved_width,
            vae_scale_factor=self._autoencoder_config.scale_factor,
        )
        denormalized = denormalize_latents(
            unpacked.astype(mx.float32), self._autoencoder_config
        )
        decoded = self._autoencoder.decode(denormalized.astype(mx.bfloat16))
        mx.eval(sample, unpacked, denormalized, decoded)
        return GenerationTrace(
            sample=sample,
            unpacked_latents=unpacked,
            denormalized_latents=denormalized,
            decoded=decoded,
            seed=resolved_seed,
            metadata={
                "variant": self.variant,
                "task": task,
                "width": resolved_width,
                "height": resolved_height,
                "num_inference_steps": steps,
                "guidance_scale": true_cfg_scale,
                "true_cfg_used": do_true_cfg,
                "scheduler_preset": self.scheduler_preset,
                "prompt_token_count": prompt_context.token_count,
                "sequence_length": prompt_context.sequence_length,
            },
            prompt_signature=_prompt_signature(prompt_context.prompt_text),
        )

    def close(self) -> None:
        return None

    def _run_denoise(
        self,
        *,
        task: str,
        image_paths: tuple[Path, ...],
        prompt_embeddings: mx.array,
        prompt_mask: mx.array,
        negative_prompt_embeddings: mx.array | None,
        negative_prompt_mask: mx.array | None,
        do_true_cfg: bool,
        true_cfg_scale: float,
        resolved_width: int,
        resolved_height: int,
        steps: int,
    ) -> mx.array:
        latent_height = resolved_height // self._autoencoder_config.scale_factor
        latent_width = resolved_width // self._autoencoder_config.scale_factor
        latent_channels = self._transformer_config.latent_channels
        latents = mx.random.normal(
            (1, 1, latent_channels, latent_height, latent_width),
            dtype=mx.bfloat16,
        )
        packed_latents = pack_latents(latents)
        image_shapes = [(1, latent_height // 2, latent_width // 2)]
        image_latents = None
        if task == "image.edit":
            condition_latents, condition_shapes = _encode_condition_images(
                image_paths=image_paths,
                autoencoder=self._autoencoder,
                autoencoder_config=self._autoencoder_config,
                dtype=mx.bfloat16,
            )
            image_latents = condition_latents
            image_shapes.extend(condition_shapes)
        self._scheduler.set_timesteps(
            num_inference_steps=steps,
            image_sequence_length=int(packed_latents.shape[1]),
        )

        sample = packed_latents
        for index, current_timestep in enumerate(self._scheduler.timesteps):
            current_sigma = self._scheduler.sigmas[index]
            next_sigma = self._scheduler.sigmas[index + 1]
            timestep = mx.full((1,), current_timestep / 1000.0, dtype=sample.dtype)
            latent_model_input = sample
            if image_latents is not None:
                latent_model_input = mx.concatenate([sample, image_latents], axis=1)
            noise_pred = self._transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeddings,
                encoder_hidden_states_mask=prompt_mask,
                timestep=timestep,
                image_shapes=image_shapes,
            )
            noise_pred = noise_pred[:, : int(sample.shape[1]), :]
            if (
                do_true_cfg
                and negative_prompt_embeddings is not None
                and negative_prompt_mask is not None
            ):
                negative_pred = self._transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=negative_prompt_embeddings,
                    encoder_hidden_states_mask=negative_prompt_mask,
                    timestep=timestep,
                    image_shapes=image_shapes,
                )
                negative_pred = negative_pred[:, : int(sample.shape[1]), :]
                combined = negative_pred + true_cfg_scale * (noise_pred - negative_pred)
                noise_pred = _cfg_normalize(noise_pred, combined)
            sample = self._scheduler.step(
                sample=sample,
                model_output=noise_pred,
                current_sigma=current_sigma,
                next_sigma=next_sigma,
            )
        return sample


def create_image_generator(
    *,
    variant: str,
    model_root: Path,
    task: str,
    quantize_bits: int | None,
    scheduler_preset: str,
    lora_paths: tuple[Path, ...],
    lora_scales: tuple[float, ...],
) -> ImageGenerator:
    if quantize_bits is not None:
        raise ValueError("Qwen-Image quantization is not implemented yet")
    if variant not in {
        "qwen-image",
        "qwen-image-2512",
        "qwen-image-edit",
        "qwen-image-edit-2509",
        "qwen-image-edit-2511",
    }:
        raise ValueError(
            "Owned Qwen-Image backend only supports released 2512 generation/edit rows, "
            f"got '{variant}'"
        )
    transformer = load_local_transformer(
        model_root / "transformer",
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )
    transformer_config = QwenImageTransformerConfig.from_path(
        model_root / "transformer" / "config.json"
    )
    autoencoder = load_local_autoencoder(model_root / "vae")
    autoencoder_config = AutoencoderConfig.from_path(model_root / "vae" / "config.json")
    return _RuntimeImageGenerator(
        variant=variant,
        model_root=model_root,
        task=task,
        scheduler_preset=scheduler_preset,
        _transformer=transformer,
        _transformer_config=transformer_config,
        _autoencoder=autoencoder,
        _autoencoder_config=autoencoder_config,
        _scheduler=load_local_scheduler(
            model_root / "scheduler",
            scheduler_preset=scheduler_preset,
        ),
    )


def _as_mx_array(value: object, *, dtype: mx.Dtype) -> mx.array:
    astype = getattr(value, "astype", None)
    if callable(astype):
        try:
            converted = astype(dtype)
        except TypeError:
            converted = None
        if (
            converted is not None
            and hasattr(converted, "shape")
            and hasattr(converted, "dtype")
        ):
            return mx.array(converted, dtype=dtype)
    return mx.array(np.asarray(value), dtype=dtype)


def _optional_mx_array(value: object | None, *, dtype: mx.Dtype) -> mx.array | None:
    if value is None:
        return None
    return _as_mx_array(value, dtype=dtype)


def _encode_condition_images(
    *,
    image_paths: tuple[Path, ...],
    autoencoder: QwenImageAutoencoderDecoder,
    autoencoder_config: AutoencoderConfig,
    dtype: mx.Dtype,
) -> tuple[mx.array, list[tuple[int, int, int]]]:
    packed_images: list[mx.array] = []
    image_shapes: list[tuple[int, int, int]] = []
    for image_path in image_paths:
        pixels, latent_height, latent_width = _load_vae_condition_image(image_path)
        encoded = autoencoder.encode(mx.array(pixels, dtype=dtype))
        normalized = normalize_latents(encoded.astype(mx.float32), autoencoder_config)
        # The VAE encodes to [batch, channels, frames, height, width], while the
        # DiT packer expects [batch, frames, channels, height, width].
        packed = pack_latents(normalized.astype(dtype).transpose(0, 2, 1, 3, 4))
        packed_images.append(packed)
        image_shapes.append((1, latent_height // 2, latent_width // 2))
    return mx.concatenate(packed_images, axis=1), image_shapes


def _load_vae_condition_image(image_path: Path) -> tuple[np.ndarray, int, int]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resized_width, resized_height = _calculate_dimensions(
        _VAE_IMAGE_AREA, width / height
    )
    image = image.resize((resized_width, resized_height), resample=Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array * 2.0 - 1.0
    channels_first = np.transpose(array, (2, 0, 1))
    return channels_first[None, :, None, :, :], resized_height // 8, resized_width // 8


def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = (target_area * ratio) ** 0.5
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def _cfg_normalize(
    conditional: mx.array,
    combined: mx.array,
) -> mx.array:
    conditional_norm = mx.sqrt(
        mx.sum(mx.square(conditional.astype(mx.float32)), axis=-1, keepdims=True)
    )
    combined_norm = mx.maximum(
        mx.sqrt(mx.sum(mx.square(combined.astype(mx.float32)), axis=-1, keepdims=True)),
        mx.array(1.0e-6, dtype=mx.float32),
    )
    return combined * (conditional_norm / combined_norm)


def _resolved_dimension(value: int | None, *, multiple_of: int) -> int:
    resolved = int(value) if value is not None else _DEFAULT_SIZE
    if resolved % multiple_of != 0:
        raise ValueError(f"Qwen-Image dimensions must be divisible by {multiple_of}")
    return resolved


def _prompt_signature(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
