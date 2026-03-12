from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .._sampling import (
    prepare_latent_images,
    prepare_text_ids,
    unpack_latent_images,
)
from ..generation import GeneratedImage, ImageGenerator
from ..prompt_encoding import (
    PromptEncoder,
    PromptEncodingResult,
    create_prompt_encoder,
)
from .autoencoder import AutoencoderKLFlux2
from .constants import _LATENT_CHANNELS, _LATENT_DOWNSAMPLE
from .images import (
    _conditioning_reference_limit,
    _prompt_signature,
    _reference_tensor,
    _resolved_dimensions,
)
from .loading import (
    load_local_autoencoder,
    load_local_flux2_transformer,
    load_local_scheduler,
)
from .scheduler import FlowMatchEulerDiscreteScheduler
from .transformer import Flux2Transformer2DModel


def _distilled_variant(variant: str) -> bool:
    return "klein" in variant and "base" not in variant


def _base_variant(variant: str) -> bool:
    return "klein-base" in variant


@dataclass(slots=True)
class _RuntimeImageGenerator(ImageGenerator):
    variant: str
    model_root: Path
    task: str
    _prompt_encoder: PromptEncoder
    _transformer: Flux2Transformer2DModel
    _vae: AutoencoderKLFlux2
    _scheduler: FlowMatchEulerDiscreteScheduler

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
    ) -> GeneratedImage:
        if negative_prompt is not None and negative_prompt.strip():
            raise ValueError(
                "FLUX.2 negative_prompt is not implemented in the owned MLX backend yet"
            )
        if task not in {"image.generate", "image.edit"}:
            raise ValueError(f"Unsupported FLUX.2 task '{task}'")
        if task == "image.edit" and not image_paths:
            raise ValueError("FLUX.2 image.edit requires at least one reference image")
        resolved_width, resolved_height = _resolved_dimensions(
            width=width,
            height=height,
            image_paths=image_paths,
        )
        default_steps = 50 if _base_variant(self.variant) else 4
        default_guidance = 4.0 if _base_variant(self.variant) else 1.0
        steps = int(
            num_inference_steps if num_inference_steps is not None else default_steps
        )
        guidance = float(
            guidance_scale if guidance_scale is not None else default_guidance
        )
        if _distilled_variant(self.variant):
            # Distilled klein rows must stay on the upstream 4-step / guidance 1.0
            # recipe instead of silently accepting base/dev-style settings.
            if steps != 4:
                raise ValueError(
                    f"{self.variant} is a distilled FLUX.2 row and requires num_inference_steps=4"
                )
            if abs(guidance - 1.0) > 1.0e-6:
                raise ValueError(
                    f"{self.variant} is a distilled FLUX.2 row and requires guidance_scale=1.0"
                )
        resolved_seed = (
            seed
            if seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0])
        )
        mx.random.seed(resolved_seed)

        prompt_context = self._prompt_encoder.encode(prompt, max_length=512)
        prompt_embeddings, text_ids = self._text_conditioning(prompt_context)

        latent_height = resolved_height // _LATENT_DOWNSAMPLE
        latent_width = resolved_width // _LATENT_DOWNSAMPLE
        latents = mx.random.normal(
            (1, latent_height, latent_width, _LATENT_CHANNELS),
            dtype=mx.bfloat16,
        )
        latent_tokens, latent_ids = prepare_latent_images(latents)
        reference_tokens, reference_ids = self._encode_reference_sequence(image_paths)

        timesteps = self._scheduler.timesteps(
            num_inference_steps=steps,
            image_sequence_length=int(latent_tokens.shape[1]),
        )
        guidance_vector = mx.full((1,), guidance, dtype=mx.bfloat16)
        sample = latent_tokens
        for current_timestep, next_timestep in zip(
            timesteps[:-1],
            timesteps[1:],
            strict=True,
        ):
            timestep = mx.full((1,), current_timestep, dtype=mx.bfloat16)
            prediction = self._predict_step(
                sample=sample,
                latent_ids=latent_ids,
                prompt_embeddings=prompt_embeddings,
                text_ids=text_ids,
                timestep=timestep,
                guidance=guidance,
                guidance_vector=guidance_vector,
                reference_tokens=reference_tokens,
                reference_ids=reference_ids,
            )
            sample = self._scheduler.step(
                sample=sample,
                model_output=prediction.astype(sample.dtype),
                timestep=current_timestep,
                next_timestep=next_timestep,
            )

        decoded_latents = unpack_latent_images(
            sample,
            latent_height=latent_height,
            latent_width=latent_width,
            channels=_LATENT_CHANNELS,
        )
        decoded_image = self._vae.decode(decoded_latents.astype(mx.float32))
        pixels = mx.clip((decoded_image[0] + 1.0) * 127.5, 0.0, 255.0).astype(mx.uint8)
        mx.eval(pixels)
        return GeneratedImage(
            pixels=np.asarray(pixels),
            seed=resolved_seed,
            backend="native_mlx_flux2",
            prompt_signature=_prompt_signature(prompt_context.prompt_text),
            metadata={
                "variant": self.variant,
                "task": task,
                "width": resolved_width,
                "height": resolved_height,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "prompt_token_count": prompt_context.token_count,
                "conditioning_reference_count": len(image_paths),
                "conditioning_sequence_length": (
                    int(reference_tokens.shape[1])
                    if reference_tokens is not None
                    else 0
                ),
            },
        )

    def _text_conditioning(
        self,
        prompt_context: object,
    ) -> tuple[mx.array, mx.array]:
        if not isinstance(prompt_context, PromptEncodingResult):
            raise ValueError("FLUX.2 prompt encoder returned an unexpected result")
        if _base_variant(self.variant):
            empty_context, prompt_context = self._prompt_encoder.encode_many(
                ("", prompt_context.prompt_text),
                max_length=512,
            )
            prompt_embeddings = mx.concatenate(
                [
                    empty_context.prompt_embeddings.astype(mx.bfloat16),
                    prompt_context.prompt_embeddings.astype(mx.bfloat16),
                ],
                axis=0,
            )
            text_ids = prepare_text_ids(2, int(prompt_embeddings.shape[1]))
            return prompt_embeddings, text_ids
        prompt_embeddings = prompt_context.prompt_embeddings.astype(mx.bfloat16)
        text_ids = prepare_text_ids(1, int(prompt_embeddings.shape[1]))
        return prompt_embeddings, text_ids

    def _predict_step(
        self,
        *,
        sample: mx.array,
        latent_ids: mx.array,
        prompt_embeddings: mx.array,
        text_ids: mx.array,
        timestep: mx.array,
        guidance: float,
        guidance_vector: mx.array,
        reference_tokens: mx.array | None,
        reference_ids: mx.array | None,
    ) -> mx.array:
        if _base_variant(self.variant):
            return self._predict_cfg_step(
                sample=sample,
                latent_ids=latent_ids,
                prompt_embeddings=prompt_embeddings,
                text_ids=text_ids,
                timestep=timestep,
                guidance=guidance,
                reference_tokens=reference_tokens,
                reference_ids=reference_ids,
            )
        sample_tokens = sample
        sample_ids = latent_ids
        if reference_tokens is not None and reference_ids is not None:
            sample_tokens = mx.concatenate([sample_tokens, reference_tokens], axis=1)
            sample_ids = mx.concatenate([sample_ids, reference_ids], axis=1)
        prediction = self._transformer(
            x=sample_tokens,
            x_ids=sample_ids,
            timesteps=timestep,
            ctx=prompt_embeddings,
            ctx_ids=text_ids,
            guidance=guidance_vector,
        )
        if reference_tokens is not None:
            prediction = prediction[:, : int(sample.shape[1]), :]
        return prediction

    def _predict_cfg_step(
        self,
        *,
        sample: mx.array,
        latent_ids: mx.array,
        prompt_embeddings: mx.array,
        text_ids: mx.array,
        timestep: mx.array,
        guidance: float,
        reference_tokens: mx.array | None,
        reference_ids: mx.array | None,
    ) -> mx.array:
        # Official base klein rows use unconditional + prompt conditioning and
        # combine the predictions with CFG instead of guidance-distilled embeds.
        duplicated_sample = mx.concatenate([sample, sample], axis=0)
        duplicated_ids = mx.concatenate([latent_ids, latent_ids], axis=0)
        sample_tokens = duplicated_sample
        sample_ids = duplicated_ids
        if reference_tokens is not None and reference_ids is not None:
            sample_tokens = mx.concatenate(
                [
                    sample_tokens,
                    mx.concatenate([reference_tokens, reference_tokens], axis=0),
                ],
                axis=1,
            )
            sample_ids = mx.concatenate(
                [sample_ids, mx.concatenate([reference_ids, reference_ids], axis=0)],
                axis=1,
            )
        duplicated_timestep = mx.concatenate([timestep, timestep], axis=0)
        prediction = self._transformer(
            x=sample_tokens,
            x_ids=sample_ids,
            timesteps=duplicated_timestep,
            ctx=prompt_embeddings,
            ctx_ids=text_ids,
            guidance=None,
        )
        if reference_tokens is not None:
            prediction = prediction[:, : int(sample.shape[1]), :]
        prediction_uncond, prediction_cond = mx.split(prediction, 2, axis=0)
        return prediction_uncond + guidance * (prediction_cond - prediction_uncond)

    def _encode_reference_sequence(
        self, image_paths: tuple[Path, ...]
    ) -> tuple[mx.array | None, mx.array | None]:
        if not image_paths:
            return None, None
        limit_pixels = _conditioning_reference_limit(len(image_paths))
        reference_tokens: list[mx.array] = []
        reference_ids: list[mx.array] = []
        for index, image_path in enumerate(image_paths):
            with Image.open(image_path) as reference_image:
                tensor = _reference_tensor(reference_image, limit_pixels=limit_pixels)
            encoded = self._vae.encode(tensor[None, ...])[0]
            flattened, ids = prepare_latent_images(encoded[None, ...])
            time_offset = mx.full(
                (1, int(flattened.shape[1]), 1),
                10 * (index + 1),
                dtype=mx.int32,
            )
            ids = mx.concatenate([time_offset, ids[:, :, 1:]], axis=-1)
            reference_tokens.append(flattened)
            reference_ids.append(ids)
        return (
            mx.concatenate(reference_tokens, axis=1),
            mx.concatenate(reference_ids, axis=1),
        )

    def close(self) -> None:
        self._prompt_encoder.close()
        return None


def create_image_generator(
    *,
    variant: str,
    model_root: Path,
    task: str,
    quantize_bits: int | None,
    lora_paths: tuple[Path, ...],
    lora_scales: tuple[float, ...],
) -> ImageGenerator:
    if variant == "flux.2-dev":
        raise ValueError("FLUX.2-dev is not implemented in the owned MLX backend yet")
    if quantize_bits is not None:
        raise ValueError(
            "FLUX.2 quantization is not implemented in the owned MLX backend yet"
        )
    if lora_paths or lora_scales:
        raise ValueError(
            "FLUX.2 LoRA loading is not implemented in the owned MLX backend yet"
        )
    prompt_encoder = create_prompt_encoder(
        text_encoder_path=model_root / "text_encoder",
        tokenizer_path=model_root / "tokenizer",
    )
    transformer = load_local_flux2_transformer(model_root / "transformer")
    vae = load_local_autoencoder(model_root / "vae")
    scheduler = load_local_scheduler(model_root / "scheduler")
    return _RuntimeImageGenerator(
        variant=variant,
        model_root=model_root,
        task=task,
        _prompt_encoder=prompt_encoder,
        _transformer=transformer,
        _vae=vae,
        _scheduler=scheduler,
    )
