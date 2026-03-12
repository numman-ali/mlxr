from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlxr.families.flux2._generation_backend.runtime import create_image_generator
from mlxr.families.flux2._generation_backend.transformer import Flux2KVCache
from mlxr.families.flux2.prompt_encoding import PromptEncodingResult
from PIL import Image


class _FakePromptEncoder:
    def __init__(self) -> None:
        self.closed = False
        self.prompts: list[str] = []

    def encode(self, prompt: str, *, max_length: int = 512) -> PromptEncodingResult:
        del max_length
        self.prompts.append(prompt)
        return PromptEncodingResult(
            prompt_embeddings=mx.ones((1, 3, 12), dtype=mx.float32),
            prompt_text=prompt,
            token_count=3,
            sequence_length=3,
            hidden_size=12,
            attention_mask=mx.ones((1, 3), dtype=mx.int32),
        )

    def encode_many(
        self,
        prompts: tuple[str, ...],
        *,
        max_length: int = 512,
    ) -> tuple[PromptEncodingResult, ...]:
        return tuple(self.encode(prompt, max_length=max_length) for prompt in prompts)

    def close(self) -> None:
        self.closed = True
        return None


class _FakeScheduler:
    def timesteps(
        self, *, num_inference_steps: int, image_sequence_length: int
    ) -> tuple[float, ...]:
        del image_sequence_length
        return tuple(float(value) for value in range(num_inference_steps, -1, -1))

    def step(
        self,
        *,
        sample: mx.array,
        model_output: mx.array,
        timestep: float,
        next_timestep: float,
    ) -> mx.array:
        del model_output, timestep, next_timestep
        return sample


class _FakeTransformer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def _record_call(
        self,
        *,
        mode: str,
        x: mx.array,
        x_ids: mx.array,
        timesteps: mx.array,
        ctx: mx.array,
        ctx_ids: mx.array,
        guidance: mx.array | None,
        reference_sequence: int = 0,
        cached_ref_tokens: int = 0,
    ) -> None:
        self.calls.append(
            {
                "mode": mode,
                "batch": int(x.shape[0]),
                "sequence": int(x.shape[1]),
                "ctx_batch": int(ctx.shape[0]),
                "guidance_is_none": guidance is None,
                "timestep_batch": int(timesteps.shape[0]),
                "ctx_ids_batch": int(ctx_ids.shape[0]),
                "x_ids_batch": int(x_ids.shape[0]),
                "reference_sequence": reference_sequence,
                "cached_ref_tokens": cached_ref_tokens,
            }
        )

    def __call__(
        self,
        *,
        x: mx.array,
        x_ids: mx.array,
        timesteps: mx.array,
        ctx: mx.array,
        ctx_ids: mx.array,
        guidance: mx.array | None,
    ) -> mx.array:
        self._record_call(
            mode="standard",
            x=x,
            x_ids=x_ids,
            timesteps=timesteps,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=guidance,
        )
        return mx.zeros_like(x)

    def forward_kv_extract(
        self,
        *,
        x: mx.array,
        x_ids: mx.array,
        x_ref: mx.array,
        x_ref_ids: mx.array,
        timesteps: mx.array,
        ctx: mx.array,
        ctx_ids: mx.array,
        guidance: mx.array | None,
        ref_fixed_timestep: float = 0.0,
    ) -> tuple[mx.array, Flux2KVCache]:
        del x_ref_ids, ref_fixed_timestep
        self._record_call(
            mode="extract",
            x=x,
            x_ids=x_ids,
            timesteps=timesteps,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=guidance,
            reference_sequence=int(x_ref.shape[1]),
        )
        return mx.zeros_like(x), Flux2KVCache.create(
            num_double_layers=1,
            num_single_layers=1,
            num_ref_tokens=int(x_ref.shape[1]),
        )

    def forward_kv_cached(
        self,
        *,
        x: mx.array,
        x_ids: mx.array,
        timesteps: mx.array,
        ctx: mx.array,
        ctx_ids: mx.array,
        guidance: mx.array | None,
        kv_cache: Flux2KVCache,
    ) -> mx.array:
        self._record_call(
            mode="cached",
            x=x,
            x_ids=x_ids,
            timesteps=timesteps,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=guidance,
            cached_ref_tokens=kv_cache.num_ref_tokens,
        )
        return mx.zeros_like(x)


class _FakeVAE:
    def encode(self, x: mx.array) -> mx.array:
        batch_size = int(x.shape[0])
        height = int(x.shape[1]) // 16
        width = int(x.shape[2]) // 16
        return mx.zeros((batch_size, height, width, 128), dtype=mx.float32)

    def decode(self, z: mx.array) -> mx.array:
        batch_size = int(z.shape[0])
        height = int(z.shape[1]) * 16
        width = int(z.shape[2]) * 16
        return mx.zeros((batch_size, height, width, 3), dtype=mx.float32)


class Flux2RuntimeTests(unittest.TestCase):
    def test_create_image_generator_rejects_quantize_and_lora_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "quantization is not implemented"):
            create_image_generator(
                variant="flux.2-klein-4b",
                model_root=Path("/tmp/model"),
                task="image.generate",
                quantize_bits=8,
                lora_paths=(),
                lora_scales=(),
            )
        with self.assertRaisesRegex(ValueError, "LoRA loading is not implemented"):
            create_image_generator(
                variant="flux.2-klein-4b",
                model_root=Path("/tmp/model"),
                task="image.generate",
                quantize_bits=None,
                lora_paths=(Path("/tmp/test.safetensors"),),
                lora_scales=(1.0,),
            )

    def test_runtime_image_generator_runs_generate_and_multiref_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ref_a = root / "ref_a.png"
            ref_b = root / "ref_b.png"
            Image.new("RGB", (64, 64), color=(32, 64, 128)).save(ref_a)
            Image.new("RGB", (64, 64), color=(200, 32, 32)).save(ref_b)

            fake_encoder = _FakePromptEncoder()
            fake_transformer = _FakeTransformer()
            with (
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.create_prompt_encoder",
                    return_value=fake_encoder,
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_flux2_transformer",
                    return_value=fake_transformer,
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_autoencoder",
                    return_value=_FakeVAE(),
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_scheduler",
                    return_value=_FakeScheduler(),
                ),
            ):
                generator = create_image_generator(
                    variant="flux.2-klein-4b",
                    model_root=root,
                    task="image.generate",
                    quantize_bits=None,
                    lora_paths=(),
                    lora_scales=(),
                )
                generated = generator.generate(
                    prompt="tiny flux scene",
                    task="image.generate",
                    width=64,
                    height=64,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    negative_prompt=None,
                    seed=7,
                    image_paths=(),
                )
                edited = generator.generate(
                    prompt="blend the two tiny references",
                    task="image.edit",
                    width=64,
                    height=64,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    negative_prompt=None,
                    seed=9,
                    image_paths=(ref_a, ref_b),
                )
                generator.close()

        self.assertEqual(generated.pixels.shape, (64, 64, 3))
        self.assertEqual(generated.metadata["conditioning_reference_count"], 0)
        self.assertFalse(bool(generated.metadata["kv_cache_used"]))
        self.assertEqual(edited.pixels.shape, (64, 64, 3))
        self.assertEqual(edited.metadata["conditioning_reference_count"], 2)
        self.assertGreater(int(edited.metadata["conditioning_sequence_length"]), 0)
        self.assertIsInstance(generated.prompt_signature, str)
        self.assertEqual(generated.pixels.dtype, np.uint8)
        self.assertTrue(fake_encoder.closed)
        self.assertEqual(fake_transformer.calls[0]["ctx_batch"], 1)
        self.assertFalse(bool(fake_transformer.calls[0]["guidance_is_none"]))

    def test_runtime_generator_rejects_invalid_runtime_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            with (
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.create_prompt_encoder",
                    return_value=_FakePromptEncoder(),
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_flux2_transformer",
                    return_value=_FakeTransformer(),
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_autoencoder",
                    return_value=_FakeVAE(),
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_scheduler",
                    return_value=_FakeScheduler(),
                ),
            ):
                generator = create_image_generator(
                    variant="flux.2-klein-4b",
                    model_root=root,
                    task="image.generate",
                    quantize_bits=None,
                    lora_paths=(),
                    lora_scales=(),
                )
                with self.assertRaisesRegex(
                    ValueError, "negative_prompt is not implemented"
                ):
                    generator.generate(
                        prompt="x",
                        task="image.generate",
                        width=64,
                        height=64,
                        num_inference_steps=4,
                        guidance_scale=1.0,
                        negative_prompt="bad",
                        seed=1,
                        image_paths=(),
                    )
                with self.assertRaisesRegex(ValueError, "Unsupported FLUX.2 task"):
                    generator.generate(
                        prompt="x",
                        task="video.generate",
                        width=64,
                        height=64,
                        num_inference_steps=4,
                        guidance_scale=1.0,
                        negative_prompt=None,
                        seed=1,
                        image_paths=(),
                    )
                with self.assertRaisesRegex(
                    ValueError, "requires at least one reference"
                ):
                    generator.generate(
                        prompt="x",
                        task="image.edit",
                        width=64,
                        height=64,
                        num_inference_steps=4,
                        guidance_scale=1.0,
                        negative_prompt=None,
                        seed=1,
                        image_paths=(),
                    )
                with self.assertRaisesRegex(
                    ValueError, "requires num_inference_steps=4"
                ):
                    generator.generate(
                        prompt="x",
                        task="image.generate",
                        width=64,
                        height=64,
                        num_inference_steps=5,
                        guidance_scale=1.0,
                        negative_prompt=None,
                        seed=1,
                        image_paths=(),
                    )
                with self.assertRaisesRegex(ValueError, "requires guidance_scale=1.0"):
                    generator.generate(
                        prompt="x",
                        task="image.generate",
                        width=64,
                        height=64,
                        num_inference_steps=4,
                        guidance_scale=1.5,
                        negative_prompt=None,
                        seed=1,
                        image_paths=(),
                    )
                self.assertEqual(generator._encode_reference_sequence(()), (None, None))
                generator.close()

    def test_base_variant_uses_cfg_conditioning_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fake_encoder = _FakePromptEncoder()
            fake_transformer = _FakeTransformer()
            with (
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.create_prompt_encoder",
                    return_value=fake_encoder,
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_flux2_transformer",
                    return_value=fake_transformer,
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_autoencoder",
                    return_value=_FakeVAE(),
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_scheduler",
                    return_value=_FakeScheduler(),
                ),
            ):
                generator = create_image_generator(
                    variant="flux.2-klein-base-4b",
                    model_root=root,
                    task="image.generate",
                    quantize_bits=None,
                    lora_paths=(),
                    lora_scales=(),
                )
                generated = generator.generate(
                    prompt="base flux scene",
                    task="image.generate",
                    width=64,
                    height=64,
                    num_inference_steps=None,
                    guidance_scale=None,
                    negative_prompt=None,
                    seed=3,
                    image_paths=(),
                )
                generator.close()

        self.assertEqual(generated.metadata["num_inference_steps"], 50)
        self.assertEqual(generated.metadata["guidance_scale"], 4.0)
        self.assertEqual(
            fake_encoder.prompts[:3], ["base flux scene", "", "base flux scene"]
        )
        self.assertTrue(bool(fake_transformer.calls))
        self.assertEqual(fake_transformer.calls[0]["batch"], 2)
        self.assertEqual(fake_transformer.calls[0]["ctx_batch"], 2)
        self.assertTrue(bool(fake_transformer.calls[0]["guidance_is_none"]))

    def test_dev_variant_fails_closed_until_owned_prompt_stack_exists(self) -> None:
        with self.assertRaisesRegex(ValueError, "FLUX.2-dev is not implemented"):
            create_image_generator(
                variant="flux.2-dev",
                model_root=Path("/tmp/model"),
                task="image.generate",
                quantize_bits=None,
                lora_paths=(),
                lora_scales=(),
            )

    def test_kv_variant_uses_extract_then_cached_steps_for_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ref_a = root / "ref_a.png"
            ref_b = root / "ref_b.png"
            Image.new("RGB", (64, 64), color=(12, 22, 32)).save(ref_a)
            Image.new("RGB", (64, 64), color=(132, 42, 52)).save(ref_b)

            fake_transformer = _FakeTransformer()
            with (
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.create_prompt_encoder",
                    return_value=_FakePromptEncoder(),
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_flux2_transformer",
                    return_value=fake_transformer,
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_autoencoder",
                    return_value=_FakeVAE(),
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_scheduler",
                    return_value=_FakeScheduler(),
                ),
            ):
                generator = create_image_generator(
                    variant="flux.2-klein-9b-kv",
                    model_root=root,
                    task="image.edit",
                    quantize_bits=None,
                    lora_paths=(),
                    lora_scales=(),
                )
                generated = generator.generate(
                    prompt="blend the two reference images into a cinematic portrait",
                    task="image.edit",
                    width=64,
                    height=64,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    negative_prompt=None,
                    seed=5,
                    image_paths=(ref_a, ref_b),
                )
                generator.close()

        self.assertTrue(bool(generated.metadata["kv_cache_used"]))
        self.assertEqual(generated.metadata["kv_cache_reference_count"], 2)
        self.assertGreater(int(generated.metadata["kv_cache_reference_token_count"]), 0)
        self.assertEqual(generated.metadata["kv_cache_reuse_steps"], 3)
        self.assertEqual(
            [call["mode"] for call in fake_transformer.calls],
            ["extract", "cached", "cached", "cached"],
        )
        self.assertGreater(int(fake_transformer.calls[0]["reference_sequence"]), 0)
        self.assertEqual(fake_transformer.calls[1]["cached_ref_tokens"], 32)

    def test_kv_variant_falls_back_to_standard_path_without_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fake_transformer = _FakeTransformer()
            with (
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.create_prompt_encoder",
                    return_value=_FakePromptEncoder(),
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_flux2_transformer",
                    return_value=fake_transformer,
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_autoencoder",
                    return_value=_FakeVAE(),
                ),
                patch(
                    "mlxr.families.flux2._generation_backend.runtime.load_local_scheduler",
                    return_value=_FakeScheduler(),
                ),
            ):
                generator = create_image_generator(
                    variant="flux.2-klein-9b-kv",
                    model_root=root,
                    task="image.generate",
                    quantize_bits=None,
                    lora_paths=(),
                    lora_scales=(),
                )
                generated = generator.generate(
                    prompt="a cat holding a sign that says hello world",
                    task="image.generate",
                    width=64,
                    height=64,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    negative_prompt=None,
                    seed=11,
                    image_paths=(),
                )
                generator.close()

        self.assertFalse(bool(generated.metadata["kv_cache_used"]))
        self.assertEqual(
            [call["mode"] for call in fake_transformer.calls],
            ["standard", "standard", "standard", "standard"],
        )


if __name__ == "__main__":
    unittest.main()
