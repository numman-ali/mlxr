from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np
from mlxr.families.z_image._generation_backend import (
    AutoencoderConfig,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SchedulerConfig,
    ZImageTransformer2DModel,
    ZImageTransformerConfig,
    _RuntimeImageGenerator,
    _weight_files,
    encode_jpg_image,
)
from mlxr.families.z_image.generation import GeneratedImage
from mlxr.families.z_image.prompt_encoding import PromptEncodingResult


class ZImageGenerationBackendTests(unittest.TestCase):
    def test_transformer_forward_returns_original_latent_shape(self) -> None:
        model = ZImageTransformer2DModel(_tiny_transformer_config())
        latents = mx.random.normal((2, 1, 8, 8), dtype=mx.float32)
        prompt_embeddings = mx.random.normal((5, 8), dtype=mx.float32)

        generated, metadata = model(
            (latents,),
            mx.array([0.5], dtype=mx.float32),
            (prompt_embeddings,),
        )

        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].shape, latents.shape)
        self.assertEqual(metadata, {})

    def test_patchify_and_unpatchify_round_trip_tiny_latents(self) -> None:
        model = ZImageTransformer2DModel(_tiny_transformer_config())
        image = mx.arange(2 * 1 * 4 * 4, dtype=mx.float32).reshape(2, 1, 4, 4)
        cap = mx.ones((3, 8), dtype=mx.float32)

        image_tokens, _, sizes, _, _ = model.patchify_and_embed((image,), (cap,))
        restored = model.unpatchify((image_tokens[0],), sizes)[0]

        np.testing.assert_allclose(np.asarray(restored), np.asarray(image))

    def test_scheduler_step_keeps_tensor_shape(self) -> None:
        scheduler = FlowMatchEulerDiscreteScheduler(
            SchedulerConfig(num_train_timesteps=16, shift=3.0)
        )
        scheduler.set_timesteps(4, mu=0.75)
        sample = mx.zeros((1, 2, 4, 4), dtype=mx.float32)
        model_output = mx.ones_like(sample)

        updated = scheduler.step(
            model_output,
            float(np.asarray(scheduler.timesteps)[0]),
            sample,
        )

        self.assertEqual(updated.shape, sample.shape)
        self.assertFalse(np.allclose(np.asarray(updated), np.asarray(sample)))

    def test_autoencoder_decode_restores_spatial_resolution(self) -> None:
        autoencoder = AutoencoderKL(_tiny_autoencoder_config())
        latent = mx.zeros((1, 2, 4, 4), dtype=mx.float32)

        decoded = autoencoder.decode(latent)

        self.assertEqual(decoded.shape, (1, 3, 8, 8))

    def test_runtime_image_generator_emits_uint8_pixels(self) -> None:
        generator = _RuntimeImageGenerator(
            transformer_path=Path("/tmp/transformer"),
            vae_path=Path("/tmp/vae"),
            scheduler_path=Path("/tmp/scheduler"),
            _transformer=ZImageTransformer2DModel(_tiny_transformer_config()),
            _vae=AutoencoderKL(_tiny_autoencoder_config()),
            _scheduler=FlowMatchEulerDiscreteScheduler(
                SchedulerConfig(num_train_timesteps=16, shift=3.0)
            ),
        )
        prompt_context = PromptEncodingResult(
            prompt_embeddings=(np.ones((4, 8), dtype=np.float32),),
            prompt_text="tiny z image",
            token_count=4,
            sequence_length=4,
            hidden_size=8,
        )

        image = generator.generate(
            prompt_context=prompt_context,
            width=16,
            height=16,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=3,
        )

        self.assertEqual(image.pixels.shape, (16, 16, 3))
        self.assertEqual(image.pixels.dtype, np.uint8)
        self.assertEqual(image.seed, 3)
        self.assertEqual(image.backend, "native_mlx_z_image")
        self.assertEqual(image.metadata["cfg_normalization"], 0.0)
        self.assertEqual(image.metadata["cfg_truncation"], 1.0)

    def test_runtime_image_generator_requires_negative_prompt_for_cfg(self) -> None:
        generator = _RuntimeImageGenerator(
            transformer_path=Path("/tmp/transformer"),
            vae_path=Path("/tmp/vae"),
            scheduler_path=Path("/tmp/scheduler"),
            _transformer=ZImageTransformer2DModel(_tiny_transformer_config()),
            _vae=AutoencoderKL(_tiny_autoencoder_config()),
            _scheduler=FlowMatchEulerDiscreteScheduler(
                SchedulerConfig(num_train_timesteps=16, shift=3.0)
            ),
        )
        prompt_context = PromptEncodingResult(
            prompt_embeddings=(np.ones((4, 8), dtype=np.float32),),
            prompt_text="tiny z image",
            token_count=4,
            sequence_length=4,
            hidden_size=8,
        )

        with self.assertRaisesRegex(ValueError, "negative prompt embeddings"):
            generator.generate(
                prompt_context=prompt_context,
                width=16,
                height=16,
                num_inference_steps=2,
                guidance_scale=2.0,
                seed=3,
            )

    def test_runtime_image_generator_supports_cfg_normalization(self) -> None:
        generator = _RuntimeImageGenerator(
            transformer_path=Path("/tmp/transformer"),
            vae_path=Path("/tmp/vae"),
            scheduler_path=Path("/tmp/scheduler"),
            _transformer=ZImageTransformer2DModel(_tiny_transformer_config()),
            _vae=AutoencoderKL(_tiny_autoencoder_config()),
            _scheduler=FlowMatchEulerDiscreteScheduler(
                SchedulerConfig(num_train_timesteps=16, shift=3.0)
            ),
        )
        prompt_context = PromptEncodingResult(
            prompt_embeddings=(np.ones((4, 8), dtype=np.float32),),
            prompt_text="tiny z image",
            token_count=4,
            sequence_length=4,
            hidden_size=8,
            negative_prompt_embeddings=(np.zeros((4, 8), dtype=np.float32),),
        )

        image = generator.generate(
            prompt_context=prompt_context,
            width=16,
            height=16,
            num_inference_steps=2,
            guidance_scale=2.0,
            cfg_normalization=1.0,
            cfg_truncation=0.5,
            seed=3,
        )

        self.assertEqual(image.metadata["cfg_normalization"], 1.0)
        self.assertEqual(image.metadata["cfg_truncation"], 0.5)

    def test_runtime_image_generator_emits_trace_metadata_when_enabled(self) -> None:
        generator = _RuntimeImageGenerator(
            transformer_path=Path("/tmp/transformer"),
            vae_path=Path("/tmp/vae"),
            scheduler_path=Path("/tmp/scheduler"),
            _transformer=ZImageTransformer2DModel(_tiny_transformer_config()),
            _vae=AutoencoderKL(_tiny_autoencoder_config()),
            _scheduler=FlowMatchEulerDiscreteScheduler(
                SchedulerConfig(num_train_timesteps=16, shift=3.0)
            ),
        )
        prompt_context = PromptEncodingResult(
            prompt_embeddings=(np.ones((4, 8), dtype=np.float32),),
            prompt_text="tiny z image",
            token_count=4,
            sequence_length=4,
            hidden_size=8,
        )

        with mock.patch.dict(os.environ, {"MLXR_Z_IMAGE_DEBUG_TRACE": "1"}):
            image = generator.generate(
                prompt_context=prompt_context,
                width=16,
                height=16,
                num_inference_steps=2,
                guidance_scale=0.0,
                seed=3,
            )

        trace = image.metadata["trace"]
        self.assertIsInstance(trace, dict)
        self.assertGreater(trace["event_count"], 0)
        self.assertIn("zimage.prepare_latents", trace["summary"])

    def test_weight_files_support_index_and_legacy_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            indexed = root / "indexed"
            indexed.mkdir()
            (
                indexed / "diffusion_pytorch_model-00001-of-00002.safetensors"
            ).write_bytes(b"one")
            (
                indexed / "diffusion_pytorch_model-00002-of-00002.safetensors"
            ).write_bytes(b"two")
            (indexed / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "diffusion_pytorch_model-00002-of-00002.safetensors",
                            "b": "diffusion_pytorch_model-00001-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            legacy = root / "legacy"
            legacy.mkdir()
            (legacy / "model.safetensors").write_bytes(b"legacy")

            indexed_files = _weight_files(indexed)
            legacy_files = _weight_files(legacy)

        self.assertEqual(
            [path.name for path in indexed_files],
            [
                "diffusion_pytorch_model-00001-of-00002.safetensors",
                "diffusion_pytorch_model-00002-of-00002.safetensors",
            ],
        )
        self.assertEqual([path.name for path in legacy_files], ["model.safetensors"])

    def test_jpg_encoder_writes_file(self) -> None:
        image = GeneratedImage(
            pixels=np.full((8, 8, 3), 64, dtype=np.uint8),
            seed=1,
            backend="test",
            prompt_signature="signature",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "image.jpg"
            encode_jpg_image(image, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


def _tiny_transformer_config() -> ZImageTransformerConfig:
    return ZImageTransformerConfig(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=2,
        dim=16,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=8,
        axes_dims=(2, 2, 4),
        axes_lens=(64, 16, 16),
    )


def _tiny_autoencoder_config() -> AutoencoderConfig:
    return AutoencoderConfig(
        block_out_channels=(32, 64),
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=8,
        scaling_factor=0.5,
        shift_factor=0.0,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )
