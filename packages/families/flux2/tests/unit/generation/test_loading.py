from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
from mlxr.families.flux2._generation_backend.autoencoder import AutoencoderKLFlux2
from mlxr.families.flux2._generation_backend.config import AutoencoderConfig
from mlxr.families.flux2._generation_backend.loading import (
    _align_autoencoder_weight_layouts,
    _load_component_weights,
    _weight_files,
    load_local_autoencoder,
    load_local_flux2_transformer,
    load_local_scheduler,
)

from ._fixtures import (
    tiny_autoencoder_config,
    write_tiny_bundle,
)


class Flux2LoadingTests(unittest.TestCase):
    def test_loaders_restore_tiny_transformer_autoencoder_and_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_tiny_bundle(root)

            transformer = load_local_flux2_transformer(root / "transformer")
            autoencoder = load_local_autoencoder(root / "vae")
            scheduler = load_local_scheduler(root / "scheduler")

        x = mx.random.normal((1, 4, 8), dtype=mx.float32)
        x_ids = mx.zeros((1, 4, 4), dtype=mx.int32)
        ctx = mx.random.normal((1, 3, 12), dtype=mx.float32)
        ctx_ids = mx.zeros((1, 3, 4), dtype=mx.int32)
        transformer_out = transformer(
            x=x,
            x_ids=x_ids,
            timesteps=mx.array([0.5], dtype=mx.float32),
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=mx.array([1.0], dtype=mx.float32),
        )
        latent = autoencoder.encode(mx.random.normal((1, 32, 32, 3), dtype=mx.float32))
        decoded = autoencoder.decode(latent)
        schedule = scheduler.timesteps(num_inference_steps=2, image_sequence_length=16)

        self.assertEqual(transformer_out.shape, (1, 4, 8))
        self.assertEqual(decoded.shape, (1, 32, 32, 3))
        self.assertEqual(len(schedule), 3)

    def test_weight_files_supports_single_file_and_index_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            single = root / "single"
            single.mkdir()
            (single / "diffusion_pytorch_model.safetensors").write_bytes(b"single")

            indexed = root / "indexed"
            indexed.mkdir()
            (indexed / "model-00001-of-00002.safetensors").write_bytes(b"one")
            (indexed / "model-00002-of-00002.safetensors").write_bytes(b"two")
            (indexed / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model-00002-of-00002.safetensors",
                            "b": "model-00001-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )

            single_files = _weight_files(single)
            indexed_files = _weight_files(indexed)

        self.assertEqual(
            [path.name for path in single_files],
            ["diffusion_pytorch_model.safetensors"],
        )
        self.assertEqual(
            [path.name for path in indexed_files],
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        )

    def test_weight_files_supports_text_weights_and_diffusion_index_layouts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            text = root / "text"
            text.mkdir()
            (text / "model.safetensors").write_bytes(b"text")

            diffusion = root / "diffusion"
            diffusion.mkdir()
            (diffusion / "part-b.safetensors").write_bytes(b"b")
            (diffusion / "part-a.safetensors").write_bytes(b"a")
            (diffusion / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "b": "part-b.safetensors",
                            "a": "part-a.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )

            text_files = _weight_files(text)
            diffusion_files = _weight_files(diffusion)

        self.assertEqual([path.name for path in text_files], ["model.safetensors"])
        self.assertEqual(
            [path.name for path in diffusion_files],
            ["part-a.safetensors", "part-b.safetensors"],
        )

    def test_weight_files_rejects_missing_component_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "No FLUX.2 weights found"):
                _weight_files(Path(tmp_dir))

    def test_load_component_weights_applies_alias_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            weight_file = root / "diffusion_pytorch_model.safetensors"
            weight_file.write_bytes(b"payload")

            with patch(
                "mlxr.families.flux2._generation_backend.loading.mx.load",
                return_value={
                    "block.to_out.0.weight": mx.ones((1,), dtype=mx.float32),
                },
            ):
                weights = _load_component_weights(
                    root,
                    alias_map={".to_out.0.": ".to_out."},
                )

        self.assertEqual(weights[0][0], "block.to_out.weight")

    def test_load_component_weights_rejects_non_mapping_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "diffusion_pytorch_model.safetensors").write_bytes(b"payload")

            with patch(
                "mlxr.families.flux2._generation_backend.loading.mx.load",
                return_value=mx.ones((1,), dtype=mx.float32),
            ):
                with self.assertRaisesRegex(RuntimeError, "Expected tensor mapping"):
                    _load_component_weights(root)

    def test_load_local_autoencoder_applies_diffusers_attention_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_tiny_bundle(root)

            with (
                patch(
                    "mlxr.families.flux2._generation_backend.loading._load_component_weights",
                    return_value=[],
                ) as load_weights,
                patch.object(
                    AutoencoderKLFlux2, "load_weights", return_value=None
                ) as model_load_weights,
                patch("mlxr.families.flux2._generation_backend.loading.mx.eval"),
            ):
                load_local_autoencoder(root / "vae")

        self.assertEqual(
            load_weights.call_args.kwargs["alias_map"],
            {".to_out.0.": ".to_out."},
        )
        model_load_weights.assert_called_once()

    def test_align_autoencoder_weight_layouts_transposes_only_when_needed(self) -> None:
        conv = mx.arange(2 * 3 * 4 * 5, dtype=mx.float32).reshape(2, 3, 4, 5)
        already_aligned = mx.arange(2 * 4 * 5 * 3, dtype=mx.float32).reshape(2, 4, 5, 3)
        bias = mx.arange(2, dtype=mx.float32)

        weights = _align_autoencoder_weight_layouts(
            [
                ("conv.weight", conv),
                ("aligned.weight", already_aligned),
                ("conv.bias", bias),
            ],
            expected_shapes={
                "conv.weight": (2, 4, 5, 3),
                "aligned.weight": (2, 4, 5, 3),
                "conv.bias": (2,),
            },
        )

        self.assertEqual(weights[0][1].shape, (2, 4, 5, 3))
        self.assertEqual(weights[1][1].shape, (2, 4, 5, 3))
        self.assertEqual(weights[2][1].shape, (2,))

    def test_autoencoder_config_rejects_invalid_patch_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            bad_config = tiny_autoencoder_config()
            config_path.write_text(
                json.dumps(
                    {
                        **asdict(bad_config),
                        "patch_size": [2, 2, 2],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "patch_size must have length 2"):
                AutoencoderConfig.from_path(config_path)


if __name__ == "__main__":
    unittest.main()
