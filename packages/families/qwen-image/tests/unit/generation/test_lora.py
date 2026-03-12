from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlxr.families.qwen_image._generation_backend.lora import apply_lora_deltas
from safetensors.numpy import save_file


class QwenImageLoraTests(unittest.TestCase):
    def test_apply_lora_deltas_updates_matching_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_path = Path(tmp_dir) / "lightning.safetensors"
            save_file(
                {
                    "transformer_blocks.0.attn.to_out.0.lora_up.weight": np.array(
                        [[1.0], [2.0]],
                        dtype=np.float32,
                    ),
                    "transformer_blocks.0.attn.to_out.0.lora_down.weight": np.array(
                        [[3.0, 4.0]],
                        dtype=np.float32,
                    ),
                    "transformer_blocks.0.attn.to_out.0.alpha": np.array(
                        1,
                        dtype=np.int64,
                    ),
                },
                str(lora_path),
            )
            base = [
                (
                    "transformer_blocks.0.attn.to_out.weight",
                    mx.zeros((2, 2), dtype=mx.float32),
                )
            ]

            updated = apply_lora_deltas(
                base,
                lora_paths=(lora_path,),
                lora_scales=(0.5,),
                expected_shapes={"transformer_blocks.0.attn.to_out.weight": (2, 2)},
            )

        self.assertEqual(len(updated), 1)
        name, value = updated[0]
        self.assertEqual(name, "transformer_blocks.0.attn.to_out.weight")
        self.assertTrue(
            np.allclose(
                np.asarray(value),
                np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float32),
            )
        )

    def test_apply_lora_deltas_rejects_unknown_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_path = Path(tmp_dir) / "broken.safetensors"
            save_file(
                {
                    "transformer_blocks.0.attn.unknown.lora_up.weight": np.array(
                        [[1.0]],
                        dtype=np.float32,
                    ),
                    "transformer_blocks.0.attn.unknown.lora_down.weight": np.array(
                        [[1.0]],
                        dtype=np.float32,
                    ),
                },
                str(lora_path),
            )

            with self.assertRaisesRegex(ValueError, "target weights were not found"):
                apply_lora_deltas(
                    [],
                    lora_paths=(lora_path,),
                    lora_scales=(1.0,),
                    expected_shapes={},
                )

    def test_apply_lora_deltas_supports_wuli_a_b_naming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_path = Path(tmp_dir) / "wuli.safetensors"
            save_file(
                {
                    "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_A.weight": np.array(
                        [[3.0, 4.0]],
                        dtype=np.float32,
                    ),
                    "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_B.weight": np.array(
                        [[1.0], [2.0]],
                        dtype=np.float32,
                    ),
                },
                str(lora_path),
            )
            base = [
                (
                    "transformer_blocks.0.attn.to_out.weight",
                    mx.zeros((2, 2), dtype=mx.float32),
                )
            ]

            updated = apply_lora_deltas(
                base,
                lora_paths=(lora_path,),
                lora_scales=(0.5,),
                expected_shapes={"transformer_blocks.0.attn.to_out.weight": (2, 2)},
            )

        self.assertEqual(len(updated), 1)
        _, value = updated[0]
        self.assertTrue(
            np.allclose(
                np.asarray(value),
                np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float32),
            )
        )

    def test_apply_lora_deltas_aliases_wuli_modulation_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_path = Path(tmp_dir) / "wuli-mod.safetensors"
            save_file(
                {
                    "diffusion_model.transformer_blocks.0.img_mod.1.lora_A.weight": np.array(
                        [[2.0, 0.0]],
                        dtype=np.float32,
                    ),
                    "diffusion_model.transformer_blocks.0.img_mod.1.lora_B.weight": np.array(
                        [[1.0], [3.0]],
                        dtype=np.float32,
                    ),
                },
                str(lora_path),
            )
            base = [
                (
                    "transformer_blocks.0.img_mod.weight",
                    mx.zeros((2, 2), dtype=mx.float32),
                )
            ]

            updated = apply_lora_deltas(
                base,
                lora_paths=(lora_path,),
                lora_scales=(1.0,),
                expected_shapes={"transformer_blocks.0.img_mod.weight": (2, 2)},
            )

        self.assertEqual(len(updated), 1)
        name, value = updated[0]
        self.assertEqual(name, "transformer_blocks.0.img_mod.weight")
        self.assertTrue(
            np.allclose(
                np.asarray(value),
                np.array([[2.0, 0.0], [6.0, 0.0]], dtype=np.float32),
            )
        )


if __name__ == "__main__":
    unittest.main()
