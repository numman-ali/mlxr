from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlxr.families.ltx._generation_backend.lora import apply_lora_deltas
from safetensors.numpy import save_file


class LTXLoraTests(unittest.TestCase):
    def test_apply_lora_deltas_updates_matching_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_path = Path(tmp_dir) / "distilled-lora.safetensors"
            save_file(
                {
                    "diffusion_model.patchify_proj.lora_A.weight": np.array(
                        [[1.0, 2.0, 3.0]], dtype=np.float32
                    ),
                    "diffusion_model.patchify_proj.lora_B.weight": np.array(
                        [[4.0], [5.0]], dtype=np.float32
                    ),
                },
                str(lora_path),
            )

            updated = apply_lora_deltas(
                {
                    "model.diffusion_model.patchify_proj.weight": mx.zeros(
                        (2, 3), dtype=mx.float32
                    )
                },
                lora_paths=(lora_path,),
                lora_scales=(0.5,),
            )

            self.assertIn("model.diffusion_model.patchify_proj.weight", updated)
            np.testing.assert_allclose(
                np.asarray(updated["model.diffusion_model.patchify_proj.weight"]),
                np.array([[2.0, 4.0, 6.0], [2.5, 5.0, 7.5]], dtype=np.float32),
            )

    def test_apply_lora_deltas_rejects_unknown_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_path = Path(tmp_dir) / "broken.safetensors"
            save_file(
                {
                    "diffusion_model.unknown_proj.lora_A.weight": np.array(
                        [[1.0, 2.0]], dtype=np.float32
                    ),
                    "diffusion_model.unknown_proj.lora_B.weight": np.array(
                        [[3.0]], dtype=np.float32
                    ),
                },
                str(lora_path),
            )

            with self.assertRaisesRegex(
                ValueError,
                "target weights were not found",
            ):
                apply_lora_deltas(
                    {
                        "model.diffusion_model.patchify_proj.weight": mx.zeros(
                            (1, 2), dtype=mx.float32
                        )
                    },
                    lora_paths=(lora_path,),
                    lora_scales=(1.0,),
                )


if __name__ == "__main__":
    unittest.main()
