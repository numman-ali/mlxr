from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlxr.families.qwen_image._generation_backend.loading import (
    load_local_scheduler,
    weight_files,
)


class QwenImageGenerationLoadingTests(unittest.TestCase):
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

            indexed_files = weight_files(indexed)
            legacy_files = weight_files(legacy)

        self.assertEqual(
            [path.name for path in indexed_files],
            [
                "diffusion_pytorch_model-00001-of-00002.safetensors",
                "diffusion_pytorch_model-00002-of-00002.safetensors",
            ],
        )
        self.assertEqual([path.name for path in legacy_files], ["model.safetensors"])

    def test_load_local_scheduler_reads_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "scheduler_config.json").write_text(
                json.dumps(
                    {
                        "base_image_seq_len": 256,
                        "base_shift": 0.5,
                        "invert_sigmas": False,
                        "max_image_seq_len": 8192,
                        "max_shift": 0.9,
                        "num_train_timesteps": 1000,
                        "shift_terminal": 0.02,
                        "time_shift_type": "exponential",
                        "use_dynamic_shifting": True,
                    }
                ),
                encoding="utf-8",
            )

            scheduler = load_local_scheduler(root)

        self.assertEqual(scheduler.config.base_image_seq_len, 256)
        self.assertEqual(scheduler.config.shift_terminal, 0.02)

    def test_load_local_scheduler_supports_lightning_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "scheduler_config.json").write_text(
                json.dumps(
                    {
                        "base_image_seq_len": 256,
                        "base_shift": 0.5,
                        "invert_sigmas": False,
                        "max_image_seq_len": 8192,
                        "max_shift": 0.9,
                        "num_train_timesteps": 1000,
                        "shift_terminal": 0.02,
                        "time_shift_type": "exponential",
                        "use_dynamic_shifting": True,
                    }
                ),
                encoding="utf-8",
            )

            scheduler = load_local_scheduler(root, scheduler_preset="lightning")

        self.assertAlmostEqual(scheduler.config.base_shift, 1.0986122886681098)
        self.assertAlmostEqual(scheduler.config.max_shift, 1.0986122886681098)
        self.assertIsNone(scheduler.config.shift_terminal)

    def test_load_local_scheduler_supports_turbo_wuli_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "scheduler_config.json").write_text(
                json.dumps(
                    {
                        "base_image_seq_len": 256,
                        "base_shift": 0.5,
                        "invert_sigmas": False,
                        "max_image_seq_len": 8192,
                        "max_shift": 0.9,
                        "num_train_timesteps": 1000,
                        "shift_terminal": 0.02,
                        "time_shift_type": "exponential",
                        "use_dynamic_shifting": True,
                    }
                ),
                encoding="utf-8",
            )

            scheduler = load_local_scheduler(root, scheduler_preset="turbo_wuli")

        self.assertAlmostEqual(scheduler.config.base_shift, 0.9162907318741551)
        self.assertAlmostEqual(scheduler.config.max_shift, 0.9162907318741551)
        self.assertIsNone(scheduler.config.shift_terminal)


if __name__ == "__main__":
    unittest.main()
