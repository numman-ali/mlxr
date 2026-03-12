from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlxr.families.qwen_image._generation_backend.config import (
    AutoencoderConfig,
    QwenImageTransformerConfig,
    SchedulerConfig,
)


class QwenImageGenerationConfigTests(unittest.TestCase):
    def test_config_loaders_parse_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            transformer_path = root / "transformer.json"
            transformer_path.write_text(
                json.dumps(
                    {
                        "attention_head_dim": 128,
                        "axes_dims_rope": [16, 56, 56],
                        "guidance_embeds": False,
                        "in_channels": 64,
                        "joint_attention_dim": 3584,
                        "num_attention_heads": 24,
                        "num_layers": 60,
                        "out_channels": 16,
                        "patch_size": 2,
                        "pooled_projection_dim": 768,
                    }
                ),
                encoding="utf-8",
            )
            vae_path = root / "vae.json"
            vae_path.write_text(
                json.dumps(
                    {
                        "attn_scales": [],
                        "base_dim": 96,
                        "dim_mult": [1, 2, 4, 4],
                        "dropout": 0.0,
                        "input_channels": 3,
                        "latents_mean": [0.1, 0.2],
                        "latents_std": [1.0, 2.0],
                        "num_res_blocks": 2,
                        "temperal_downsample": [False, True, True],
                        "z_dim": 16,
                    }
                ),
                encoding="utf-8",
            )
            scheduler_path = root / "scheduler.json"
            scheduler_path.write_text(
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

            transformer = QwenImageTransformerConfig.from_path(transformer_path)
            vae = AutoencoderConfig.from_path(vae_path)
            scheduler = SchedulerConfig.from_path(scheduler_path)

        self.assertEqual(transformer.hidden_size, 3072)
        self.assertEqual(transformer.axes_dims_rope, (16, 56, 56))
        self.assertEqual(transformer.latent_channels, 16)
        self.assertEqual(vae.scale_factor, 8)
        self.assertEqual(vae.pixel_multiple, 16)
        self.assertEqual(vae.temporal_upsample, (True, True, False))
        self.assertTrue(scheduler.use_dynamic_shifting)


if __name__ == "__main__":
    unittest.main()
