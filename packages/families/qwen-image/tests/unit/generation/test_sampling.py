from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlxr.families.qwen_image._generation_backend.config import (
    AutoencoderConfig,
    SchedulerConfig,
)
from mlxr.families.qwen_image._generation_backend.sampling import (
    calculate_shift,
    denormalize_latents,
    normalize_latents,
    pack_latents,
    shifted_sigmas,
    unpack_latents,
)


class QwenImageGenerationSamplingTests(unittest.TestCase):
    def test_pack_and_unpack_latents_round_trip(self) -> None:
        latents = mx.arange(1 * 1 * 2 * 4 * 4, dtype=mx.float32).reshape(1, 1, 2, 4, 4)

        packed = pack_latents(latents)
        unpacked = unpack_latents(packed, height=32, width=32, vae_scale_factor=8)

        self.assertEqual(tuple(packed.shape), (1, 4, 8))
        np.testing.assert_array_equal(
            np.asarray(unpacked),
            np.asarray(latents).transpose(0, 2, 1, 3, 4),
        )

    def test_normalize_and_denormalize_latents_round_trip(self) -> None:
        config = AutoencoderConfig(
            attn_scales=(),
            base_dim=96,
            dim_mult=(1, 2, 4, 4),
            dropout=0.0,
            input_channels=3,
            latents_mean=(0.1, 0.2),
            latents_std=(2.0, 4.0),
            num_res_blocks=2,
            temperal_downsample=(False, True, True),
            z_dim=16,
        )
        latents = mx.arange(1 * 2 * 1 * 2 * 2, dtype=mx.float32).reshape(1, 2, 1, 2, 2)

        restored = denormalize_latents(normalize_latents(latents, config), config)

        np.testing.assert_allclose(np.asarray(restored), np.asarray(latents))

    def test_shift_and_sigmas_follow_dynamic_schedule(self) -> None:
        config = SchedulerConfig(
            base_image_seq_len=256,
            base_shift=0.5,
            invert_sigmas=False,
            max_image_seq_len=8192,
            max_shift=0.9,
            num_train_timesteps=1000,
            shift_terminal=0.02,
            time_shift_type="exponential",
            use_dynamic_shifting=True,
        )

        self.assertGreater(
            calculate_shift(4096, config),
            calculate_shift(256, config),
        )
        sigmas = shifted_sigmas(
            num_inference_steps=4,
            image_seq_len=4096,
            config=config,
        )

        self.assertEqual(len(sigmas), 5)
        self.assertEqual(sigmas[-1], 0.0)
        self.assertGreater(sigmas[0], sigmas[-2])


if __name__ == "__main__":
    unittest.main()
