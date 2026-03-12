from __future__ import annotations

import unittest

import mlx.core as mx
from mlxr.families.qwen_image._generation_backend.autoencoder import (
    QwenImageAutoencoderDecoder,
)
from mlxr.families.qwen_image._generation_backend.config import AutoencoderConfig


class QwenImageAutoencoderTests(unittest.TestCase):
    def test_encoder_compresses_rgb_frame_into_latents(self) -> None:
        decoder = QwenImageAutoencoderDecoder(
            AutoencoderConfig(
                attn_scales=(),
                base_dim=8,
                dim_mult=(1, 2),
                dropout=0.0,
                input_channels=3,
                latents_mean=(0.1, 0.2, 0.3, 0.4),
                latents_std=(1.0, 1.0, 1.0, 1.0),
                num_res_blocks=1,
                temperal_downsample=(False,),
                z_dim=4,
            )
        )

        images = mx.zeros((1, 3, 1, 16, 16), dtype=mx.float32)

        latents = decoder.encode(images)

        self.assertEqual(tuple(latents.shape), (1, 4, 1, 8, 8))

    def test_decoder_expand_latents_into_rgb_frame(self) -> None:
        decoder = QwenImageAutoencoderDecoder(
            AutoencoderConfig(
                attn_scales=(),
                base_dim=8,
                dim_mult=(1, 2),
                dropout=0.0,
                input_channels=3,
                latents_mean=(0.1, 0.2, 0.3, 0.4),
                latents_std=(1.0, 1.0, 1.0, 1.0),
                num_res_blocks=1,
                temperal_downsample=(False,),
                z_dim=4,
            )
        )

        latents = mx.zeros((1, 4, 1, 8, 8), dtype=mx.float32)

        decoded = decoder.decode(latents)

        self.assertEqual(tuple(decoded.shape), (1, 3, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
