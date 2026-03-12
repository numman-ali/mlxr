from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlxr.families.flux2._generation_backend.autoencoder import (
    AttentionBlock2D,
    AutoencoderKLFlux2,
    DownEncoderBlock2D,
    Downsample2D,
    ResnetBlock2D,
    UNetMidBlock2D,
    UpDecoderBlock2D,
    Upsample2D,
)

from ._fixtures import tiny_autoencoder_config


class Flux2AutoencoderTests(unittest.TestCase):
    def test_basic_blocks_preserve_expected_shapes(self) -> None:
        x = mx.random.normal((1, 8, 8, 32), dtype=mx.float32)

        self.assertEqual(ResnetBlock2D(32, 32, 32)(x).shape, x.shape)
        self.assertEqual(AttentionBlock2D(32)(x).shape, x.shape)
        self.assertEqual(Downsample2D(32)(x).shape, (1, 4, 4, 32))
        self.assertEqual(Upsample2D(32)(x).shape, (1, 16, 16, 32))
        self.assertEqual(UNetMidBlock2D(32, 32)(x).shape, x.shape)

    def test_encoder_decoder_blocks_change_resolution_and_channels(self) -> None:
        encoded = DownEncoderBlock2D(
            32,
            64,
            num_layers=1,
            add_downsample=True,
            groups=32,
        )(mx.random.normal((1, 8, 8, 32), dtype=mx.float32))
        decoded = UpDecoderBlock2D(
            64,
            32,
            num_layers=1,
            add_upsample=True,
            groups=32,
        )(mx.random.normal((1, 8, 8, 64), dtype=mx.float32))

        self.assertEqual(encoded.shape, (1, 4, 4, 64))
        self.assertEqual(decoded.shape, (1, 16, 16, 32))

    def test_autoencoder_normalize_inverse_and_roundtrip_shapes(self) -> None:
        autoencoder = AutoencoderKLFlux2(tiny_autoencoder_config())
        autoencoder.bn.running_mean = mx.arange(16, dtype=mx.float32)
        autoencoder.bn.running_var = mx.full((16,), 4.0, dtype=mx.float32)
        latent = mx.random.normal((1, 8, 8, 16), dtype=mx.float32)

        normalized = autoencoder.normalize(latent)
        restored = autoencoder.inv_normalize(normalized)
        encoded = autoencoder.encode(mx.random.normal((1, 32, 32, 3), dtype=mx.float32))
        decoded = autoencoder.decode(encoded)

        np.testing.assert_allclose(
            np.asarray(restored),
            np.asarray(latent),
            rtol=1.0e-4,
            atol=1.0e-4,
        )
        self.assertEqual(encoded.shape, (1, 8, 8, 16))
        self.assertEqual(decoded.shape, (1, 32, 32, 3))


if __name__ == "__main__":
    unittest.main()
