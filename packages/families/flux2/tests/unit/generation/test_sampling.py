from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlxr.families.flux2._sampling import (
    compute_empirical_mu,
    prepare_latent_images,
    prepare_text_ids,
    scalar_schedule,
    unpack_latent_images,
)


class Flux2SamplingTests(unittest.TestCase):
    def test_prepare_and_unpack_latent_images_round_trip(self) -> None:
        latents = mx.arange(1 * 4 * 4 * 2, dtype=mx.float32).reshape(1, 4, 4, 2)

        packed, image_ids = prepare_latent_images(latents)
        unpacked = unpack_latent_images(
            packed,
            latent_height=4,
            latent_width=4,
            channels=2,
        )

        self.assertEqual(tuple(packed.shape), (1, 16, 2))
        self.assertEqual(tuple(image_ids.shape), (1, 16, 4))
        np.testing.assert_array_equal(np.asarray(unpacked), np.asarray(latents))

    def test_prepare_text_ids_returns_zero_position_tensor(self) -> None:
        text_ids = prepare_text_ids(batch_size=2, sequence_length=5)
        self.assertEqual(tuple(text_ids.shape), (2, 5, 4))
        np.testing.assert_array_equal(np.asarray(text_ids)[:, :, :3], 0)
        np.testing.assert_array_equal(np.asarray(text_ids)[0, :, 3], np.arange(5))

    def test_scalar_schedule_monotonically_descends(self) -> None:
        schedule = scalar_schedule(num_steps=4, image_sequence_length=1024)
        self.assertEqual(len(schedule), 5)
        self.assertGreater(schedule[0], schedule[-1])
        self.assertTrue(
            all(
                left >= right
                for left, right in zip(schedule[:-1], schedule[1:], strict=True)
            )
        )

    def test_compute_empirical_mu_grows_with_sequence_length(self) -> None:
        self.assertGreater(
            compute_empirical_mu(4096, 4),
            compute_empirical_mu(256, 4),
        )


if __name__ == "__main__":
    unittest.main()
