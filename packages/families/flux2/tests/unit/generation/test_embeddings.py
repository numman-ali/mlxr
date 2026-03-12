from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlxr.families.flux2._generation_backend import (
    FinalModulation,
    TimestepGuidanceEmbeddings,
    _silu,
    _timestep_embedding,
)


class Flux2EmbeddingTests(unittest.TestCase):
    def test_timestep_guidance_embeddings_adds_guidance_only_when_enabled(self) -> None:
        module = TimestepGuidanceEmbeddings(
            in_channels=4,
            hidden_size=4,
            guidance_embeds=True,
        )
        timestep = mx.array([0.25], dtype=mx.float32)
        guidance = mx.array([1.5], dtype=mx.float32)

        without_guidance = np.asarray(module(timestep, None))
        with_guidance = np.asarray(module(timestep, guidance))

        self.assertEqual(without_guidance.shape, (1, 4))
        self.assertEqual(with_guidance.shape, (1, 4))
        self.assertFalse(np.allclose(with_guidance, without_guidance))

    def test_final_modulation_returns_scale_then_shift(self) -> None:
        modulation = FinalModulation(hidden_size=2)
        modulation.linear.weight = mx.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=mx.float32,
        )
        vec = mx.array([[1.0, 2.0]], dtype=mx.float32)

        scale, shift = modulation(vec)
        projected = modulation.linear(_silu(vec))
        expected_scale, expected_shift = mx.split(projected, 2, axis=-1)

        np.testing.assert_allclose(np.asarray(scale), np.asarray(expected_scale))
        np.testing.assert_allclose(np.asarray(shift), np.asarray(expected_shift))

    def test_timestep_embedding_matches_expected_flip_sin_to_cos_layout(self) -> None:
        timestep = mx.array([0.5], dtype=mx.float32)
        embedding = np.asarray(
            _timestep_embedding(
                timestep,
                4,
                flip_sin_to_cos=True,
                downscale_freq_shift=0.0,
            )
        )
        raw = np.asarray(
            _timestep_embedding(
                timestep,
                4,
                flip_sin_to_cos=False,
                downscale_freq_shift=0.0,
            )
        )

        np.testing.assert_allclose(embedding[:, :2], raw[:, 2:])
        np.testing.assert_allclose(embedding[:, 2:], raw[:, :2])


if __name__ == "__main__":
    unittest.main()
