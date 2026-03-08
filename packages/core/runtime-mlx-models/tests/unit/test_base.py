from __future__ import annotations

import unittest

import mlx.core as mx
from mlxr.core.mlx_models.base import create_attention_mask, create_causal_mask
from mlxr.core.mlx_models.cache import KVCache


class BaseModelHelpersTests(unittest.TestCase):
    def test_create_causal_mask_applies_window_and_padding(self) -> None:
        mask = create_causal_mask(
            4,
            offset=2,
            window_size=2,
            right_padding=mx.array([1], dtype=mx.int32),
            left_padding=mx.array([1], dtype=mx.int32),
        )
        self.assertEqual(mask.shape, (1, 1, 4, 6))
        self.assertEqual(mask.dtype, mx.bool_)

    def test_create_attention_mask_without_cache_uses_fast_paths(self) -> None:
        single_hidden = mx.zeros((1, 1, 4), dtype=mx.float32)
        self.assertIsNone(create_attention_mask(single_hidden))

        hidden = mx.zeros((1, 3, 4), dtype=mx.float32)
        self.assertEqual(create_attention_mask(hidden), "causal")

        mask = create_attention_mask(hidden, return_array=True)
        self.assertIsInstance(mask, mx.array)
        assert isinstance(mask, mx.array)
        self.assertEqual(mask.shape, (3, 3))

    def test_create_attention_mask_uses_cache_factory_when_available(self) -> None:
        cache = KVCache()
        mask = create_attention_mask(
            mx.zeros((1, 2, 4), dtype=mx.float32),
            cache,
            return_array=True,
        )
        self.assertIsInstance(mask, mx.array)
        assert isinstance(mask, mx.array)
        self.assertEqual(mask.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
