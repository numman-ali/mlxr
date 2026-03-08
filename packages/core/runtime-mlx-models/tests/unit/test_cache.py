from __future__ import annotations

import unittest

import mlx.core as mx
from mlxr.core.mlx_models.cache import KVCache, RotatingKVCache


class KVCacheTests(unittest.TestCase):
    def test_update_and_fetch_initializes_and_grows_storage(self) -> None:
        cache = KVCache()
        keys = mx.ones((1, 2, 3, 4), dtype=mx.float32)
        values = mx.ones((1, 2, 3, 5), dtype=mx.float32)

        stored_keys, stored_values = cache.update_and_fetch(keys, values)
        self.assertEqual(stored_keys.shape, (1, 2, 3, 4))
        self.assertEqual(stored_values.shape, (1, 2, 3, 5))
        self.assertEqual(cache.offset, 3)

        extra_keys = mx.full((1, 2, 260, 4), 2.0, dtype=mx.float32)
        extra_values = mx.full((1, 2, 260, 5), 3.0, dtype=mx.float32)
        stored_keys, stored_values = cache.update_and_fetch(extra_keys, extra_values)
        self.assertEqual(stored_keys.shape, (1, 2, 263, 4))
        self.assertEqual(stored_values.shape, (1, 2, 263, 5))
        self.assertEqual(cache.offset, 263)

    def test_make_mask_handles_empty_and_populated_cache(self) -> None:
        cache = KVCache()
        initial_mask = cache.make_mask(3, return_array=True, window_size=None)
        self.assertIsInstance(initial_mask, mx.array)
        assert isinstance(initial_mask, mx.array)
        self.assertEqual(initial_mask.shape, (3, 3))

        cache.update_and_fetch(
            mx.ones((1, 1, 2, 4), dtype=mx.float32),
            mx.ones((1, 1, 2, 4), dtype=mx.float32),
        )
        self.assertEqual(cache.make_mask(1, return_array=False, window_size=None), None)
        self.assertEqual(
            cache.make_mask(2, return_array=False, window_size=None), "causal"
        )


class RotatingKVCacheTests(unittest.TestCase):
    def test_single_token_updates_use_in_place_path(self) -> None:
        cache = RotatingKVCache(max_size=4, keep=1)
        keys = mx.arange(8, dtype=mx.float32).reshape(1, 1, 2, 4)
        values = keys

        first_keys, first_values = cache.update_and_fetch(
            keys[:, :, :1, :],
            values[:, :, :1, :],
        )
        self.assertEqual(first_keys.shape, (1, 1, 1, 4))
        self.assertEqual(first_values.shape, (1, 1, 1, 4))

        second_keys, second_values = cache.update_and_fetch(
            keys[:, :, 1:, :],
            values[:, :, 1:, :],
        )
        self.assertEqual(second_keys.shape, (1, 1, 2, 4))
        self.assertEqual(second_values.shape, (1, 1, 2, 4))
        self.assertEqual(cache.offset, 2)

    def test_multi_token_updates_trim_and_rotate(self) -> None:
        cache = RotatingKVCache(max_size=3, keep=1)
        keys = mx.arange(16, dtype=mx.float32).reshape(1, 1, 4, 4)
        values = keys

        stored_keys, stored_values = cache.update_and_fetch(
            keys[:, :, :2, :], values[:, :, :2, :]
        )
        self.assertEqual(stored_keys.shape, (1, 1, 2, 4))
        self.assertEqual(stored_values.shape, (1, 1, 2, 4))

        stored_keys, stored_values = cache.update_and_fetch(
            keys[:, :, 2:, :], values[:, :, 2:, :]
        )
        self.assertEqual(stored_keys.shape, (1, 1, 4, 4))
        self.assertEqual(stored_values.shape, (1, 1, 4, 4))
        self.assertEqual(cache.offset, 4)

    def test_make_mask_uses_effective_offset(self) -> None:
        cache = RotatingKVCache(max_size=3, keep=1)
        cache.update_and_fetch(
            mx.ones((1, 1, 1, 2), dtype=mx.float32),
            mx.ones((1, 1, 1, 2), dtype=mx.float32),
        )
        self.assertIsNone(cache.make_mask(1, return_array=False, window_size=None))
        mask = cache.make_mask(2, return_array=True, window_size=2)
        self.assertIsInstance(mask, mx.array)
        assert isinstance(mask, mx.array)
        self.assertEqual(mask.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
