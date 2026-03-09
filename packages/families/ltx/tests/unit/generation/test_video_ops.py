from __future__ import annotations

import unittest

import mlx.core as mx
from mlxr.families.ltx._generation_backend.video_encoder import patchify_video
from mlxr.families.ltx._generation_backend.video_ops import unpatchify_video


class VideoOpsTests(unittest.TestCase):
    def test_patchify_unpatchify_round_trip_for_video_tensor(self) -> None:
        original = mx.arange(1 * 3 * 2 * 4 * 6, dtype=mx.float32)
        original = mx.reshape(original, (1, 3, 2, 4, 6))

        patched = patchify_video(original, patch_size_hw=2, patch_size_t=1)
        restored = unpatchify_video(patched, patch_size_hw=2, patch_size_t=1)

        self.assertEqual(restored.shape, original.shape)
        self.assertTrue(bool(mx.array_equal(restored, original).item()))


if __name__ == "__main__":
    unittest.main()
