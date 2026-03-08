from __future__ import annotations

import unittest

import mlx.core as mx
from mlxr.families.ltx._generation_backend.rope_ops import (
    LTXRopeType,
    apply_rotary_emb,
    precompute_freqs_cis,
)


class RopeOpsTests(unittest.TestCase):
    def test_precompute_freqs_cis_returns_split_head_shaped_tensors(self) -> None:
        positions = mx.zeros((1, 3, 4, 2), dtype=mx.float32)
        cos_freqs, sin_freqs = precompute_freqs_cis(
            positions,
            dim=64,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=4,
            rope_type="split",
            double_precision=True,
        )

        self.assertEqual(tuple(int(size) for size in cos_freqs.shape), (1, 4, 4, 8))
        self.assertEqual(tuple(int(size) for size in sin_freqs.shape), (1, 4, 4, 8))

    def test_precompute_freqs_cis_returns_interleaved_sequence_tensors(self) -> None:
        positions = mx.zeros((1, 3, 4, 2), dtype=mx.float32)
        cos_freqs, sin_freqs = precompute_freqs_cis(
            positions,
            dim=64,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=4,
            rope_type="interleaved",
            double_precision=False,
        )

        self.assertEqual(tuple(int(size) for size in cos_freqs.shape), (1, 4, 64))
        self.assertEqual(tuple(int(size) for size in sin_freqs.shape), (1, 4, 64))

    def test_apply_rotary_emb_preserves_interleaved_tensor_shape(self) -> None:
        tensor = mx.ones((1, 4, 64), dtype=mx.float32)
        cos_freqs, sin_freqs = precompute_freqs_cis(
            mx.zeros((1, 3, 4, 2), dtype=mx.float32),
            dim=64,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=4,
            rope_type="interleaved",
            double_precision=False,
        )

        rotated = apply_rotary_emb(
            tensor, (cos_freqs, sin_freqs), LTXRopeType.INTERLEAVED
        )

        self.assertEqual(tuple(int(size) for size in rotated.shape), (1, 4, 64))

    def test_apply_rotary_emb_preserves_split_tensor_shape(self) -> None:
        tensor = mx.ones((1, 4, 32), dtype=mx.float32)
        cos_freqs, sin_freqs = precompute_freqs_cis(
            mx.zeros((1, 1, 4, 2), dtype=mx.float32),
            dim=32,
            theta=10000.0,
            max_pos=[20],
            use_middle_indices_grid=True,
            num_attention_heads=4,
            rope_type="split",
            double_precision=False,
        )

        rotated = apply_rotary_emb(tensor, (cos_freqs, sin_freqs), "split")

        self.assertEqual(tuple(int(size) for size in rotated.shape), (1, 4, 32))


if __name__ == "__main__":
    unittest.main()
