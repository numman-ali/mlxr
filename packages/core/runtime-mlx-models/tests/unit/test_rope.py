from __future__ import annotations

import unittest

import mlx.core as mx
from mlxr.core.mlx_models.rope import (
    Llama3RoPE,
    SuScaledRoPE,
    YarnRoPE,
    _require_float,
    _require_float_list,
    _require_int,
    initialize_rope,
)


class RopeHelpersTests(unittest.TestCase):
    def test_require_helpers_validate_types(self) -> None:
        self.assertEqual(_require_float(3, context="x"), 3.0)
        self.assertEqual(_require_int(4.0, context="x"), 4)
        self.assertEqual(_require_float_list([1, 2.5], context="x"), [1.0, 2.5])
        with self.assertRaises(ValueError):
            _require_float("bad", context="x")
        with self.assertRaises(ValueError):
            _require_int(True, context="x")
        with self.assertRaises(ValueError):
            _require_float_list(["bad"], context="x")

    def test_default_and_linear_initialize_rope(self) -> None:
        default_rope = initialize_rope(dims=8, base=10000.0, traditional=False)
        linear_rope = initialize_rope(
            dims=8,
            base=10000.0,
            traditional=False,
            scaling_config={"type": "linear", "factor": 2.0},
        )
        x = mx.ones((1, 2, 1, 8), dtype=mx.float32)
        self.assertEqual(default_rope(x).shape, x.shape)
        self.assertEqual(linear_rope(x).shape, x.shape)

    def test_su_scaled_rope_returns_same_shape(self) -> None:
        rope = SuScaledRoPE(
            8,
            max_position_embeddings=8192,
            original_max_position_embeddings=4096,
            long_factor=[1.0, 1.0, 1.0, 1.0],
        )
        x = mx.ones((1, 2, 1, 8), dtype=mx.float32)
        self.assertEqual(rope(x).shape, x.shape)

    def test_llama3_rope_requires_scaling_metadata(self) -> None:
        rope = Llama3RoPE(
            8,
            scaling_config={
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        )
        x = mx.ones((1, 2, 1, 8), dtype=mx.float32)
        self.assertEqual(rope(x).shape, x.shape)

        with self.assertRaises(ValueError):
            initialize_rope(
                dims=8,
                base=10000.0,
                traditional=False,
                scaling_config={"type": "llama3", "factor": 8.0},
            )

    def test_yarn_rope_and_initialize_variant(self) -> None:
        rope = YarnRoPE(
            8,
            scaling_factor=2.0,
            original_max_position_embeddings=4096,
        )
        x = mx.ones((1, 2, 1, 8), dtype=mx.float32)
        self.assertEqual(rope(x).shape, x.shape)

        initialized = initialize_rope(
            dims=8,
            base=10000.0,
            traditional=False,
            scaling_config={
                "type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4096,
            },
        )
        self.assertEqual(initialized(x).shape, x.shape)

    def test_initialize_rope_rejects_unknown_type(self) -> None:
        with self.assertRaises(ValueError):
            initialize_rope(
                dims=8,
                base=10000.0,
                traditional=False,
                scaling_config={"type": "unknown"},
            )


if __name__ == "__main__":
    unittest.main()
