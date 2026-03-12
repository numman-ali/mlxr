from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlxr.core.mlx_models import Qwen3Model, Qwen3TextConfig
from mlxr.core.mlx_models.qwen3_text import MLP, RMSNorm, _attention_mask


class Qwen3TextTests(unittest.TestCase):
    def test_model_preserves_batch_sequence_hidden_shape(self) -> None:
        config = Qwen3TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            use_sliding_window=False,
        )
        model = Qwen3Model(config)
        inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        hidden = model(inputs)
        self.assertEqual(tuple(hidden.shape), (1, 4, 32))

    def test_model_can_return_hidden_state_trace(self) -> None:
        config = Qwen3TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            use_sliding_window=False,
        )
        model = Qwen3Model(config)
        inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
        hidden, hidden_states = model(inputs, return_hidden_states=True)
        self.assertEqual(tuple(hidden.shape), (1, 3, 32))
        self.assertEqual(len(hidden_states), 4)

    def test_model_registers_standard_qwen_parameter_names(self) -> None:
        config = Qwen3TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            use_sliding_window=False,
        )
        model = Qwen3Model(config)
        parameter_tree = tree_flatten(model.parameters(), destination={})
        self.assertIn("embed_tokens.weight", parameter_tree)
        self.assertIn("layers.0.self_attn.q_proj.weight", parameter_tree)
        self.assertIn("layers.0.mlp.gate_proj.weight", parameter_tree)

    def test_rms_norm_matches_weighted_normalization(self) -> None:
        norm = RMSNorm(4, eps=1.0e-6)
        norm.weight = mx.array([1.0, 1.5, 0.5, 2.0], dtype=mx.float32)
        x = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float32)

        output = np.asarray(norm(x))
        x_np = np.asarray(x, dtype=np.float32)
        expected = x_np * (
            1.0 / np.sqrt(np.mean(np.square(x_np), axis=-1, keepdims=True) + 1.0e-6)
        )
        expected = expected * np.asarray(norm.weight)

        np.testing.assert_allclose(output, expected, rtol=1.0e-5, atol=1.0e-5)

    def test_mlp_uses_silu_gating(self) -> None:
        mlp = MLP(dim=2, hidden_dim=2)
        mlp.gate_proj.weight = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
        mlp.up_proj.weight = mx.array([[0.5, 0.0], [0.0, 0.25]], dtype=mx.float32)
        mlp.down_proj.weight = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
        x = mx.array([[2.0, -1.0]], dtype=mx.float32)

        output = np.asarray(mlp(x))
        gate = np.asarray(x)
        up = np.asarray([[1.0, -0.25]], dtype=np.float32)
        expected = (gate * (1.0 / (1.0 + np.exp(-gate)))) * up

        np.testing.assert_allclose(output, expected, rtol=1.0e-5, atol=1.0e-5)

    def test_attention_mask_handles_right_padding(self) -> None:
        hidden = mx.zeros((1, 5, 8), dtype=mx.float32)
        attention_mask = mx.array([[1, 1, 1, 0, 0]], dtype=mx.int32)

        mask = _attention_mask(
            hidden,
            cache=None,
            window_size=None,
            attention_mask=attention_mask,
        )

        expected = np.asarray(
            mx.array(
                [
                    [
                        [
                            [True, False, False, False, False],
                            [True, True, False, False, False],
                            [True, True, True, False, False],
                            [True, True, True, False, False],
                            [True, True, True, False, False],
                        ]
                    ]
                ],
                dtype=mx.bool_,
            )
        )
        np.testing.assert_array_equal(np.asarray(mask), expected)


if __name__ == "__main__":
    unittest.main()
