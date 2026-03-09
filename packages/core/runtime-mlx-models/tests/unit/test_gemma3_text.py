from __future__ import annotations

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten
from mlxr.core.mlx_models import (
    Gemma3Model,
    TextConfig,
    create_causal_mask,
)


class Gemma3TextTests(unittest.TestCase):
    def test_create_causal_mask_supports_left_padding(self) -> None:
        left_padding = mx.array([2], dtype=mx.int32)
        mask = create_causal_mask(4, left_padding=left_padding)
        self.assertEqual(tuple(mask.shape), (1, 1, 4, 4))
        self.assertFalse(bool(mask[0, 0, 0, 0]))
        self.assertFalse(bool(mask[0, 0, 1, 1]))
        self.assertTrue(bool(mask[0, 0, 2, 2]))

    def test_model_preserves_batch_sequence_hidden_shape(self) -> None:
        config = TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            sliding_window=16,
            _sliding_window_pattern=2,
            rope_parameters=None,
        )
        model = Gemma3Model(config)
        inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        hidden = model(inputs)
        self.assertEqual(tuple(hidden.shape), (1, 4, 32))

    def test_model_uses_explicit_layer_types_when_present(self) -> None:
        config = TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            sliding_window=16,
            _sliding_window_pattern=99,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            rope_parameters=None,
        )
        model = Gemma3Model(config)

        self.assertEqual(
            [layer.self_attn.is_sliding for layer in model.layers],
            [True, False, True, False],
        )

    def test_model_registers_embed_tokens_weight(self) -> None:
        config = TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            sliding_window=16,
            _sliding_window_pattern=2,
            rope_parameters=None,
        )
        model = Gemma3Model(config)

        parameter_tree = tree_flatten(model.parameters(), destination={})

        self.assertIn("embed_tokens.weight", parameter_tree)


if __name__ == "__main__":
    unittest.main()
