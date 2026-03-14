from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlxr.core.mlx_models import Qwen2TextConfig, Qwen2TextModel
from mlxr.core.mlx_models.qwen2_text import MLP, RMSNorm, _attention_mask

from tests.optional_dependencies import HAS_QWEN_TEXT_REFERENCE

if HAS_QWEN_TEXT_REFERENCE:
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel


class Qwen2TextTests(unittest.TestCase):
    def test_model_preserves_batch_sequence_hidden_shape(self) -> None:
        config = Qwen2TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            use_sliding_window=False,
            layer_types=["full_attention", "full_attention"],
        )
        model = Qwen2TextModel(config)
        inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        hidden = model(inputs)
        self.assertEqual(tuple(hidden.shape), (1, 4, 32))

    def test_model_can_return_hidden_state_trace(self) -> None:
        config = Qwen2TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            use_sliding_window=False,
            layer_types=["full_attention", "full_attention"],
        )
        model = Qwen2TextModel(config)
        inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
        hidden, hidden_states = model(inputs, return_hidden_states=True)
        self.assertEqual(tuple(hidden.shape), (1, 3, 32))
        self.assertEqual(len(hidden_states), 4)

    def test_model_registers_standard_qwen_parameter_names(self) -> None:
        config = Qwen2TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            use_sliding_window=False,
            layer_types=["full_attention", "full_attention"],
        )
        model = Qwen2TextModel(config)
        parameter_tree = tree_flatten(model.parameters(), destination={})
        self.assertIn("embed_tokens.weight", parameter_tree)
        self.assertIn("layers.0.self_attn.q_proj.weight", parameter_tree)
        self.assertIn("layers.0.self_attn.q_proj.bias", parameter_tree)
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

    @unittest.skipUnless(
        HAS_QWEN_TEXT_REFERENCE,
        "requires optional torch and transformers reference dependencies",
    )
    def test_model_matches_tiny_official_qwen25_vl_text_forward(self) -> None:
        config = Qwen2TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            use_sliding_window=False,
            sliding_window=None,
            attention_dropout=0.0,
            rope_theta=1_000_000.0,
            layer_types=["full_attention", "full_attention"],
            rope_scaling={
                "type": "default",
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
            },
        )
        torch.manual_seed(0)
        official = Qwen2_5_VLTextModel(config).eval()
        model = Qwen2TextModel(config)
        model.load_weights(
            [
                (name, mx.array(tensor.detach().cpu().numpy()))
                for name, tensor in official.state_dict().items()
            ],
            strict=True,
        )
        mx.eval(model.parameters())

        input_ids = torch.randint(0, 128, (1, 7), dtype=torch.long)
        attention_mask = torch.ones((1, 7), dtype=torch.long)
        with torch.no_grad():
            expected = official(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        actual = model(
            mx.array(input_ids.numpy(), dtype=mx.int32),
            attention_mask=mx.array(attention_mask.numpy(), dtype=mx.int32),
        )

        np.testing.assert_allclose(
            np.asarray(actual.astype(mx.float32)),
            expected.detach().cpu().numpy(),
            atol=1.0e-2,
            rtol=1.0e-2,
        )

    @unittest.skipUnless(
        HAS_QWEN_TEXT_REFERENCE,
        "requires optional torch and transformers reference dependencies",
    )
    def test_model_matches_tiny_official_qwen25_vl_text_with_multimodal_positions(
        self,
    ) -> None:
        config = Qwen2TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            use_sliding_window=False,
            sliding_window=None,
            attention_dropout=0.0,
            rope_theta=1_000_000.0,
            layer_types=["full_attention", "full_attention"],
            rope_scaling={
                "type": "default",
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
            },
        )
        torch.manual_seed(0)
        official = Qwen2_5_VLTextModel(config).eval()
        model = Qwen2TextModel(config)
        model.load_weights(
            [
                (name, mx.array(tensor.detach().cpu().numpy()))
                for name, tensor in official.state_dict().items()
            ],
            strict=True,
        )
        mx.eval(model.parameters())

        input_ids = torch.randint(0, 128, (1, 7), dtype=torch.long)
        attention_mask = torch.ones((1, 7), dtype=torch.long)
        position_ids = torch.tensor(
            [
                [[0, 0, 0, 1, 1, 2, 2]],
                [[0, 1, 2, 0, 1, 0, 1]],
                [[0, 1, 2, 3, 4, 5, 6]],
            ],
            dtype=torch.long,
        )
        inputs_embeds = official.embed_tokens(input_ids)
        with torch.no_grad():
            expected = official(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            ).last_hidden_state
        actual = model(
            mx.array(input_ids.numpy(), dtype=mx.int32),
            input_embeddings=mx.array(inputs_embeds.detach().cpu().numpy()),
            attention_mask=mx.array(attention_mask.numpy(), dtype=mx.int32),
            position_ids=mx.array(position_ids.numpy(), dtype=mx.int32),
        )

        np.testing.assert_allclose(
            np.asarray(actual.astype(mx.float32)),
            expected.detach().cpu().numpy(),
            atol=1.0e-2,
            rtol=1.0e-2,
        )


if __name__ == "__main__":
    unittest.main()
