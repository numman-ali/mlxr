from __future__ import annotations

import unittest

import mlx.core as mx
from mlxr.families.qwen_image._generation_backend.config import (
    QwenImageTransformerConfig,
)
from mlxr.families.qwen_image._generation_backend.transformer import (
    QwenImageTransformer2DModel,
)


class QwenImageTransformerTests(unittest.TestCase):
    def test_transformer_forward_preserves_packed_latent_shape(self) -> None:
        model = QwenImageTransformer2DModel(
            QwenImageTransformerConfig(
                attention_head_dim=8,
                axes_dims_rope=(2, 2, 4),
                guidance_embeds=False,
                in_channels=4,
                joint_attention_dim=16,
                num_attention_heads=2,
                num_layers=2,
                out_channels=4,
                patch_size=1,
                pooled_projection_dim=0,
                use_additional_t_cond=False,
                use_layer3d_rope=False,
                zero_cond_t=False,
            )
        )

        hidden_states = mx.zeros((1, 4, 4), dtype=mx.float32)
        encoder_hidden_states = mx.zeros((1, 3, 16), dtype=mx.float32)
        encoder_hidden_states_mask = mx.ones((1, 3), dtype=mx.int32)
        timestep = mx.array([1.0], dtype=mx.float32)

        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            image_shapes=[(1, 2, 2)],
        )

        self.assertEqual(tuple(output.shape), (1, 4, 4))

    def test_transformer_supports_zero_cond_t_edit_modulation(self) -> None:
        model = QwenImageTransformer2DModel(
            QwenImageTransformerConfig(
                attention_head_dim=8,
                axes_dims_rope=(2, 2, 4),
                guidance_embeds=False,
                in_channels=4,
                joint_attention_dim=16,
                num_attention_heads=2,
                num_layers=2,
                out_channels=4,
                patch_size=1,
                pooled_projection_dim=0,
                use_additional_t_cond=False,
                use_layer3d_rope=False,
                zero_cond_t=True,
            )
        )

        hidden_states = mx.zeros((1, 8, 4), dtype=mx.float32)
        encoder_hidden_states = mx.zeros((1, 3, 16), dtype=mx.float32)
        encoder_hidden_states_mask = mx.ones((1, 3), dtype=mx.int32)
        timestep = mx.array([0.75], dtype=mx.float32)

        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            image_shapes=[(1, 2, 2), (1, 2, 2)],
        )

        self.assertEqual(tuple(output.shape), (1, 8, 4))


if __name__ == "__main__":
    unittest.main()
