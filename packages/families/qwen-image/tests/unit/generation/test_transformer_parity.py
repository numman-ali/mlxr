from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
import torch
from diffusers import (
    QwenImageTransformer2DModel as DiffusersQwenImageTransformer2DModel,
)
from mlxr.families.qwen_image._generation_backend.config import (
    QwenImageTransformerConfig,
)
from mlxr.families.qwen_image._generation_backend.transformer import (
    QwenImageTransformer2DModel,
)


class QwenImageTransformerParityTests(unittest.TestCase):
    def test_transformer_matches_tiny_diffusers_forward(self) -> None:
        torch.manual_seed(0)
        diffusers_model = DiffusersQwenImageTransformer2DModel(
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            guidance_embeds=False,
            axes_dims_rope=(2, 2, 4),
            zero_cond_t=False,
            use_additional_t_cond=False,
            use_layer3d_rope=False,
        ).eval()
        mlx_model = QwenImageTransformer2DModel(
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
        mlx_model.load_weights(_transformer_weights(diffusers_model), strict=True)
        mx.eval(mlx_model.parameters())

        hidden_states = torch.randn(1, 4, 4, dtype=torch.float32)
        encoder_hidden_states = torch.randn(1, 3, 16, dtype=torch.float32)
        encoder_hidden_states_mask = torch.ones(1, 3, dtype=torch.int64)
        timestep = torch.tensor([0.75], dtype=torch.float32)

        with torch.no_grad():
            expected = diffusers_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                img_shapes=[[(1, 2, 2)]],
                return_dict=False,
            )[0]

        actual = mlx_model(
            hidden_states=mx.array(hidden_states.numpy()),
            encoder_hidden_states=mx.array(encoder_hidden_states.numpy()),
            encoder_hidden_states_mask=mx.array(
                encoder_hidden_states_mask.numpy(),
                dtype=mx.int32,
            ),
            timestep=mx.array(timestep.numpy()),
            image_shapes=[(1, 2, 2)],
        )

        np.testing.assert_allclose(
            np.asarray(actual),
            expected.detach().cpu().numpy(),
            atol=1.0e-4,
            rtol=1.0e-4,
        )

    def test_transformer_matches_tiny_diffusers_forward_for_zero_cond_t(self) -> None:
        torch.manual_seed(0)
        diffusers_model = DiffusersQwenImageTransformer2DModel(
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            guidance_embeds=False,
            axes_dims_rope=(2, 2, 4),
            zero_cond_t=True,
            use_additional_t_cond=False,
            use_layer3d_rope=False,
        ).eval()
        mlx_model = QwenImageTransformer2DModel(
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
        mlx_model.load_weights(_transformer_weights(diffusers_model), strict=True)
        mx.eval(mlx_model.parameters())

        hidden_states = torch.randn(1, 8, 4, dtype=torch.float32)
        encoder_hidden_states = torch.randn(1, 3, 16, dtype=torch.float32)
        encoder_hidden_states_mask = torch.ones(1, 3, dtype=torch.int64)
        timestep = torch.tensor([0.75], dtype=torch.float32)

        with torch.no_grad():
            expected = diffusers_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                img_shapes=[[(1, 2, 2), (1, 2, 2)]],
                return_dict=False,
            )[0]

        actual = mlx_model(
            hidden_states=mx.array(hidden_states.numpy()),
            encoder_hidden_states=mx.array(encoder_hidden_states.numpy()),
            encoder_hidden_states_mask=mx.array(
                encoder_hidden_states_mask.numpy(),
                dtype=mx.int32,
            ),
            timestep=mx.array(timestep.numpy()),
            image_shapes=[(1, 2, 2), (1, 2, 2)],
        )

        np.testing.assert_allclose(
            np.asarray(actual),
            expected.detach().cpu().numpy(),
            atol=1.0e-4,
            rtol=1.0e-4,
        )


def _transformer_weights(
    model: DiffusersQwenImageTransformer2DModel,
) -> list[tuple[str, mx.array]]:
    alias_map = {
        "time_text_embed.timestep_embedder.": "time_text_embed.",
        ".img_mod.1.": ".img_mod.",
        ".txt_mod.1.": ".txt_mod.",
        ".img_mlp.net.0.proj.": ".img_mlp.proj_in.",
        ".img_mlp.net.2.": ".img_mlp.proj_out.",
        ".txt_mlp.net.0.proj.": ".txt_mlp.proj_in.",
        ".txt_mlp.net.2.": ".txt_mlp.proj_out.",
        ".to_out.0.": ".to_out.",
    }
    weights: list[tuple[str, mx.array]] = []
    for name, tensor in model.state_dict().items():
        aliased = name
        for old, new in alias_map.items():
            aliased = aliased.replace(old, new)
        weights.append((aliased, mx.array(tensor.detach().cpu().numpy())))
    return weights


if __name__ == "__main__":
    unittest.main()
