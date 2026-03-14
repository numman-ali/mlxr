from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlxr.core.mlx_models import Qwen2VLVisionConfig, Qwen2VLVisionModel

from tests.optional_dependencies import HAS_QWEN_VISION_REFERENCE

if HAS_QWEN_VISION_REFERENCE:
    import torch
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLVisionConfig,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )


class Qwen2VLVisionTests(unittest.TestCase):
    def test_model_preserves_expected_output_shape(self) -> None:
        config = Qwen2VLVisionConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=1,
            window_size=8,
            out_hidden_size=64,
            fullatt_block_indexes=(0, 1),
        )
        model = Qwen2VLVisionModel(config)
        hidden_states = mx.zeros((16, 12), dtype=mx.float32)
        grid_thw = mx.array([[1, 4, 4]], dtype=mx.int32)

        output = model(hidden_states, grid_thw=grid_thw)

        self.assertEqual(tuple(output.shape), (4, 64))

    @unittest.skipUnless(
        HAS_QWEN_VISION_REFERENCE,
        "requires optional torch and transformers reference dependencies",
    )
    def test_model_matches_tiny_official_qwen25_vl_vision_forward(self) -> None:
        config = Qwen2_5_VLVisionConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=1,
            window_size=8,
            out_hidden_size=64,
            fullatt_block_indexes=[0, 1],
        )
        torch.manual_seed(0)
        official = Qwen2_5_VisionTransformerPretrainedModel(config).eval()
        model = Qwen2VLVisionModel(Qwen2VLVisionConfig.from_dict(config.to_dict()))
        model.load_weights(_vision_weights(official), strict=True)
        mx.eval(model.parameters())

        hidden_states = torch.randn(16, 12, dtype=torch.float32)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
        with torch.no_grad():
            expected = official(
                hidden_states,
                grid_thw=grid_thw,
                return_dict=True,
            ).pooler_output

        actual = model(
            mx.array(hidden_states.numpy(), dtype=mx.float32),
            grid_thw=mx.array(grid_thw.numpy(), dtype=mx.int32),
        )

        np.testing.assert_allclose(
            np.asarray(actual.astype(mx.float32)),
            expected.detach().cpu().numpy(),
            atol=1.0e-4,
            rtol=1.0e-4,
        )


def _vision_weights(
    model: Qwen2_5_VisionTransformerPretrainedModel,
) -> list[tuple[str, mx.array]]:
    weights: list[tuple[str, mx.array]] = []
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().numpy()
        name = name.replace("merger.mlp.0.", "merger.mlp0.")
        name = name.replace("merger.mlp.2.", "merger.mlp2.")
        if name == "patch_embed.proj.weight":
            value = value.reshape(value.shape[0], -1)
        weights.append((name, mx.array(value)))
    return weights


if __name__ == "__main__":
    unittest.main()
