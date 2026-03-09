from __future__ import annotations

import unittest

import mlx.core as mx
from mlxr.families.ltx import _nn_compat as nn
from mlxr.families.ltx._generation_backend.adaln_ops import (
    AdaLayerNormSingle,
    adaln_embedding_coefficient,
)
from mlxr.families.ltx._generation_backend.transformer_preprocessors import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgsPreprocessor,
)
from mlxr.families.ltx._generation_backend.types import _PatchedModality


class TransformerPreprocessorTests(unittest.TestCase):
    def test_adaln_coefficient_matches_cross_attention_mode(self) -> None:
        self.assertEqual(adaln_embedding_coefficient(False), 6)
        self.assertEqual(adaln_embedding_coefficient(True), 9)

    def test_transformer_preprocessor_requires_post_connector_width(self) -> None:
        preprocessor = TransformerArgsPreprocessor(
            patchify_proj=nn.Linear(8, 8),
            adaln=AdaLayerNormSingle(8),
            inner_dim=8,
            max_pos=[20],
            num_attention_heads=2,
            use_middle_indices_grid=True,
            timestep_scale_multiplier=1000,
            positional_embedding_theta=10000.0,
            rope_type="split",
            caption_projection=None,
        )
        modality = _PatchedModality(
            latent=mx.ones((1, 2, 8), dtype=mx.float32),
            sigma=mx.ones((1,), dtype=mx.float32),
            timesteps=mx.ones((1,), dtype=mx.float32),
            positions=mx.zeros((1, 1, 2, 2), dtype=mx.float32),
            context=mx.ones((1, 2, 4), dtype=mx.float32),
        )
        with self.assertRaisesRegex(ValueError, "post-connector"):
            preprocessor.prepare(modality)

    def test_transformer_preprocessor_emits_prompt_timestep_when_prompt_adaln_present(
        self,
    ) -> None:
        preprocessor = TransformerArgsPreprocessor(
            patchify_proj=nn.Linear(8, 8),
            adaln=AdaLayerNormSingle(8),
            inner_dim=8,
            max_pos=[20],
            num_attention_heads=2,
            use_middle_indices_grid=True,
            timestep_scale_multiplier=1000,
            positional_embedding_theta=10000.0,
            rope_type="split",
            caption_projection=None,
            prompt_adaln=AdaLayerNormSingle(8, embedding_coefficient=2),
        )
        modality = _PatchedModality(
            latent=mx.ones((1, 2, 8), dtype=mx.float32),
            sigma=mx.ones((1,), dtype=mx.float32),
            timesteps=mx.ones((1,), dtype=mx.float32),
            positions=mx.zeros((1, 1, 2, 2), dtype=mx.float32),
            context=mx.ones((1, 2, 8), dtype=mx.float32),
        )
        result = preprocessor.prepare(modality)
        self.assertIsNotNone(result.prompt_timestep)

    def test_transformer_preprocessor_prepares_self_attention_mask(self) -> None:
        preprocessor = TransformerArgsPreprocessor(
            patchify_proj=nn.Linear(8, 8),
            adaln=AdaLayerNormSingle(8),
            inner_dim=8,
            max_pos=[20],
            num_attention_heads=2,
            use_middle_indices_grid=True,
            timestep_scale_multiplier=1000,
            positional_embedding_theta=10000.0,
            rope_type="split",
            caption_projection=None,
        )
        modality = _PatchedModality(
            latent=mx.ones((1, 2, 8), dtype=mx.float32),
            sigma=mx.ones((1,), dtype=mx.float32),
            timesteps=mx.ones((1,), dtype=mx.float32),
            positions=mx.zeros((1, 1, 2, 2), dtype=mx.float32),
            context=mx.ones((1, 2, 8), dtype=mx.float32),
            attention_mask=mx.array([[[1.0, 0.5], [0.0, 1.0]]], dtype=mx.float32),
        )
        result = preprocessor.prepare(modality)
        self.assertIsNotNone(result.self_attention_mask)
        assert result.self_attention_mask is not None
        self.assertEqual(
            tuple(int(x) for x in result.self_attention_mask.shape), (1, 1, 2, 2)
        )
        self.assertAlmostEqual(
            float(result.self_attention_mask[0, 0, 0, 0].item()), 0.0, places=5
        )
        self.assertLess(float(result.self_attention_mask[0, 0, 1, 0].item()), -1e20)

    def test_multimodal_preprocessor_adds_cross_attention_payloads(self) -> None:
        preprocessor = MultiModalTransformerArgsPreprocessor(
            patchify_proj=nn.Linear(8, 8),
            adaln=AdaLayerNormSingle(8),
            cross_scale_shift_adaln=AdaLayerNormSingle(8, embedding_coefficient=4),
            cross_gate_adaln=AdaLayerNormSingle(8, embedding_coefficient=1),
            inner_dim=8,
            max_pos=[20],
            num_attention_heads=2,
            cross_pe_max_pos=20,
            use_middle_indices_grid=True,
            audio_cross_attention_dim=4,
            timestep_scale_multiplier=1000,
            positional_embedding_theta=10000.0,
            rope_type="split",
            av_ca_timestep_scale_multiplier=1,
            caption_projection=None,
        )
        modality = _PatchedModality(
            latent=mx.ones((1, 2, 8), dtype=mx.float32),
            sigma=mx.ones((1,), dtype=mx.float32),
            timesteps=mx.ones((1,), dtype=mx.float32),
            positions=mx.zeros((1, 1, 2, 2), dtype=mx.float32),
            context=mx.ones((1, 2, 8), dtype=mx.float32),
        )
        result = preprocessor.prepare(modality, modality)
        self.assertIsNotNone(result.cross_positional_embeddings)
        self.assertIsNotNone(result.cross_scale_shift_timestep)
        self.assertIsNotNone(result.cross_gate_timestep)

    def test_multimodal_preprocessor_uses_cross_modality_sigma(self) -> None:
        preprocessor = MultiModalTransformerArgsPreprocessor(
            patchify_proj=nn.Linear(8, 8),
            adaln=AdaLayerNormSingle(8),
            cross_scale_shift_adaln=AdaLayerNormSingle(8, embedding_coefficient=4),
            cross_gate_adaln=AdaLayerNormSingle(8, embedding_coefficient=1),
            inner_dim=8,
            max_pos=[20],
            num_attention_heads=2,
            cross_pe_max_pos=20,
            use_middle_indices_grid=True,
            audio_cross_attention_dim=4,
            timestep_scale_multiplier=1000,
            positional_embedding_theta=10000.0,
            rope_type="split",
            av_ca_timestep_scale_multiplier=1,
            caption_projection=None,
        )
        modality = _PatchedModality(
            latent=mx.ones((1, 2, 8), dtype=mx.float32),
            sigma=mx.array([0.9], dtype=mx.float32),
            timesteps=mx.ones((1,), dtype=mx.float32),
            positions=mx.zeros((1, 1, 2, 2), dtype=mx.float32),
            context=mx.ones((1, 2, 8), dtype=mx.float32),
        )
        cross_modality = _PatchedModality(
            latent=mx.ones((1, 2, 8), dtype=mx.float32),
            sigma=mx.array([0.25], dtype=mx.float32),
            timesteps=mx.ones((1,), dtype=mx.float32) * 99,
            positions=mx.zeros((1, 1, 2, 2), dtype=mx.float32),
            context=mx.ones((1, 2, 8), dtype=mx.float32),
        )
        result = preprocessor.prepare(modality, cross_modality)
        baseline = preprocessor.prepare(modality, modality)
        self.assertFalse(
            mx.allclose(
                result.cross_scale_shift_timestep, baseline.cross_scale_shift_timestep
            ).item()
        )


if __name__ == "__main__":
    unittest.main()
