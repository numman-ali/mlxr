from __future__ import annotations

import unittest

import mlx.core as mx
from mlxr.families.ltx._generation_backend.model_config import TransformerConfig
from mlxr.families.ltx._generation_backend.transformer_blocks import (
    BasicAVTransformerBlock,
)
from mlxr.families.ltx._generation_backend.types import _PatchedTransformerArgs


class TransformerBlockTests(unittest.TestCase):
    def test_block_enables_gated_attention_and_prompt_adaln(self) -> None:
        config = TransformerConfig(
            dim=8,
            heads=2,
            d_head=4,
            context_dim=8,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        )
        block = BasicAVTransformerBlock(idx=0, video=config)
        self.assertIsNotNone(block.attn1.to_gate_logits)
        self.assertEqual(
            tuple(int(size) for size in block.scale_shift_table.shape), (9, 8)
        )
        self.assertEqual(
            tuple(int(size) for size in block.prompt_scale_shift_table.shape),
            (2, 8),
        )

    def test_block_preserves_video_tensor_shape(self) -> None:
        config = TransformerConfig(
            dim=8,
            heads=2,
            d_head=4,
            context_dim=8,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        )
        block = BasicAVTransformerBlock(idx=0, video=config)
        args = _PatchedTransformerArgs(
            x=mx.ones((1, 2, 8), dtype=mx.float32),
            context=mx.ones((1, 2, 8), dtype=mx.float32),
            context_mask=None,
            self_attention_mask=None,
            timesteps=mx.ones((1, 1, 72), dtype=mx.float32),
            embedded_timestep=mx.ones((1, 1, 8), dtype=mx.float32),
            positional_embeddings=(
                mx.ones((1, 2, 8), dtype=mx.float32),
                mx.zeros((1, 2, 8), dtype=mx.float32),
            ),
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=True,
            prompt_timestep=mx.ones((1, 1, 16), dtype=mx.float32),
        )
        video_out, audio_out = block(video=args, audio=None)
        self.assertIsNone(audio_out)
        assert video_out is not None
        self.assertEqual(tuple(int(size) for size in video_out.x.shape), (1, 2, 8))


if __name__ == "__main__":
    unittest.main()
