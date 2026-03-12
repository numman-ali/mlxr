from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlxr.families.flux2._generation_backend.embeddings import (
    EmbedND,
    TimestepProjection,
)
from mlxr.families.flux2._generation_backend.transformer import (
    DoubleStreamAttention,
    DoubleStreamBlock,
    FeedForward,
    Flux2KVCache,
    Flux2KVLayerCache,
    Flux2Transformer2DModel,
    Modulation,
    RMSNorm,
    SingleStreamAttention,
    SingleStreamBlock,
    _apply_rope,
    _rope,
)

from ._fixtures import tiny_transformer_config


class Flux2TransformerTests(unittest.TestCase):
    def test_rope_and_apply_rope_preserve_shape(self) -> None:
        positions = mx.zeros((1, 4), dtype=mx.int32)
        rope = _rope(positions, dim=2, theta=10_000.0)
        values = mx.arange(16, dtype=mx.float32).reshape(1, 2, 4, 2)

        rotated = _apply_rope(values, rope)

        self.assertEqual(rotated.shape, values.shape)
        np.testing.assert_allclose(np.asarray(rotated), np.asarray(values))

    def test_embedding_and_projection_helpers_return_expected_shapes(self) -> None:
        ids = mx.zeros((1, 5, 2), dtype=mx.int32)
        rope = EmbedND(theta=10_000.0, axes_dims=(2, 2))(ids)
        projected = TimestepProjection(8, 8)(mx.ones((1, 8), dtype=mx.float32))
        modulation = Modulation(8, double=True)(mx.ones((1, 8), dtype=mx.float32))

        self.assertEqual(rope.shape, (1, 1, 5, 2, 2, 2))
        self.assertEqual(projected.shape, (1, 8))
        self.assertEqual(len(modulation), 6)

    def test_norm_feedforward_attention_and_blocks_preserve_shapes(self) -> None:
        tokens = mx.random.normal((1, 5, 8), dtype=mx.float32)
        context = mx.random.normal((1, 3, 8), dtype=mx.float32)
        rope = mx.broadcast_to(
            _rope(mx.zeros((1, 8), dtype=mx.int32), dim=2, theta=10_000.0)[:, None],
            (1, 2, 8, 1, 2, 2),
        )
        vec = mx.ones((1, 8), dtype=mx.float32)

        normalized = RMSNorm(4)(mx.random.normal((1, 2, 5, 4), dtype=mx.float32))
        fed = FeedForward(8, 16)(tokens)
        attended_img, attended_ctx = DoubleStreamAttention(8, 2)(
            tokens,
            context,
            rope,
        )
        block_img, block_ctx = DoubleStreamBlock(8, 2, 2.0)(
            tokens,
            context,
            rope,
            Modulation(8, double=True)(vec),
            Modulation(8, double=True)(vec),
        )
        single_attended = SingleStreamAttention(8, 2)(
            mx.concatenate([context, tokens], axis=1),
            rope,
        )
        single_block = SingleStreamBlock(8, 2)(
            mx.concatenate([context, tokens], axis=1),
            rope,
            Modulation(8, double=False)(vec),
        )

        self.assertEqual(normalized.shape, (1, 2, 5, 4))
        self.assertEqual(fed.shape, tokens.shape)
        self.assertEqual(attended_img.shape, tokens.shape)
        self.assertEqual(attended_ctx.shape, context.shape)
        self.assertEqual(block_img.shape, tokens.shape)
        self.assertEqual(block_ctx.shape, context.shape)
        self.assertEqual(single_attended.shape, (1, 8, 8))
        self.assertEqual(single_block.shape, (1, 8, 8))

    def test_transformer_runs_with_and_without_guidance_embeddings(self) -> None:
        x = mx.random.normal((1, 5, 8), dtype=mx.float32)
        ctx = mx.random.normal((1, 3, 12), dtype=mx.float32)
        x_ids = mx.zeros((1, 5, 2), dtype=mx.int32)
        ctx_ids = mx.zeros((1, 3, 2), dtype=mx.int32)

        guided = Flux2Transformer2DModel(tiny_transformer_config(guidance_embeds=True))(
            x=x,
            x_ids=x_ids,
            timesteps=mx.array([0.5], dtype=mx.float32),
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=mx.array([1.0], dtype=mx.float32),
        )
        unguided = Flux2Transformer2DModel(
            tiny_transformer_config(guidance_embeds=False)
        )(
            x=x,
            x_ids=x_ids,
            timesteps=mx.array([0.5], dtype=mx.float32),
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=None,
        )

        self.assertEqual(guided.shape, (1, 5, 8))
        self.assertEqual(unguided.shape, (1, 5, 8))

    def test_kv_cache_helpers_store_and_clear(self) -> None:
        layer_cache = Flux2KVLayerCache()
        k_ref = mx.ones((1, 2, 3, 4), dtype=mx.float32)
        v_ref = mx.zeros((1, 2, 3, 4), dtype=mx.float32)
        layer_cache.store(k_ref=k_ref, v_ref=v_ref)

        stored_k, stored_v = layer_cache.get()
        self.assertEqual(stored_k.shape, (1, 2, 3, 4))
        self.assertEqual(stored_v.shape, (1, 2, 3, 4))

        cache = Flux2KVCache.create(
            num_double_layers=1,
            num_single_layers=1,
            num_ref_tokens=3,
        )
        cache.double_block_caches[0].store(k_ref=k_ref, v_ref=v_ref)
        cache.single_block_caches[0].store(k_ref=k_ref, v_ref=v_ref)
        self.assertEqual(len(cache.arrays()), 4)
        cache.clear()
        self.assertEqual(cache.num_ref_tokens, 0)
        with self.assertRaisesRegex(RuntimeError, "has not been populated"):
            cache.double_block_caches[0].get()

    def test_transformer_kv_extract_and_cached_paths_preserve_shapes(self) -> None:
        transformer = Flux2Transformer2DModel(
            tiny_transformer_config(guidance_embeds=True)
        )
        x = mx.random.normal((1, 4, 8), dtype=mx.float32)
        x_ref = mx.random.normal((1, 2, 8), dtype=mx.float32)
        ctx = mx.random.normal((1, 3, 12), dtype=mx.float32)
        x_ids = mx.zeros((1, 4, 2), dtype=mx.int32)
        x_ref_ids = mx.zeros((1, 2, 2), dtype=mx.int32)
        ctx_ids = mx.zeros((1, 3, 2), dtype=mx.int32)

        extracted, kv_cache = transformer.forward_kv_extract(
            x=x,
            x_ids=x_ids,
            x_ref=x_ref,
            x_ref_ids=x_ref_ids,
            timesteps=mx.array([0.5], dtype=mx.float32),
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=mx.array([1.0], dtype=mx.float32),
        )
        cached = transformer.forward_kv_cached(
            x=x,
            x_ids=x_ids,
            timesteps=mx.array([0.25], dtype=mx.float32),
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=mx.array([1.0], dtype=mx.float32),
            kv_cache=kv_cache,
        )

        self.assertEqual(extracted.shape, (1, 4, 8))
        self.assertEqual(cached.shape, (1, 4, 8))
        self.assertEqual(kv_cache.num_ref_tokens, 2)
        self.assertEqual(len(kv_cache.double_block_caches), 1)
        self.assertEqual(len(kv_cache.single_block_caches), 1)
        double_k, double_v = kv_cache.double_block_caches[0].get()
        single_k, single_v = kv_cache.single_block_caches[0].get()
        self.assertEqual(double_k.shape[2], 2)
        self.assertEqual(double_v.shape[2], 2)
        self.assertEqual(single_k.shape[2], 2)
        self.assertEqual(single_v.shape[2], 2)


if __name__ == "__main__":
    unittest.main()
