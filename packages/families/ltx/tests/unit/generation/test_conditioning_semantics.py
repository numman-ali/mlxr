from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
from mlxr.families.ltx._generation_backend.primitives import (
    LatentState,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    apply_conditioning,
)
from mlxr.families.ltx._generation_backend.runtime_helpers import (
    _prepare_conditionings,
)
from mlxr.families.ltx.generation import ConditioningInput
from PIL import Image


class ConditioningSemanticsTests(unittest.TestCase):
    def test_latent_conditioning_replaces_target_frame_and_updates_mask(self) -> None:
        state = LatentState(
            latent=mx.zeros((1, 2, 3, 1, 1), dtype=mx.float32),
            clean_latent=mx.zeros((1, 2, 3, 1, 1), dtype=mx.float32),
            denoise_mask=mx.ones((1, 1, 3, 1, 1), dtype=mx.float32),
        )
        cond = VideoConditionByLatentIndex(
            latent=mx.ones((1, 2, 1, 1, 1), dtype=mx.float32),
            frame_idx=0,
            strength=1.0,
        )

        updated = apply_conditioning(state, [cond])

        self.assertAlmostEqual(float(updated.latent[0, 0, 0, 0, 0].item()), 1.0)
        self.assertAlmostEqual(float(updated.clean_latent[0, 0, 0, 0, 0].item()), 1.0)
        self.assertAlmostEqual(float(updated.denoise_mask[0, 0, 0, 0, 0].item()), 0.0)
        self.assertAlmostEqual(float(updated.latent[0, 0, 1, 0, 0].item()), 0.0)

    def test_keyframe_conditioning_guides_target_frame_without_replacement(
        self,
    ) -> None:
        state = LatentState(
            latent=mx.zeros((1, 2, 3, 1, 1), dtype=mx.float32),
            clean_latent=mx.zeros((1, 2, 3, 1, 1), dtype=mx.float32),
            denoise_mask=mx.ones((1, 1, 3, 1, 1), dtype=mx.float32),
        )
        cond = VideoConditionByKeyframeIndex(
            latent=mx.ones((1, 2, 1, 1, 1), dtype=mx.float32),
            frame_idx=2,
            strength=0.5,
        )

        updated = apply_conditioning(state, [cond])

        self.assertAlmostEqual(float(updated.latent[0, 0, 2, 0, 0].item()), 0.5)
        self.assertAlmostEqual(float(updated.clean_latent[0, 0, 2, 0, 0].item()), 0.0)
        self.assertAlmostEqual(float(updated.denoise_mask[0, 0, 2, 0, 0].item()), 1.0)
        self.assertAlmostEqual(float(updated.latent[0, 0, 0, 0, 0].item()), 0.0)

    def test_prepare_conditionings_uses_keyframe_guidance_for_nonzero_frames(
        self,
    ) -> None:
        class _FakeEncoder:
            def __call__(self, image: mx.array) -> mx.array:
                return mx.ones((1, 4, 1, 2, 2), dtype=image.dtype)

        class _FakeImports:
            @staticmethod
            def load_image(
                path: str | Path,
                *,
                height: int | None = None,
                width: int | None = None,
                dtype: mx.Dtype = mx.float32,
            ) -> mx.array:
                return mx.ones((1, int(height or 4), int(width or 4), 3), dtype=dtype)

            @staticmethod
            def prepare_image_for_encoding(
                image: mx.array,
                target_height: int,
                target_width: int,
                *,
                dtype: mx.Dtype = mx.float32,
            ) -> mx.array:
                del target_height, target_width
                return image.astype(dtype)

            latent_condition_class = VideoConditionByLatentIndex
            keyframe_condition_class = VideoConditionByKeyframeIndex

        class _FakeHost:
            _vae_encoder = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "conditioning.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            with patch(
                "mlxr.families.ltx._generation_backend.runtime_helpers._ensure_vae_encoder",
                return_value=_FakeEncoder(),
            ):
                plan = _prepare_conditionings(
                    _FakeHost(),
                    imports=_FakeImports(),
                    conditioning_inputs=(
                        ConditioningInput(
                            handle_id="img0",
                            payload_path=image_path,
                            frame_index=0,
                            strength=1.0,
                        ),
                        ConditioningInput(
                            handle_id="img8",
                            payload_path=image_path,
                            frame_index=8,
                            strength=0.75,
                        ),
                    ),
                    num_frames=17,
                    latent_frames=3,
                    padded_shape=type(
                        "_Shape",
                        (),
                        {
                            "internal_width": 64,
                            "internal_height": 64,
                        },
                    )(),
                    model_dtype=mx.float32,
                )

        self.assertIsInstance(plan.stage1[0], VideoConditionByLatentIndex)
        self.assertIsInstance(plan.stage2[0], VideoConditionByLatentIndex)
        self.assertIsInstance(plan.stage1[1], VideoConditionByKeyframeIndex)
        self.assertIsInstance(plan.stage2[1], VideoConditionByKeyframeIndex)


if __name__ == "__main__":
    unittest.main()
