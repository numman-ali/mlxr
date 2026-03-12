from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from mlxr.core.mlx_models import (
    Qwen2VLImageProcessorConfig,
    expand_image_placeholders,
    preprocess_qwen2_vl_images,
    smart_resize,
)
from PIL import Image


class Qwen2VLProcessingTests(unittest.TestCase):
    def test_config_loads_from_preprocessor_json(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "preprocessor_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "min_pixels": 3136,
                        "max_pixels": 12845056,
                        "patch_size": 14,
                        "temporal_patch_size": 2,
                        "merge_size": 2,
                        "rescale_factor": 1 / 255,
                        "image_mean": [0.1, 0.2, 0.3],
                        "image_std": [0.4, 0.5, 0.6],
                    }
                ),
                encoding="utf-8",
            )

            config = Qwen2VLImageProcessorConfig.from_path(config_path)

        self.assertEqual(config.min_pixels, 3136)
        self.assertEqual(config.max_pixels, 12845056)
        self.assertEqual(config.patch_size, 14)
        self.assertEqual(config.temporal_patch_size, 2)
        self.assertEqual(config.merge_size, 2)
        self.assertEqual(config.image_mean, (0.1, 0.2, 0.3))

    def test_smart_resize_preserves_factor_and_limits(self) -> None:
        height, width = smart_resize(
            300,
            500,
            factor=28,
            min_pixels=3136,
            max_pixels=12845056,
        )

        self.assertEqual(height % 28, 0)
        self.assertEqual(width % 28, 0)
        self.assertGreaterEqual(height * width, 3136)
        self.assertLessEqual(height * width, 12845056)

    def test_preprocess_qwen2_vl_images_returns_flattened_patches(self) -> None:
        image = Image.fromarray(np.full((64, 96, 3), 128, dtype=np.uint8), mode="RGB")
        config = Qwen2VLImageProcessorConfig(
            min_pixels=3136,
            max_pixels=12845056,
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            rescale_factor=1 / 255,
            image_mean=(0.1, 0.2, 0.3),
            image_std=(0.4, 0.5, 0.6),
        )

        processed = preprocess_qwen2_vl_images((image,), config=config)

        self.assertEqual(tuple(processed.image_grid_thw.shape), (1, 3))
        grid_t, grid_h, grid_w = processed.image_grid_thw[0].tolist()
        self.assertEqual(grid_t, 1)
        self.assertEqual(
            tuple(processed.pixel_values.shape),
            (
                grid_t * grid_h * grid_w,
                3 * config.temporal_patch_size * config.patch_size * config.patch_size,
            ),
        )

    def test_expand_image_placeholders_matches_grid_token_count(self) -> None:
        image_grid_thw = np.asarray([[1, 24, 16]], dtype=np.int32)

        expanded = expand_image_placeholders(
            ["Picture 1: <|image_pad|> edit this scene"],
            image_grid_thw=image_grid_thw,
            merge_size=2,
        )

        expected_tokens = int(np.prod(image_grid_thw[0])) // 4
        self.assertEqual(expanded[0].count("<|image_pad|>"), expected_tokens)


if __name__ == "__main__":
    unittest.main()
