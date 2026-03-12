from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from mlxr.families.flux2._generation_backend.images import (
    _cap_pixels,
    _center_crop_to_multiple,
    _conditioning_reference_limit,
    _reference_tensor,
    _resolved_dimensions,
    _validate_reference_image,
)
from PIL import Image


class Flux2ImageHelperTests(unittest.TestCase):
    def test_conditioning_reference_limit_shrinks_for_multi_ref(self) -> None:
        self.assertEqual(_conditioning_reference_limit(1), 2_024**2)
        self.assertEqual(_conditioning_reference_limit(2), 1_024**2)

    def test_crop_cap_validate_and_reference_tensor_helpers(self) -> None:
        cropped = _center_crop_to_multiple(
            Image.new("RGB", (35, 19), color="white"), 16
        )
        capped = _cap_pixels(Image.new("RGB", (400, 200), color="white"), 10_000)
        tensor = _reference_tensor(
            Image.new("RGB", (79, 65), color="navy"), limit_pixels=4096
        )

        self.assertEqual(cropped.size, (32, 16))
        self.assertLessEqual(capped.size[0] * capped.size[1], 10_000)
        self.assertEqual(int(tensor.shape[-1]), 3)
        self.assertEqual(int(tensor.shape[0]) % 16, 0)
        self.assertEqual(int(tensor.shape[1]) % 16, 0)
        self.assertTrue(np.max(np.asarray(tensor)) <= 1.0)
        self.assertTrue(np.min(np.asarray(tensor)) >= -1.0)

    def test_validate_reference_image_rejects_small_and_extreme_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 64px"):
            _validate_reference_image(Image.new("RGB", (63, 64), color="white"))
        with self.assertRaisesRegex(ValueError, "within 8:1 aspect ratio"):
            _validate_reference_image(Image.new("RGB", (1024, 64), color="white"))

    def test_resolved_dimensions_default_to_first_reference_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "reference.png"
            Image.new("RGB", (771, 1027), color="white").save(path)

            width, height = _resolved_dimensions(
                width=None,
                height=None,
                image_paths=(path,),
            )

        self.assertEqual((width, height), (768, 1024))

    def test_resolved_dimensions_defaults_and_validation(self) -> None:
        self.assertEqual(
            _resolved_dimensions(width=None, height=None, image_paths=()),
            (1024, 1024),
        )
        self.assertEqual(
            _resolved_dimensions(width=256, height=512, image_paths=()),
            (256, 512),
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _resolved_dimensions(width=0, height=256, image_paths=())
        with self.assertRaisesRegex(ValueError, "must be multiples of 16"):
            _resolved_dimensions(width=255, height=256, image_paths=())


if __name__ == "__main__":
    unittest.main()
