from __future__ import annotations

import unittest
from unittest.mock import patch

from mlxr.families.ltx.family_options import (
    distilled_guidance_mode_from_extensions,
    effective_seed,
)


class LTXFamilyOptionsTests(unittest.TestCase):
    def test_effective_seed_preserves_explicit_seed(self) -> None:
        self.assertEqual(effective_seed(seed=17), 17)

    def test_effective_seed_generates_random_seed_when_unspecified(self) -> None:
        with patch("mlxr.families.ltx.family_options.secrets.randbelow") as randbelow:
            randbelow.return_value = 123456

            self.assertEqual(effective_seed(seed=None), 123456)
            randbelow.assert_called_once_with(2**31)

    def test_distilled_guidance_mode_defaults_to_positive_only(self) -> None:
        self.assertEqual(
            distilled_guidance_mode_from_extensions({}),
            "positive_only",
        )

    def test_distilled_guidance_mode_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "distilled_guidance_mode must be 'positive_only' or 'cfg'",
        ):
            distilled_guidance_mode_from_extensions({"distilled_guidance_mode": "apg"})


if __name__ == "__main__":
    unittest.main()
