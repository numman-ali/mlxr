from __future__ import annotations

import unittest
from unittest.mock import patch

from mlxr.families.ltx.family_options import (
    conditioning_attention_strength_from_extensions,
    control_variant_from_extensions,
    distilled_guidance_mode_from_extensions,
    effective_seed,
    family_extensions,
    hq_stage_1_distilled_lora_strength_from_extensions,
    hq_stage_2_distilled_lora_strength_from_extensions,
    workflow_variant_from_extensions,
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

    def test_workflow_variant_defaults_when_unspecified(self) -> None:
        self.assertEqual(
            workflow_variant_from_extensions(
                {},
                supported_variants=("distilled_two_stage",),
                default="distilled_two_stage",
            ),
            "distilled_two_stage",
        )

    def test_workflow_variant_rejects_unsupported_value(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "workflow variant 'two_stage' is not supported",
        ):
            workflow_variant_from_extensions(
                {"workflow_variant": "two_stage"},
                supported_variants=("distilled_two_stage",),
                default="distilled_two_stage",
            )

    def test_workflow_variant_rejects_invalid_value(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "workflow_variant must be one of",
        ):
            workflow_variant_from_extensions(
                {"workflow_variant": "bad_variant"},
                supported_variants=("distilled_two_stage",),
                default="distilled_two_stage",
            )

    def test_control_variant_parses_supported_value(self) -> None:
        self.assertEqual(
            control_variant_from_extensions({"control_variant": "ic_lora"}),
            "ic_lora",
        )
        self.assertEqual(
            control_variant_from_extensions(
                {"control_variant": "motion_track_control"}
            ),
            "motion_track_control",
        )

    def test_control_variant_rejects_invalid_value(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "control_variant must be 'ic_lora'",
        ):
            control_variant_from_extensions({"control_variant": "depth"})

    def test_conditioning_attention_strength_validates_range(self) -> None:
        self.assertEqual(
            conditioning_attention_strength_from_extensions(
                {"conditioning_attention_strength": 0.5}
            ),
            0.5,
        )
        with self.assertRaisesRegex(
            ValueError,
            "conditioning_attention_strength must be between 0.0 and 1.0",
        ):
            conditioning_attention_strength_from_extensions(
                {"conditioning_attention_strength": 1.5}
            )

    def test_family_extensions_rejects_non_mapping(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "LTX extensions must be an object",
        ):
            family_extensions("not-a-mapping")

    def test_distilled_guidance_mode_rejects_non_string(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "distilled_guidance_mode must be a string",
        ):
            distilled_guidance_mode_from_extensions({"distilled_guidance_mode": 1})

    def test_hq_distilled_lora_strength_defaults_and_validates(self) -> None:
        self.assertEqual(hq_stage_1_distilled_lora_strength_from_extensions({}), 0.25)
        self.assertEqual(hq_stage_2_distilled_lora_strength_from_extensions({}), 0.5)
        self.assertEqual(
            hq_stage_1_distilled_lora_strength_from_extensions(
                {"hq_distilled_lora_strength_stage_1": 0.4}
            ),
            0.4,
        )
        with self.assertRaisesRegex(
            ValueError,
            "hq_distilled_lora_strength_stage_2 must be >= 0.0",
        ):
            hq_stage_2_distilled_lora_strength_from_extensions(
                {"hq_distilled_lora_strength_stage_2": -0.1}
            )


if __name__ == "__main__":
    unittest.main()
