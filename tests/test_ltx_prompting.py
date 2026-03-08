from __future__ import annotations

import unittest

from mlxr.families.ltx.prompting import (
    PromptShapingOptions,
    shape_text_first_prompt_bundle,
)


class LTXPromptingTests(unittest.TestCase):
    def test_prompt_bundle_derives_negative_prompt_for_natural_audio(self) -> None:
        bundle = shape_text_first_prompt_bundle(
            "A golden retriever runs beside its owner in a park.",
            options=PromptShapingOptions(
                audio_prompt="happy barking and park ambience",
                natural_audio=True,
                no_music=True,
                style_family="naturalistic",
                duration_seconds=10.0,
                orientation="landscape",
            ),
        )

        self.assertIn("Audio details: happy barking and park ambience", bundle.prompt)
        self.assertIsInstance(bundle.negative_prompt, str)
        assert isinstance(bundle.negative_prompt, str)
        self.assertIn("background music", bundle.negative_prompt)
        self.assertIn("piano melody", bundle.negative_prompt)
        self.assertIn("synthetic sound design", bundle.negative_prompt)
        self.assertIn("soft piano bed", bundle.negative_prompt)

    def test_prompt_bundle_uses_style_specific_negative_terms(self) -> None:
        documentary_bundle = shape_text_first_prompt_bundle(
            "A heron stands in a marsh at dawn.",
            options=PromptShapingOptions(
                audio_prompt="water lapping, reed rustle, distant frog, splash",
                natural_audio=True,
                no_music=True,
                style_family="documentary",
            ),
        )
        vintage_bundle = shape_text_first_prompt_bundle(
            "Inside a 1970s laundromat, a young man reads a paperback.",
            options=PromptShapingOptions(
                audio_prompt="washer churn, dryer buzzer, fluorescent hum",
                natural_audio=True,
                no_music=True,
                style_family="vintage",
            ),
        )

        assert isinstance(documentary_bundle.negative_prompt, str)
        assert isinstance(vintage_bundle.negative_prompt, str)
        self.assertIn("nature documentary score", documentary_bundle.negative_prompt)
        self.assertIn("gentle piano underscore", documentary_bundle.negative_prompt)
        self.assertIn("nostalgic music", vintage_bundle.negative_prompt)
        self.assertIn("retro soundtrack", vintage_bundle.negative_prompt)

    def test_prompt_bundle_skips_negative_prompt_for_plain_visual_scene(self) -> None:
        bundle = shape_text_first_prompt_bundle(
            "An astronaut walks across the moon.",
            options=PromptShapingOptions(duration_seconds=10.0),
        )

        self.assertIsNone(bundle.negative_prompt)


if __name__ == "__main__":
    unittest.main()
