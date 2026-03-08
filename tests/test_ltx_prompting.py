from __future__ import annotations

import unittest

from ltx.prompting import (
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

    def test_prompt_bundle_skips_negative_prompt_for_plain_visual_scene(self) -> None:
        bundle = shape_text_first_prompt_bundle(
            "An astronaut walks across the moon.",
            options=PromptShapingOptions(duration_seconds=10.0),
        )

        self.assertIsNone(bundle.negative_prompt)


if __name__ == "__main__":
    unittest.main()
