from __future__ import annotations

import unittest

from mlxr.core.schemas import WorkflowReferenceRequirement
from mlxr.core.workflows.presentation import (
    image_controls,
    slots_from_requirements,
    subworkflow,
    video_controls,
)


class WorkflowPresentationHelpersTests(unittest.TestCase):
    def test_image_controls_expose_expected_defaults(self) -> None:
        controls = image_controls()

        self.assertEqual(
            [option.value for option in controls.quality_presets],
            ["draft", "standard", "cinema"],
        )
        self.assertEqual(
            [option.value for option in controls.aspect_presets],
            ["square", "landscape", "portrait", "story"],
        )
        self.assertEqual(
            [option.value for option in controls.aspect_presets if option.default],
            ["landscape"],
        )
        self.assertEqual(controls.variation_counts, [1, 2, 4])

    def test_video_controls_expose_expected_defaults(self) -> None:
        controls = video_controls()

        self.assertEqual(
            [option.value for option in controls.aspect_presets if option.default],
            ["landscape"],
        )
        self.assertEqual(
            [option.label for option in controls.duration_presets],
            ["4s", "8s", "12s"],
        )
        self.assertEqual(
            [option.value for option in controls.duration_presets if option.default],
            ["medium"],
        )

    def test_subworkflow_marks_only_selected_task_as_default(self) -> None:
        generate = subworkflow(
            task="image.generate",
            label="Generate from text",
            mode="image",
            selected_task="image.edit",
        )
        edit = subworkflow(
            task="image.edit",
            label="Edit an existing image",
            mode="image",
            selected_task="image.edit",
        )

        self.assertFalse(generate.default)
        self.assertTrue(edit.default)

    def test_slots_from_requirements_maps_counts_and_roles(self) -> None:
        slots = slots_from_requirements(
            [
                WorkflowReferenceRequirement(
                    kind="image",
                    minimum_count=1,
                    maximum_count=1,
                    accepted_roles=["start_frame"],
                    description="Choose one source image.",
                ),
                WorkflowReferenceRequirement(
                    kind="video",
                    minimum_count=0,
                    maximum_count=2,
                    accepted_roles=["guide_video"],
                    description="Optional guide videos.",
                ),
            ]
        )

        self.assertEqual(len(slots), 2)
        self.assertTrue(slots[0].required)
        self.assertFalse(slots[0].allows_multiple)
        self.assertEqual(slots[0].accepted_roles, ["start_frame"])
        self.assertFalse(slots[1].required)
        self.assertTrue(slots[1].allows_multiple)
        self.assertEqual(slots[1].maximum_count, 2)


if __name__ == "__main__":
    unittest.main()
