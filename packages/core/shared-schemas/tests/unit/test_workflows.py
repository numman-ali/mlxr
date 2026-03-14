from __future__ import annotations

import unittest

from mlxr.core.schemas import (
    CapabilityDescriptor,
    PolicyDescriptor,
    WorkflowPlan,
    WorkflowPlanPresentation,
    WorkflowPlanResult,
    WorkflowPresentationControlOption,
    WorkflowPresentationControls,
    WorkflowPresentationReferenceSlot,
    WorkflowPresentationSubworkflow,
)
from pydantic import ValidationError


class WorkflowPresentationSchemaTests(unittest.TestCase):
    def test_presentation_round_trips_through_model_dump(self) -> None:
        presentation = WorkflowPlanPresentation(
            primary_mode="video",
            selected_task="video.condition.image",
            subworkflows=[
                WorkflowPresentationSubworkflow(
                    task="video.generate",
                    label="Generate from text",
                    mode="video",
                    default=False,
                ),
                WorkflowPresentationSubworkflow(
                    task="video.condition.image",
                    label="Animate an image",
                    mode="video",
                    default=True,
                ),
            ],
            reference_slots=[
                WorkflowPresentationReferenceSlot(
                    slot_id="start-frame",
                    label="Start Frame",
                    kind="image",
                    description="Choose a still image to animate.",
                    required=True,
                    minimum_count=1,
                    maximum_count=1,
                    accepted_roles=["reference"],
                    allows_multiple=False,
                )
            ],
            controls=WorkflowPresentationControls(
                quality_presets=[
                    WorkflowPresentationControlOption(
                        value="draft",
                        label="Draft",
                    )
                ],
                aspect_presets=[
                    WorkflowPresentationControlOption(
                        value="landscape",
                        label="Landscape",
                        default=True,
                    )
                ],
                duration_presets=[
                    WorkflowPresentationControlOption(
                        value="medium",
                        label="8s",
                        default=True,
                    )
                ],
                variation_counts=[1, 2, 4],
            ),
        )

        restored = WorkflowPlanPresentation.model_validate(presentation.model_dump())

        self.assertEqual(restored, presentation)

    def test_presentation_models_forbid_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            WorkflowPresentationReferenceSlot.model_validate(
                {
                    "slot_id": "source-image",
                    "label": "Source image",
                    "kind": "image",
                    "required": True,
                    "minimum_count": 1,
                    "extra_field": "nope",
                }
            )

    def test_plan_result_preserves_explicit_presentation(self) -> None:
        presentation = WorkflowPlanPresentation(
            primary_mode="image",
            selected_task="image.edit",
            subworkflows=[
                WorkflowPresentationSubworkflow(
                    task="image.edit",
                    label="Edit an existing image",
                    mode="image",
                    default=True,
                )
            ],
        )

        result = WorkflowPlanResult(
            capability=CapabilityDescriptor(
                model_id="flux2-klein-9b-local",
                artifact_digest="sha256:test",
                family="flux2",
                tasks=["image.generate", "image.edit"],
                modalities_in=["text", "image"],
                modalities_out=["image"],
                scheduler_class="image_diffusion",
                policy=PolicyDescriptor(access_state="public"),
                metadata={},
            ),
            plan=WorkflowPlan(
                model_id="flux2-klein-9b-local",
                family="flux2",
                selected_task="image.edit",
                resolved_prompt="keep the subject but change the location",
            ),
            presentation=presentation,
        )

        self.assertEqual(result.presentation, presentation)


if __name__ == "__main__":
    unittest.main()
