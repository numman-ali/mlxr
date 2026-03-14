from __future__ import annotations

import unittest

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobOutputPolicy,
    ModelRecord,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowReference,
)
from mlxr.core.workflows import WorkflowPlanningContext
from mlxr.families.z_image.family_options import (
    cfg_normalization_from_extensions,
    cfg_truncation_from_extensions,
    family_extensions,
    max_sequence_length_from_extensions,
)
from mlxr.families.z_image.workflows import ZImageWorkflowStrategy


def _context(*, tasks: list[str] | None = None) -> WorkflowPlanningContext:
    capability = CapabilityDescriptor(
        model_id="z-image-turbo-local",
        artifact_digest="sha256:test",
        family="z_image",
        family_variant="z-image-turbo",
        scheduler_class="image_diffusion",
        tasks=tasks or ["image.generate"],
        profiles_by_task={"image.generate": ["default"]},
    )
    model = ModelRecord(
        model_id="z-image-turbo-local",
        family="z_image",
        capability=capability,
    )
    return WorkflowPlanningContext(model=model, capability=capability)


class ZImageWorkflowAndOptionsTests(unittest.TestCase):
    def test_family_extensions_rejects_non_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an object"):
            family_extensions("bad")

    def test_cfg_normalization_supports_bool_and_rejects_negative(self) -> None:
        self.assertEqual(
            cfg_normalization_from_extensions({"cfg_normalization": True}),
            1.0,
        )
        self.assertEqual(
            cfg_normalization_from_extensions({"cfg_normalization": False}),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "must be >= 0.0"):
            cfg_normalization_from_extensions({"cfg_normalization": -1.0})

    def test_cfg_truncation_and_max_sequence_length_validate_types(self) -> None:
        self.assertEqual(
            cfg_truncation_from_extensions({"cfg_truncation": 0.5}),
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "must be a number"):
            cfg_truncation_from_extensions({"cfg_truncation": True})
        self.assertEqual(
            max_sequence_length_from_extensions({"max_sequence_length": 1024}),
            1024,
        )
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            max_sequence_length_from_extensions({"max_sequence_length": 0})

    def test_plan_rejects_conditioning_references(self) -> None:
        strategy = ZImageWorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="z-image-turbo-local",
            prompt="bookstore",
            references=[WorkflowReference(kind="image")],
        )

        with self.assertRaisesRegex(ValueError, "does not support conditioning"):
            strategy.plan(context, intent)

    def test_plan_rejects_unsupported_task(self) -> None:
        strategy = ZImageWorkflowStrategy()
        context = _context(tasks=["image.generate"])
        intent = WorkflowIntent(
            model_id="z-image-turbo-local",
            prompt="bookstore",
            task="image.edit",
        )

        with self.assertRaisesRegex(ValueError, "does not support workflow task"):
            strategy.plan(context, intent)

    def test_to_job_request_rejects_plan_task_not_in_capability(self) -> None:
        strategy = ZImageWorkflowStrategy()
        context = _context(tasks=["image.generate"])
        intent = WorkflowIntent(
            model_id="z-image-turbo-local",
            prompt="bookstore",
            output=JobOutputPolicy(artifact_format="png"),
        )
        plan = WorkflowPlan(
            model_id="z-image-turbo-local",
            family="z_image",
            selected_task="image.edit",
            resolved_prompt="bookstore",
        )

        with self.assertRaisesRegex(ValueError, "does not support task"):
            strategy.to_job_request(context, intent, plan)

    def test_to_job_request_omits_empty_family_extensions(self) -> None:
        strategy = ZImageWorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="z-image-turbo-local",
            prompt="bookstore",
            output=JobOutputPolicy(artifact_format="png"),
        )
        plan = WorkflowPlan(
            model_id="z-image-turbo-local",
            family="z_image",
            selected_task="image.generate",
            resolved_prompt="bookstore",
        )

        request = strategy.to_job_request(context, intent, plan)

        self.assertNotIn("z_image", request.extensions)
        self.assertEqual(
            request.extensions["workflow"]["selected_task"], "image.generate"
        )

    def test_presentation_exposes_single_generate_subworkflow(self) -> None:
        strategy = ZImageWorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="z-image-turbo-local",
            prompt="bookstore",
            output=JobOutputPolicy(artifact_format="png"),
        )

        plan = strategy.plan(context, intent)
        readiness = strategy.readiness(context, intent, plan)
        presentation = strategy.presentation(context, intent, plan, readiness)

        self.assertEqual(presentation.primary_mode, "image")
        self.assertEqual(presentation.selected_task, "image.generate")
        self.assertEqual(len(presentation.subworkflows), 1)
        self.assertEqual(presentation.subworkflows[0].label, "Generate from text")
        self.assertTrue(presentation.subworkflows[0].default)
        self.assertEqual(presentation.reference_slots, [])
        self.assertEqual(
            [option.value for option in presentation.controls.aspect_presets],
            ["square", "landscape", "portrait", "story"],
        )
        self.assertEqual(presentation.controls.duration_presets, [])
        self.assertEqual(presentation.controls.variation_counts, [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
