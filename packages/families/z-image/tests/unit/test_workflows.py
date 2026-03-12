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


if __name__ == "__main__":
    unittest.main()
