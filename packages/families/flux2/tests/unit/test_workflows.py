from __future__ import annotations

import unittest

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobOutputPolicy,
    ModelRecord,
    WorkflowIntent,
    WorkflowReference,
)
from mlxr.core.workflows import WorkflowPlanningContext
from mlxr.families.flux2.family_options import (
    family_extensions,
    prompt_upsampling_mode_from_extensions,
    quantize_bits_from_extensions,
)
from mlxr.families.flux2.workflows import Flux2WorkflowStrategy


def _context() -> WorkflowPlanningContext:
    capability = CapabilityDescriptor(
        model_id="flux2-klein-9b-local",
        artifact_digest="sha256:test",
        family="flux2",
        family_variant="flux.2-klein-9b",
        scheduler_class="image_diffusion",
        tasks=["image.generate", "image.edit"],
        profiles_by_task={"image.generate": ["default"], "image.edit": ["default"]},
    )
    model = ModelRecord(
        model_id="flux2-klein-9b-local",
        family="flux2",
        capability=capability,
    )
    return WorkflowPlanningContext(model=model, capability=capability)


class Flux2WorkflowTests(unittest.TestCase):
    def test_family_extensions_validate_prompt_upsampling_mode(self) -> None:
        self.assertEqual(family_extensions(None), {})
        self.assertEqual(family_extensions({"quantize_bits": 8}), {"quantize_bits": 8})
        self.assertEqual(
            prompt_upsampling_mode_from_extensions(
                {"prompt_upsampling_mode": "openrouter"}
            ),
            "openrouter",
        )
        self.assertEqual(
            quantize_bits_from_extensions({"quantize_bits": 6}),
            6,
        )
        self.assertIsNone(quantize_bits_from_extensions({"quantize_bits": 0}))
        with self.assertRaisesRegex(ValueError, "must be an object"):
            family_extensions("invalid")
        with self.assertRaisesRegex(ValueError, "must be a string"):
            prompt_upsampling_mode_from_extensions({"prompt_upsampling_mode": 7})
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            quantize_bits_from_extensions({"quantize_bits": "8"})
        with self.assertRaisesRegex(ValueError, "must be one of 2, 3, 4, 6, or 8"):
            quantize_bits_from_extensions({"quantize_bits": 5})
        with self.assertRaisesRegex(
            ValueError, "must be 'none', 'local', or 'openrouter'"
        ):
            prompt_upsampling_mode_from_extensions(
                {"prompt_upsampling_mode": "dashscope"}
            )

    def test_plan_defaults_to_generate_without_references(self) -> None:
        strategy = Flux2WorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="flux2-klein-9b-local",
            prompt="cinematic portrait",
            output=JobOutputPolicy(artifact_format="png"),
        )

        plan = strategy.plan(context, intent)
        self.assertEqual(plan.selected_task, "image.generate")

    def test_plan_switches_to_edit_with_image_references(self) -> None:
        strategy = Flux2WorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="flux2-klein-9b-local",
            prompt="keep the same person and move them to a rainy alley",
            references=[WorkflowReference(kind="image", input_handle="inp_ref")],
            output=JobOutputPolicy(artifact_format="png"),
        )

        plan = strategy.plan(context, intent)
        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(plan.selected_task, "image.edit")
        self.assertEqual(request.inputs["images"][0]["input_handle"], "inp_ref")

    def test_readiness_requires_image_reference_for_edit(self) -> None:
        strategy = Flux2WorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="flux2-klein-9b-local",
            prompt="keep the subject and change the location",
            task="image.edit",
            output=JobOutputPolicy(artifact_format="png"),
        )

        plan = strategy.plan(context, intent)
        readiness = strategy.readiness(context, intent, plan)

        self.assertFalse(readiness.ready)
        self.assertEqual(
            readiness.reference_requirements[0].description,
            "Choose at least one image to edit.",
        )
        self.assertIn(
            "Choose at least one image to edit.",
            readiness.blocking_issues,
        )

    def test_readiness_blocks_unsupported_output_format(self) -> None:
        strategy = Flux2WorkflowStrategy()
        context = _context()
        context.capability.artifacts_out = ["png"]
        intent = WorkflowIntent(
            model_id="flux2-klein-9b-local",
            prompt="cinematic portrait",
            output=JobOutputPolicy(artifact_format="jpg"),
        )

        plan = strategy.plan(context, intent)
        readiness = strategy.readiness(context, intent, plan)

        self.assertFalse(readiness.ready)
        self.assertIn(
            "Output format 'jpg' is not supported for this model.",
            readiness.blocking_issues,
        )

    def test_to_job_request_preserves_flux_extensions(self) -> None:
        strategy = Flux2WorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="flux2-klein-9b-local",
            prompt="edit",
            references=[WorkflowReference(kind="image", input_handle="inp_ref")],
            output=JobOutputPolicy(artifact_format="png"),
            extensions={"flux2": {"prompt_upsampling_mode": "openrouter"}},
        )

        plan = strategy.plan(context, intent)
        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(
            request.extensions["flux2"]["prompt_upsampling_mode"],
            "openrouter",
        )

    def test_to_job_request_preserves_lora_references(self) -> None:
        strategy = Flux2WorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="flux2-klein-9b-local",
            prompt="Turn this into a glossy magazine portrait",
            references=[
                WorkflowReference(kind="image", input_handle="inp_ref"),
                WorkflowReference(
                    kind="lora",
                    input_handle="inp_lora",
                    metadata={"strength": 0.4},
                ),
            ],
            output=JobOutputPolicy(artifact_format="png"),
        )

        plan = strategy.plan(context, intent)
        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(request.inputs["loras"][0]["input_handle"], "inp_lora")
        self.assertEqual(request.inputs["loras"][0]["strength"], 0.4)

    def test_explicit_task_must_be_supported(self) -> None:
        strategy = Flux2WorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="flux2-klein-9b-local",
            prompt="portrait",
            task="video.generate",
        )

        with self.assertRaisesRegex(ValueError, "does not support workflow task"):
            strategy.plan(context, intent)

    def test_generate_presentation_exposes_two_image_subworkflows(self) -> None:
        strategy = Flux2WorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="flux2-klein-9b-local",
            prompt="cinematic portrait",
            output=JobOutputPolicy(artifact_format="png"),
        )

        plan = strategy.plan(context, intent)
        readiness = strategy.readiness(context, intent, plan)
        presentation = strategy.presentation(context, intent, plan, readiness)

        self.assertEqual(presentation.primary_mode, "image")
        self.assertEqual(
            [item.task for item in presentation.subworkflows],
            ["image.generate", "image.edit"],
        )
        self.assertEqual(
            [item.task for item in presentation.subworkflows if item.default],
            ["image.generate"],
        )
        self.assertEqual(presentation.reference_slots, [])

    def test_edit_presentation_requires_source_image_slot(self) -> None:
        strategy = Flux2WorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="flux2-klein-9b-local",
            prompt="keep the subject and change the location",
            task="image.edit",
            output=JobOutputPolicy(artifact_format="png"),
        )

        plan = strategy.plan(context, intent)
        readiness = strategy.readiness(context, intent, plan)
        presentation = strategy.presentation(context, intent, plan, readiness)

        self.assertEqual(
            [item.task for item in presentation.subworkflows if item.default],
            ["image.edit"],
        )
        self.assertEqual(len(presentation.reference_slots), 1)
        self.assertEqual(presentation.reference_slots[0].slot_id, "source-image")
        self.assertTrue(presentation.reference_slots[0].required)
        self.assertEqual(presentation.reference_slots[0].minimum_count, 1)
        self.assertTrue(presentation.reference_slots[0].allows_multiple)


if __name__ == "__main__":
    unittest.main()
