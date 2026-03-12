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
from mlxr.families.qwen_image.family_options import (
    family_extensions,
    prompt_enhance_mode_from_extensions,
    quantize_bits_from_extensions,
    scheduler_preset_from_extensions,
)
from mlxr.families.qwen_image.workflows import QwenImageWorkflowStrategy


def _context(*, tasks: list[str] | None = None) -> WorkflowPlanningContext:
    capability = CapabilityDescriptor(
        model_id="qwen-image-local",
        artifact_digest="sha256:test",
        family="qwen_image",
        family_variant="qwen-image-2512",
        scheduler_class="image_diffusion",
        tasks=tasks or ["image.generate"],
        profiles_by_task={task: ["default"] for task in (tasks or ["image.generate"])},
    )
    model = ModelRecord(
        model_id="qwen-image-local",
        family="qwen_image",
        capability=capability,
    )
    return WorkflowPlanningContext(model=model, capability=capability)


class QwenImageWorkflowTests(unittest.TestCase):
    def test_family_extensions_validate_prompt_enhance_mode(self) -> None:
        self.assertEqual(family_extensions(None), {})
        self.assertEqual(family_extensions({"quantize_bits": 8}), {"quantize_bits": 8})
        self.assertEqual(
            prompt_enhance_mode_from_extensions({"prompt_enhance_mode": "dashscope"}),
            "dashscope",
        )
        self.assertEqual(quantize_bits_from_extensions({"quantize_bits": 6}), 6)
        self.assertIsNone(quantize_bits_from_extensions({"quantize_bits": 0}))
        self.assertEqual(
            scheduler_preset_from_extensions({"scheduler_preset": "lightning"}),
            "lightning",
        )
        self.assertEqual(
            scheduler_preset_from_extensions({"scheduler_preset": "turbo_wuli"}),
            "turbo_wuli",
        )
        with self.assertRaisesRegex(ValueError, "must be an object"):
            family_extensions("invalid")
        with self.assertRaisesRegex(ValueError, "must be a string"):
            prompt_enhance_mode_from_extensions({"prompt_enhance_mode": 1})
        with self.assertRaisesRegex(ValueError, "must be a string"):
            scheduler_preset_from_extensions({"scheduler_preset": 1})
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            quantize_bits_from_extensions({"quantize_bits": "8"})
        with self.assertRaisesRegex(ValueError, "must be one of 2, 3, 4, 6, or 8"):
            quantize_bits_from_extensions({"quantize_bits": 5})
        with self.assertRaisesRegex(ValueError, "must be 'none' or 'dashscope'"):
            prompt_enhance_mode_from_extensions({"prompt_enhance_mode": "local"})
        with self.assertRaisesRegex(
            ValueError, "must be 'default', 'lightning', or 'turbo_wuli'"
        ):
            scheduler_preset_from_extensions({"scheduler_preset": "turbo"})

    def test_plan_defaults_to_generate_without_references(self) -> None:
        strategy = QwenImageWorkflowStrategy()
        context = _context(tasks=["image.generate"])
        intent = WorkflowIntent(
            model_id="qwen-image-local",
            prompt="poster with bilingual typography",
            output=JobOutputPolicy(artifact_format="png"),
        )

        plan = strategy.plan(context, intent)
        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(plan.selected_task, "image.generate")
        self.assertNotIn("images", request.inputs)

    def test_plan_switches_to_image_edit_with_image_references(self) -> None:
        strategy = QwenImageWorkflowStrategy()
        context = _context(tasks=["image.edit"])
        intent = WorkflowIntent(
            model_id="qwen-image-local",
            prompt="preserve identity and place the subject on a rainy neon street",
            references=[
                WorkflowReference(
                    kind="image",
                    input_handle="inp_ref",
                    role="reference",
                )
            ],
            output=JobOutputPolicy(artifact_format="png"),
        )

        plan = strategy.plan(context, intent)
        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(plan.selected_task, "image.edit")
        self.assertEqual(request.inputs["images"][0]["input_handle"], "inp_ref")

    def test_plan_rejects_non_image_references(self) -> None:
        strategy = QwenImageWorkflowStrategy()
        context = _context(tasks=["image.edit"])
        intent = WorkflowIntent(
            model_id="qwen-image-local",
            prompt="edit",
            references=[WorkflowReference(kind="video")],
        )

        with self.assertRaisesRegex(
            ValueError, "only supports image and lora references"
        ):
            strategy.plan(context, intent)

    def test_to_job_request_preserves_family_extensions_for_edit(self) -> None:
        strategy = QwenImageWorkflowStrategy()
        context = _context(tasks=["image.edit"])
        intent = WorkflowIntent(
            model_id="qwen-image-local",
            prompt="edit",
            references=[WorkflowReference(kind="image", input_handle="inp_ref")],
            output=JobOutputPolicy(artifact_format="png"),
            extensions={"qwen_image": {"prompt_enhance_mode": "dashscope"}},
        )

        plan = strategy.plan(context, intent)
        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(
            request.extensions["qwen_image"]["prompt_enhance_mode"],
            "dashscope",
        )

    def test_to_job_request_preserves_negative_prompt_and_loras(self) -> None:
        strategy = QwenImageWorkflowStrategy()
        context = _context(tasks=["image.edit"])
        intent = WorkflowIntent(
            model_id="qwen-image-local",
            prompt="give the subject a red scarf",
            negative_prompt="warped hands",
            references=[
                WorkflowReference(kind="image", input_handle="inp_ref"),
                WorkflowReference(
                    kind="lora",
                    input_handle="inp_lora",
                    metadata={"strength": 0.25},
                ),
            ],
            output=JobOutputPolicy(artifact_format="png"),
        )

        plan = strategy.plan(context, intent)
        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(request.inputs["negative_prompt"], "warped hands")
        self.assertEqual(request.inputs["loras"][0]["input_handle"], "inp_lora")
        self.assertEqual(request.inputs["loras"][0]["strength"], 0.25)

    def test_explicit_task_must_be_supported(self) -> None:
        strategy = QwenImageWorkflowStrategy()
        context = _context(tasks=["image.generate"])
        intent = WorkflowIntent(
            model_id="qwen-image-local",
            prompt="poster",
            task="image.edit",
        )

        with self.assertRaisesRegex(ValueError, "does not support workflow task"):
            strategy.plan(context, intent)


if __name__ == "__main__":
    unittest.main()
