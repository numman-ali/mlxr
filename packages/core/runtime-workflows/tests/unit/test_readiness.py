from __future__ import annotations

import unittest

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobOutputPolicy,
    PolicyDescriptor,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowReference,
    WorkflowReferenceRequirement,
)
from mlxr.core.workflows.readiness import (
    _integer_constraint_issues,
    _numeric_constraint_issues,
    _reference_requirement_issues,
    build_plan_readiness,
)


def _capability(
    *, constraints: dict[str, object] | None = None
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        model_id="test-model",
        artifact_digest="sha256:test",
        family="test",
        scheduler_class="test.scheduler",
        tasks=["image.generate", "image.edit"],
        modalities_in=["text", "image"],
        modalities_out=["image"],
        artifacts_out=["png", "jpg"],
        policy=PolicyDescriptor(access_state="public"),
        constraints=constraints or {},
        metadata={},
    )


def _plan(*, task: str = "image.generate") -> WorkflowPlan:
    return WorkflowPlan(
        model_id="test-model",
        family="test",
        selected_task=task,
        resolved_prompt="prompt",
    )


class WorkflowReadinessTests(unittest.TestCase):
    def test_build_plan_readiness_reports_prompt_format_reference_and_warning_state(
        self,
    ) -> None:
        capability = _capability(
            constraints={
                "width": {"multiple_of": 16},
                "guidance_scale": {"fixed": 1.0},
            }
        )
        intent = WorkflowIntent(
            model_id="test-model",
            prompt="   ",
            task="image.edit",
            references=[
                WorkflowReference(kind="image", input_handle="inp_a"),
                WorkflowReference(kind="image", input_handle="inp_b"),
            ],
            params={"width": 513, "guidance_scale": 2.0},
            output=JobOutputPolicy(artifact_format="webp"),
        )
        plan = _plan(task="image.edit").model_copy(
            update={"warnings": ["This row is slower at large sizes."]}
        )

        readiness = build_plan_readiness(
            capability=capability,
            intent=intent,
            plan=plan,
            requirements=[
                WorkflowReferenceRequirement(
                    kind="image",
                    minimum_count=1,
                    maximum_count=1,
                    description="Choose exactly one image to edit.",
                )
            ],
            extra_blocking_issues=["Model-specific issue."],
        )

        self.assertFalse(readiness.ready)
        self.assertIn(
            "Add a prompt so MLXR knows what to make or change.",
            readiness.blocking_issues,
        )
        self.assertIn(
            "Output format 'webp' is not supported for this model.",
            readiness.blocking_issues,
        )
        self.assertIn(
            "width must be an integer multiple of 16.", readiness.blocking_issues
        )
        self.assertIn("guidance_scale must be exactly 1.0.", readiness.blocking_issues)
        self.assertIn(
            "Use no more than 1 image reference for this workflow.",
            readiness.blocking_issues,
        )
        self.assertIn("Model-specific issue.", readiness.blocking_issues)
        self.assertEqual(readiness.warnings, ["This row is slower at large sizes."])
        self.assertEqual(readiness.allowed_output_formats, ["png", "jpg"])

    def test_integer_constraint_issues_cover_type_fixed_minimum_maximum_and_formula(
        self,
    ) -> None:
        self.assertEqual(
            _integer_constraint_issues(
                name="width",
                value="512",
                constraints={"width": {"multiple_of": 16}},
            ),
            ["width must be an integer."],
        )
        self.assertIn(
            "num_frames must be exactly 9.",
            _integer_constraint_issues(
                name="num_frames",
                value=5,
                constraints={"num_frames": {"fixed": 9}},
            ),
        )
        self.assertIn(
            "height must be >= 256.",
            _integer_constraint_issues(
                name="height",
                value=128,
                constraints={"height": {"minimum": 256}},
            ),
        )
        self.assertIn(
            "height must be <= 1024.",
            _integer_constraint_issues(
                name="height",
                value=2048,
                constraints={"height": {"maximum": 1024}},
            ),
        )
        self.assertIn(
            "num_frames must satisfy the current 8n+1 rule.",
            _integer_constraint_issues(
                name="num_frames",
                value=10,
                constraints={"num_frames": {"formula": "8n+1"}},
            ),
        )

    def test_numeric_constraint_issues_cover_type_fixed_minimum_and_maximum(
        self,
    ) -> None:
        self.assertEqual(
            _numeric_constraint_issues(
                name="guidance_scale",
                value="1.0",
                constraints={"guidance_scale": {"fixed": 1.0}},
            ),
            ["guidance_scale must be numeric."],
        )
        self.assertIn(
            "guidance_scale must be exactly 1.0.",
            _numeric_constraint_issues(
                name="guidance_scale",
                value=2.5,
                constraints={"guidance_scale": {"fixed": 1.0}},
            ),
        )
        self.assertIn(
            "guidance_scale must be >= 0.5.",
            _numeric_constraint_issues(
                name="guidance_scale",
                value=0.25,
                constraints={"guidance_scale": {"minimum": 0.5}},
            ),
        )
        self.assertIn(
            "guidance_scale must be <= 7.5.",
            _numeric_constraint_issues(
                name="guidance_scale",
                value=8.0,
                constraints={"guidance_scale": {"maximum": 7.5}},
            ),
        )

    def test_reference_requirement_issues_handle_missing_and_too_many_references(
        self,
    ) -> None:
        requirements = [
            WorkflowReferenceRequirement(
                kind="video",
                minimum_count=1,
                maximum_count=1,
                description="Choose exactly one guide video.",
            ),
            WorkflowReferenceRequirement(
                kind="image",
                minimum_count=0,
                maximum_count=2,
                description="Optional image references.",
            ),
        ]

        missing = _reference_requirement_issues(counts={}, requirements=requirements)
        self.assertEqual(missing, ["Choose exactly one guide video."])

        too_many = _reference_requirement_issues(
            counts={"video": 1, "image": 3},
            requirements=requirements,
        )
        self.assertEqual(
            too_many,
            ["Use no more than 2 image references for this workflow."],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
