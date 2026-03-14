from __future__ import annotations

import unittest

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobOutputPolicy,
    JobRequest,
    ModelRecord,
    PolicyDescriptor,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowPlanPresentation,
    WorkflowPlanReadiness,
    WorkflowPresentationSubworkflow,
)
from mlxr.core.workflows import (
    WorkflowPlanner,
    WorkflowPlanningContext,
    WorkflowStrategyRegistry,
)


class _Strategy:
    family_id = "test"

    def plan(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
    ) -> WorkflowPlan:
        return WorkflowPlan(
            model_id=context.model.model_id,
            family=context.model.family,
            selected_task=intent.task or "image.generate",
            resolved_prompt=intent.prompt,
        )

    def to_job_request(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
        plan: WorkflowPlan,
    ) -> JobRequest:
        return JobRequest(
            model_id=context.model.model_id,
            task=plan.selected_task,
            inputs={"prompt": plan.resolved_prompt},
            params={},
            output=JobOutputPolicy(artifact_format="png"),
        )

    def readiness(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
        plan: WorkflowPlan,
    ) -> WorkflowPlanReadiness:
        return WorkflowPlanReadiness(ready=True)

    def presentation(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
        plan: WorkflowPlan,
        readiness: WorkflowPlanReadiness,
    ) -> WorkflowPlanPresentation:
        return WorkflowPlanPresentation(
            primary_mode="image",
            selected_task=plan.selected_task,
            subworkflows=[
                WorkflowPresentationSubworkflow(
                    task=plan.selected_task,
                    label="Generate from text",
                    mode="image",
                    default=True,
                )
            ],
        )


class WorkflowPlannerTests(unittest.TestCase):
    def test_plan_for_model_returns_strategy_presentation(self) -> None:
        registry = WorkflowStrategyRegistry()
        registry.register(_Strategy())
        planner = WorkflowPlanner(registry)
        model = ModelRecord(
            model_id="test-model",
            family="test",
            capability=CapabilityDescriptor(
                model_id="test-model",
                artifact_digest="sha256:test",
                family="test",
                tasks=["image.generate"],
                modalities_in=["text"],
                modalities_out=["image"],
                scheduler_class="image_diffusion",
                policy=PolicyDescriptor(access_state="public"),
                metadata={},
            ),
        )

        result = planner.plan_for_model(
            model=model,
            intent=WorkflowIntent(
                model_id="test-model",
                prompt="cinematic lighthouse",
                task="image.generate",
            ),
        )

        self.assertEqual(result.presentation.primary_mode, "image")
        self.assertEqual(result.presentation.selected_task, "image.generate")
        self.assertEqual(
            result.presentation.subworkflows[0].label, "Generate from text"
        )

    def test_capability_lookup_requires_model_or_artifact_capability(self) -> None:
        planner = WorkflowPlanner(WorkflowStrategyRegistry())
        model = ModelRecord(model_id="test-model", family="test")

        with self.assertRaisesRegex(ValueError, "has no capability descriptor"):
            planner._capability_for_model(model)


if __name__ == "__main__":
    unittest.main()
