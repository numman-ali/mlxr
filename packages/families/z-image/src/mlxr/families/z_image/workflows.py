from __future__ import annotations

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobRequest,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowPlanPresentation,
    WorkflowPlanReadiness,
    WorkflowStageSpec,
)
from mlxr.core.workflows import FamilyWorkflowStrategy, WorkflowPlanningContext
from mlxr.core.workflows.presentation import (
    image_presentation,
    subworkflow,
)
from mlxr.core.workflows.readiness import build_plan_readiness

from .family_options import family_extensions


class ZImageWorkflowStrategy(FamilyWorkflowStrategy):
    family_id = "z_image"

    def plan(
        self, context: WorkflowPlanningContext, intent: WorkflowIntent
    ) -> WorkflowPlan:
        capability = context.capability
        if intent.references:
            kinds = ", ".join(
                sorted({reference.kind for reference in intent.references})
            )
            raise ValueError(
                "The current Z-Image workflow strategy does not support conditioning "
                f"references yet; got {kinds}"
            )
        task = intent.task or "image.generate"
        if task not in capability.tasks:
            raise ValueError(
                f"Model '{context.model.model_id}' does not support workflow task '{task}'"
            )
        return WorkflowPlan(
            model_id=context.model.model_id,
            family=context.model.family,
            scheduler_class=capability.scheduler_class,
            selected_task=task,
            selected_profile=_selected_profile(capability, task),
            pipeline_variant=context.model.capability.family_variant
            if context.model.capability is not None
            else capability.family_variant,
            resolved_prompt=intent.prompt,
            references=[],
            stages=[
                WorkflowStageSpec(
                    stage_id="generate_image",
                    stage_type="runtime_job",
                    summary="Run the selected Z-Image still-image generation task through the shared runtime job API.",
                    task=task,
                    params={"artifact_format": intent.output.artifact_format},
                    metadata={
                        "family": context.model.family,
                        "quality_preference": intent.preferences.quality,
                    },
                )
            ],
            warnings=[],
            metadata={
                "implemented_task_surface": list(capability.tasks),
                "supported_reference_kinds": [],
                "workflow_mode": "simple_generation",
            },
        )

    def to_job_request(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
        plan: WorkflowPlan,
    ) -> JobRequest:
        if plan.selected_task not in context.capability.tasks:
            raise ValueError(
                f"Model '{context.model.model_id}' does not support task '{plan.selected_task}'"
            )
        inputs: dict[str, object] = {"prompt": plan.resolved_prompt}
        if intent.negative_prompt is not None:
            inputs["negative_prompt"] = intent.negative_prompt
        extensions = dict(intent.extensions)
        z_image_extensions = dict(family_extensions(extensions.get("z_image")))
        if z_image_extensions:
            extensions["z_image"] = z_image_extensions
        extensions["workflow"] = {
            "selected_task": plan.selected_task,
            "selected_profile": plan.selected_profile,
            "reference_count": 0,
            "warnings": list(plan.warnings),
        }
        return JobRequest(
            model_id=context.model.model_id,
            task=plan.selected_task,
            inputs=inputs,
            params=dict(intent.params),
            output=intent.output,
            extensions=extensions,
        )

    def readiness(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
        plan: WorkflowPlan,
    ) -> WorkflowPlanReadiness:
        return build_plan_readiness(
            capability=context.capability,
            intent=intent,
            plan=plan,
            requirements=[],
        )

    def presentation(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
        plan: WorkflowPlan,
        readiness: WorkflowPlanReadiness,
    ) -> WorkflowPlanPresentation:
        return image_presentation(
            selected_task=plan.selected_task,
            subworkflows=[
                subworkflow(
                    task="image.generate",
                    label="Generate from text",
                    mode="image",
                    selected_task=plan.selected_task,
                )
            ],
        )


def _selected_profile(capability: CapabilityDescriptor, task: str) -> str | None:
    profiles = capability.profiles_by_task.get(task, [])
    if profiles:
        return profiles[0]
    return None
