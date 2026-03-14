from __future__ import annotations

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobRequest,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowPlanPresentation,
    WorkflowPlanReadiness,
    WorkflowReference,
    WorkflowReferenceRequirement,
    WorkflowStageSpec,
)
from mlxr.core.workflows import FamilyWorkflowStrategy, WorkflowPlanningContext
from mlxr.core.workflows.presentation import (
    image_presentation,
    slot,
    subworkflow,
)
from mlxr.core.workflows.readiness import build_plan_readiness

from .family_options import family_extensions


class QwenImageWorkflowStrategy(FamilyWorkflowStrategy):
    family_id = "qwen_image"

    def plan(
        self, context: WorkflowPlanningContext, intent: WorkflowIntent
    ) -> WorkflowPlan:
        _validate_references(intent.references)
        task = _selected_task(intent)
        if task not in context.capability.tasks:
            raise ValueError(
                f"Model '{context.model.model_id}' does not support workflow task '{task}'"
            )
        stage_summary = (
            "Run the selected Qwen-Image text-to-image task through the shared runtime job API."
            if task == "image.generate"
            else "Run the selected Qwen-Image editing task through the shared runtime job API."
        )
        return WorkflowPlan(
            model_id=context.model.model_id,
            family=context.model.family,
            scheduler_class=context.capability.scheduler_class,
            selected_task=task,
            selected_profile=_selected_profile(context.capability, task),
            pipeline_variant=context.model.capability.family_variant
            if context.model.capability is not None
            else context.capability.family_variant,
            resolved_prompt=intent.prompt,
            references=list(intent.references),
            stages=[
                WorkflowStageSpec(
                    stage_id="generate_image",
                    stage_type="runtime_job",
                    summary=stage_summary,
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
                "implemented_task_surface": list(context.capability.tasks),
                "supported_reference_kinds": ["image", "lora"],
                "workflow_mode": "generation" if task == "image.generate" else "edit",
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
        if plan.selected_task == "image.edit":
            inputs["images"] = [
                _image_reference_payload(reference)
                for reference in plan.references
                if reference.kind == "image"
            ]
        lora_references = [
            _lora_reference_payload(reference)
            for reference in plan.references
            if reference.kind == "lora"
        ]
        if lora_references:
            inputs["loras"] = lora_references
        extensions = dict(intent.extensions)
        qwen_extensions = dict(family_extensions(extensions.get("qwen_image")))
        if qwen_extensions:
            extensions["qwen_image"] = qwen_extensions
        extensions["workflow"] = {
            "selected_task": plan.selected_task,
            "selected_profile": plan.selected_profile,
            "reference_count": len(plan.references),
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
        requirements: list[WorkflowReferenceRequirement] = []
        if plan.selected_task == "image.edit":
            requirements.append(
                WorkflowReferenceRequirement(
                    kind="image",
                    minimum_count=1,
                    description="Choose at least one image to edit.",
                )
            )
        return _readiness_for_plan(
            capability=context.capability,
            intent=intent,
            plan=plan,
            requirements=requirements,
        )

    def presentation(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
        plan: WorkflowPlan,
        readiness: WorkflowPlanReadiness,
    ) -> WorkflowPlanPresentation:
        reference_slots = (
            [
                slot(
                    slot_id="source-image",
                    label="Source image",
                    kind="image",
                    description="Choose at least one image to edit.",
                    required=True,
                    minimum_count=1,
                    allows_multiple=True,
                )
            ]
            if plan.selected_task == "image.edit"
            else []
        )
        return image_presentation(
            selected_task=plan.selected_task,
            subworkflows=[
                subworkflow(
                    task="image.generate",
                    label="Generate from text",
                    mode="image",
                    selected_task=plan.selected_task,
                ),
                subworkflow(
                    task="image.edit",
                    label="Edit an existing image",
                    mode="image",
                    selected_task=plan.selected_task,
                ),
            ],
            reference_slots=reference_slots,
        )


def _validate_references(references: list[WorkflowReference]) -> None:
    for reference in references:
        if reference.kind not in {"image", "lora"}:
            raise ValueError(
                "The current Qwen-Image workflow strategy only supports image and lora references"
            )


def _selected_task(intent: WorkflowIntent) -> str:
    if intent.task is not None:
        return intent.task
    if any(reference.kind == "image" for reference in intent.references):
        return "image.edit"
    return "image.generate"


def _selected_profile(capability: CapabilityDescriptor, task: str) -> str | None:
    profiles = capability.profiles_by_task.get(task, [])
    if profiles:
        return profiles[0]
    return None


def _image_reference_payload(reference: WorkflowReference) -> dict[str, object]:
    payload: dict[str, object] = {}
    if reference.input_handle is not None:
        payload["input_handle"] = reference.input_handle
    if reference.role is not None:
        payload["role"] = reference.role
    if reference.metadata:
        payload["metadata"] = dict(reference.metadata)
    return payload


def _lora_reference_payload(reference: WorkflowReference) -> dict[str, object]:
    payload: dict[str, object] = {}
    if reference.input_handle is not None:
        payload["input_handle"] = reference.input_handle
    if reference.metadata:
        payload["strength"] = float(reference.metadata.get("strength", 1.0))
    return payload


def _readiness_for_plan(
    *,
    capability: CapabilityDescriptor,
    intent: WorkflowIntent,
    plan: WorkflowPlan,
    requirements: list[WorkflowReferenceRequirement],
) -> WorkflowPlanReadiness:
    return build_plan_readiness(
        capability=capability,
        intent=intent,
        plan=plan,
        requirements=requirements,
    )
