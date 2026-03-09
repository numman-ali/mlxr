from __future__ import annotations

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobRequest,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowReference,
    WorkflowStageSpec,
)
from mlxr.core.workflows import FamilyWorkflowStrategy, WorkflowPlanningContext

from .family_options import family_extensions


class LTXWorkflowStrategy(FamilyWorkflowStrategy):
    family_id = "ltx"

    def plan(
        self, context: WorkflowPlanningContext, intent: WorkflowIntent
    ) -> WorkflowPlan:
        capability = context.capability
        references = tuple(intent.references)
        by_kind = _references_by_kind(references)
        warnings: list[str] = []

        supported_kinds = set(_supported_reference_kinds(capability))
        unsupported_kinds = sorted(
            kind for kind in by_kind if kind not in supported_kinds
        )
        if unsupported_kinds:
            unsupported = ", ".join(sorted(unsupported_kinds))
            raise ValueError(
                f"The current LTX workflow strategy does not support {unsupported} references yet"
            )

        task = _selected_task(by_kind)
        if task not in capability.tasks:
            raise ValueError(
                f"Model '{context.model.model_id}' does not support workflow task '{task}'"
            )
        pipeline_variant = _pipeline_variant(capability)
        selected_profile = _selected_profile(capability, task)

        return WorkflowPlan(
            model_id=context.model.model_id,
            family=context.model.family,
            scheduler_class=capability.scheduler_class,
            selected_task=task,
            selected_profile=selected_profile,
            pipeline_variant=pipeline_variant,
            resolved_prompt=intent.prompt,
            references=list(references),
            stages=[
                WorkflowStageSpec(
                    stage_id="generate_media",
                    stage_type="runtime_job",
                    summary="Run the selected LTX generation task through the shared runtime job API.",
                    task=task,
                    inputs={"reference_kinds": sorted(by_kind.keys())},
                    params={
                        "artifact_format": intent.output.artifact_format,
                        "pipeline_variant": pipeline_variant,
                    },
                    metadata={
                        "family": context.model.family,
                        "quality_preference": intent.preferences.quality,
                    },
                )
            ],
            warnings=warnings,
            metadata={
                "implemented_task_surface": list(capability.tasks),
                "supported_reference_kinds": _supported_reference_kinds(capability),
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
        if plan.selected_task in {"video.condition.image", "video.condition.audio"}:
            images: list[dict[str, object]] = []
            for reference in intent.references:
                if reference.kind != "image":
                    continue
                if reference.input_handle is None:
                    raise ValueError(
                        "Image-conditioned workflow execution requires bound input handles"
                    )
                frame_index = reference.metadata.get("frame_index", 0)
                strength = reference.metadata.get("strength", 1.0)
                if not isinstance(frame_index, int) or frame_index < 0:
                    raise ValueError(
                        "Workflow image frame_index must be a non-negative integer"
                    )
                if not isinstance(strength, (int, float)):
                    raise ValueError("Workflow image strength must be numeric")
                strength_value = float(strength)
                if not (0.0 <= strength_value <= 1.0):
                    raise ValueError(
                        "Workflow image strength must be between 0.0 and 1.0"
                    )
                images.append(
                    {
                        "input_handle": reference.input_handle,
                        "frame_index": frame_index,
                        "strength": strength_value,
                    }
                )
            if not images:
                if plan.selected_task == "video.condition.image":
                    raise ValueError(
                        "Image-conditioned workflow execution requires at least one bound image reference"
                    )
            else:
                inputs["images"] = images

        if plan.selected_task == "video.condition.audio":
            audio_references = [
                reference
                for reference in intent.references
                if reference.kind == "audio"
            ]
            if len(audio_references) != 1:
                raise ValueError(
                    "Audio-conditioned workflow execution requires exactly one bound audio reference"
                )
            audio_reference = audio_references[0]
            if audio_reference.input_handle is None:
                raise ValueError(
                    "Audio-conditioned workflow execution requires a bound audio input handle"
                )
            start_time_seconds = audio_reference.metadata.get("start_time_seconds", 0.0)
            max_duration_seconds = audio_reference.metadata.get("max_duration_seconds")
            if not isinstance(start_time_seconds, (int, float)):
                raise ValueError("Workflow audio start_time_seconds must be numeric")
            start_time_value = float(start_time_seconds)
            if start_time_value < 0.0:
                raise ValueError("Workflow audio start_time_seconds must be >= 0.0")
            max_duration_value: float | None = None
            if max_duration_seconds is not None:
                if not isinstance(max_duration_seconds, (int, float)):
                    raise ValueError(
                        "Workflow audio max_duration_seconds must be numeric when provided"
                    )
                max_duration_value = float(max_duration_seconds)
                if max_duration_value <= 0.0:
                    raise ValueError(
                        "Workflow audio max_duration_seconds must be > 0.0 when provided"
                    )
            inputs["audio"] = {
                "input_handle": audio_reference.input_handle,
                "start_time_seconds": start_time_value,
                "max_duration_seconds": max_duration_value,
            }

        extensions = dict(intent.extensions)
        ltx_extensions = dict(family_extensions(extensions.get("ltx")))
        ltx_extensions.update(
            {
                "workflow_variant": plan.pipeline_variant,
            }
        )
        extensions["ltx"] = ltx_extensions
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


def _references_by_kind(
    references: tuple[WorkflowReference, ...],
) -> dict[str, list[WorkflowReference]]:
    grouped: dict[str, list[WorkflowReference]] = {}
    for reference in references:
        grouped.setdefault(reference.kind, []).append(reference)
    return grouped


def _selected_profile(capability: CapabilityDescriptor, task: str) -> str | None:
    profiles = capability.profiles_by_task.get(task, [])
    if profiles:
        return profiles[0]
    return None


def _pipeline_variant(capability: CapabilityDescriptor) -> str | None:
    implemented_surface = capability.metadata.get("implemented_surface", {})
    variants = implemented_surface.get("pipeline_variants", [])
    if isinstance(variants, list) and variants:
        first = variants[0]
        if isinstance(first, str):
            return first
    return None


def _supported_reference_kinds(capability: CapabilityDescriptor) -> list[str]:
    conditioning = capability.conditioning
    supported: list[str] = []
    for kind in ("image", "video", "audio", "lora"):
        if conditioning.get(kind):
            supported.append(kind)
    return supported


def _selected_task(by_kind: dict[str, list[WorkflowReference]]) -> str:
    if by_kind.get("audio"):
        return "video.condition.audio"
    if by_kind.get("image"):
        return "video.condition.image"
    return "video.generate"
