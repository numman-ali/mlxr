from __future__ import annotations

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobRequest,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowPlanReadiness,
    WorkflowReference,
    WorkflowReferenceRequirement,
    WorkflowStageSpec,
)
from mlxr.core.workflows import FamilyWorkflowStrategy, WorkflowPlanningContext
from mlxr.core.workflows.readiness import build_plan_readiness

from .family_options import (
    conditioning_attention_strength_from_extensions,
    control_variant_from_extensions,
    family_extensions,
    workflow_variant_from_extensions,
)


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

        if intent.task is None and by_kind.get("video"):
            raise ValueError(
                "LTX workflows with video references currently require an explicit task"
            )
        task = intent.task or _selected_task(by_kind)
        if task not in capability.tasks:
            raise ValueError(
                f"Model '{context.model.model_id}' does not support workflow task '{task}'"
            )
        if by_kind.get("lora") and task != "video.condition.video":
            raise ValueError(
                "LTX LoRA references are currently only supported for video.condition.video workflows"
            )
        pipeline_variant = workflow_variant_from_extensions(
            intent.extensions.get("ltx"),
            supported_variants=tuple(_pipeline_variants(capability)),
            default=_default_pipeline_variant(capability, task),
        )
        _validate_task_pipeline_variant(task, pipeline_variant)
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
        if intent.negative_prompt is not None:
            inputs["negative_prompt"] = intent.negative_prompt
        if plan.selected_task in {
            "video.condition.image",
            "video.condition.audio",
            "video.condition.video",
            "video.interpolate",
        }:
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
                if plan.selected_task in {
                    "video.condition.image",
                    "video.interpolate",
                }:
                    raise ValueError(
                        f"{plan.selected_task} requires at least one bound image reference"
                    )
            else:
                if plan.selected_task == "video.interpolate":
                    if len(images) < 2:
                        raise ValueError(
                            "video.interpolate requires at least two bound image references"
                        )
                    if len({image["frame_index"] for image in images}) < 2:
                        raise ValueError(
                            "video.interpolate requires image references at at least two distinct frame indices"
                        )
                inputs["images"] = images

        if plan.selected_task in {"video.condition.video", "video.retake"}:
            videos: list[dict[str, object]] = []
            for reference in intent.references:
                if reference.kind != "video":
                    continue
                if reference.input_handle is None:
                    raise ValueError(
                        f"{plan.selected_task} workflow execution requires bound video input handles"
                    )
                strength = reference.metadata.get("strength", 1.0)
                if not isinstance(strength, (int, float)):
                    raise ValueError("Workflow video strength must be numeric")
                strength_value = float(strength)
                if not (0.0 <= strength_value <= 1.0):
                    raise ValueError(
                        "Workflow video strength must be between 0.0 and 1.0"
                    )
                videos.append(
                    {
                        "input_handle": reference.input_handle,
                        "strength": strength_value,
                    }
                )
            if plan.selected_task == "video.condition.video":
                if not videos:
                    raise ValueError(
                        "video.condition.video requires at least one bound video reference"
                    )
            elif len(videos) != 1:
                raise ValueError(
                    "video.retake requires exactly one bound source video reference"
                )
            if videos:
                inputs["videos"] = videos

        if plan.selected_task == "video.condition.video":
            loras: list[dict[str, object]] = []
            for reference in intent.references:
                if reference.kind != "lora":
                    continue
                if reference.input_handle is None:
                    raise ValueError(
                        "video.condition.video workflow execution requires bound LoRA input handles"
                    )
                strength = reference.metadata.get("strength", 1.0)
                if not isinstance(strength, (int, float)):
                    raise ValueError("Workflow LoRA strength must be numeric")
                strength_value = float(strength)
                if strength_value <= 0.0:
                    raise ValueError("Workflow LoRA strength must be greater than 0.0")
                loras.append(
                    {
                        "input_handle": reference.input_handle,
                        "strength": strength_value,
                    }
                )
            if len(loras) != 1:
                raise ValueError(
                    "video.condition.video requires exactly one bound LoRA reference"
                )
            inputs["loras"] = loras

        lora_references = [
            _lora_reference_payload(reference)
            for reference in intent.references
            if reference.kind == "lora"
        ]
        if lora_references and plan.selected_task != "video.condition.video":
            raise ValueError(
                "LTX LoRA references are only valid for video.condition.video workflows"
            )
        if plan.selected_task == "video.condition.video":
            if len(lora_references) != 1:
                raise ValueError(
                    "video.condition.video requires exactly one bound LoRA reference"
                )
            inputs["loras"] = lora_references

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

        if plan.selected_task == "video.retake":
            start_seconds = intent.params.get("window_start_seconds")
            end_seconds = intent.params.get("window_end_seconds")
            if not isinstance(start_seconds, (int, float)):
                raise ValueError(
                    "video.retake requires numeric params.window_start_seconds"
                )
            if not isinstance(end_seconds, (int, float)):
                raise ValueError(
                    "video.retake requires numeric params.window_end_seconds"
                )
            if float(start_seconds) < 0.0:
                raise ValueError(
                    "video.retake params.window_start_seconds must be >= 0.0"
                )
            if float(end_seconds) <= float(start_seconds):
                raise ValueError(
                    "video.retake params.window_end_seconds must be greater than params.window_start_seconds"
                )
            regenerate_video = intent.params.get("regenerate_video", True)
            regenerate_audio = intent.params.get("regenerate_audio", True)
            if not isinstance(regenerate_video, bool):
                raise ValueError(
                    "video.retake params.regenerate_video must be boolean when provided"
                )
            if not isinstance(regenerate_audio, bool):
                raise ValueError(
                    "video.retake params.regenerate_audio must be boolean when provided"
                )

        extensions = dict(intent.extensions)
        ltx_extensions = dict(family_extensions(extensions.get("ltx")))
        control_variant = control_variant_from_extensions(ltx_extensions)
        conditioning_attention_strength = (
            conditioning_attention_strength_from_extensions(ltx_extensions)
        )
        if (
            control_variant is not None or conditioning_attention_strength is not None
        ) and plan.selected_task != "video.condition.video":
            raise ValueError(
                "LTX control_variant and conditioning_attention_strength are only valid "
                "for video.condition.video workflows"
            )
        if plan.selected_task == "video.condition.video" and control_variant is None:
            ltx_extensions["control_variant"] = "ic_lora"
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

    def readiness(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
        plan: WorkflowPlan,
    ) -> WorkflowPlanReadiness:
        return _readiness_for_plan(
            capability=context.capability,
            intent=intent,
            plan=plan,
            requirements=_reference_requirements_for_task(plan.selected_task),
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


def _default_pipeline_variant(
    capability: CapabilityDescriptor, task: str
) -> str | None:
    variants = _pipeline_variants(capability)
    if task == "video.interpolate" and "two_stage" in variants:
        return "two_stage"
    if variants:
        return variants[0]
    return None


def _pipeline_variants(capability: CapabilityDescriptor) -> list[str]:
    implemented_surface = capability.metadata.get("implemented_surface", {})
    variants = implemented_surface.get("pipeline_variants", [])
    if isinstance(variants, list) and variants:
        return [variant for variant in variants if isinstance(variant, str)]
    return []


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
        # Multiple image refs still ride the existing image-conditioned row for now.
        # Official keyframe interpolation is a separate future pipeline with
        # different token-level conditioning semantics.
        return "video.condition.image"
    return "video.generate"


def _validate_task_pipeline_variant(task: str, pipeline_variant: str | None) -> None:
    if task == "video.condition.video" and pipeline_variant != "distilled_two_stage":
        raise ValueError(
            "LTX video.condition.video currently requires workflow variant 'distilled_two_stage'"
        )
    if task == "video.interpolate" and pipeline_variant != "two_stage":
        raise ValueError(
            "LTX video.interpolate currently requires workflow variant 'two_stage'"
        )


def _lora_reference_payload(reference: WorkflowReference) -> dict[str, object]:
    if reference.input_handle is None:
        raise ValueError("LTX LoRA execution requires bound input handles")
    strength = reference.metadata.get("strength", 1.0)
    if not isinstance(strength, (int, float)):
        raise ValueError("Workflow LoRA strength must be numeric")
    strength_value = float(strength)
    if strength_value <= 0.0:
        raise ValueError("Workflow LoRA strength must be greater than 0.0")
    return {
        "input_handle": reference.input_handle,
        "strength": strength_value,
    }


def _reference_requirements_for_task(
    task: str,
) -> list[WorkflowReferenceRequirement]:
    if task == "video.condition.image":
        return [
            WorkflowReferenceRequirement(
                kind="image",
                minimum_count=1,
                description="Choose at least one image to animate.",
            )
        ]
    if task == "video.condition.audio":
        return [
            WorkflowReferenceRequirement(
                kind="audio",
                minimum_count=1,
                maximum_count=1,
                description="Choose exactly one audio asset to drive the clip.",
            )
        ]
    if task == "video.condition.video":
        return [
            WorkflowReferenceRequirement(
                kind="video",
                minimum_count=1,
                maximum_count=1,
                description="Choose exactly one guide video.",
            ),
            WorkflowReferenceRequirement(
                kind="lora",
                minimum_count=1,
                maximum_count=1,
                description="Choose exactly one compatible control LoRA.",
            ),
        ]
    if task == "video.interpolate":
        return [
            WorkflowReferenceRequirement(
                kind="image",
                minimum_count=2,
                description="Choose at least two key images to blend between.",
            )
        ]
    if task == "video.retake":
        return [
            WorkflowReferenceRequirement(
                kind="video",
                minimum_count=1,
                maximum_count=1,
                description="Choose exactly one source video to retake.",
            )
        ]
    return []


def _readiness_for_plan(
    *,
    capability: CapabilityDescriptor,
    intent: WorkflowIntent,
    plan: WorkflowPlan,
    requirements: list[WorkflowReferenceRequirement],
) -> WorkflowPlanReadiness:
    extra_blocking_issues: list[str] = []
    counts = {
        kind: len(references)
        for kind, references in _references_by_kind(tuple(intent.references)).items()
    }
    if plan.selected_task == "video.interpolate":
        frame_indices = {
            int(reference.metadata.get("frame_index", 0))
            for reference in intent.references
            if reference.kind == "image"
            and isinstance(reference.metadata.get("frame_index", 0), int)
        }
        if counts.get("image", 0) >= 2 and len(frame_indices) < 2:
            extra_blocking_issues.append(
                "Set at least two distinct keyframe positions for interpolation."
            )

    if plan.selected_task == "video.retake" and (
        not isinstance(intent.params.get("window_start_seconds"), (int, float))
        or not isinstance(intent.params.get("window_end_seconds"), (int, float))
    ):
        extra_blocking_issues.append(
            "Set a valid retake window before running this workflow."
        )

    return build_plan_readiness(
        capability=capability,
        intent=intent,
        plan=plan,
        requirements=requirements,
        extra_blocking_issues=extra_blocking_issues,
    )
