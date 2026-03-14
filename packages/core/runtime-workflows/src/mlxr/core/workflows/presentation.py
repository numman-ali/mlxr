from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from mlxr.core.schemas import (
    WorkflowPlanPresentation,
    WorkflowPresentationControlOption,
    WorkflowPresentationControls,
    WorkflowPresentationReferenceSlot,
    WorkflowPresentationSubworkflow,
    WorkflowReferenceRequirement,
)

PresentationMode = Literal["image", "video", "audio"]
ReferenceKind = Literal["image", "video", "audio", "lora"]


def quality_presets() -> list[WorkflowPresentationControlOption]:
    return [
        WorkflowPresentationControlOption(
            value="draft",
            label="Draft",
            default=False,
        ),
        WorkflowPresentationControlOption(
            value="standard",
            label="Standard",
            default=True,
        ),
        WorkflowPresentationControlOption(
            value="cinema",
            label="Cinema",
            default=False,
        ),
    ]


def image_aspect_presets() -> list[WorkflowPresentationControlOption]:
    return [
        WorkflowPresentationControlOption(value="square", label="Square"),
        WorkflowPresentationControlOption(
            value="landscape",
            label="Landscape",
            default=True,
        ),
        WorkflowPresentationControlOption(value="portrait", label="Portrait"),
        WorkflowPresentationControlOption(value="story", label="Story"),
    ]


def video_aspect_presets() -> list[WorkflowPresentationControlOption]:
    return [
        WorkflowPresentationControlOption(value="square", label="Square"),
        WorkflowPresentationControlOption(
            value="landscape",
            label="Landscape",
            default=True,
        ),
        WorkflowPresentationControlOption(value="portrait", label="Portrait"),
    ]


def video_duration_presets() -> list[WorkflowPresentationControlOption]:
    return [
        WorkflowPresentationControlOption(value="short", label="4s"),
        WorkflowPresentationControlOption(
            value="medium",
            label="8s",
            default=True,
        ),
        WorkflowPresentationControlOption(value="long", label="12s"),
    ]


def image_variation_counts() -> list[int]:
    return [1, 2, 4]


def image_controls() -> WorkflowPresentationControls:
    return WorkflowPresentationControls(
        quality_presets=quality_presets(),
        aspect_presets=image_aspect_presets(),
        variation_counts=image_variation_counts(),
    )


def video_controls() -> WorkflowPresentationControls:
    return WorkflowPresentationControls(
        quality_presets=quality_presets(),
        aspect_presets=video_aspect_presets(),
        duration_presets=video_duration_presets(),
    )


def image_presentation(
    *,
    selected_task: str,
    subworkflows: Iterable[WorkflowPresentationSubworkflow],
    reference_slots: Iterable[WorkflowPresentationReferenceSlot] = (),
) -> WorkflowPlanPresentation:
    return WorkflowPlanPresentation(
        primary_mode="image",
        selected_task=selected_task,
        subworkflows=list(subworkflows),
        reference_slots=list(reference_slots),
        controls=image_controls(),
    )


def video_presentation(
    *,
    selected_task: str,
    subworkflows: Iterable[WorkflowPresentationSubworkflow],
    reference_slots: Iterable[WorkflowPresentationReferenceSlot] = (),
) -> WorkflowPlanPresentation:
    return WorkflowPlanPresentation(
        primary_mode="video",
        selected_task=selected_task,
        subworkflows=list(subworkflows),
        reference_slots=list(reference_slots),
        controls=video_controls(),
    )


def subworkflow(
    *,
    task: str,
    label: str,
    mode: PresentationMode,
    selected_task: str,
) -> WorkflowPresentationSubworkflow:
    return WorkflowPresentationSubworkflow(
        task=task,
        label=label,
        mode=mode,
        default=task == selected_task,
    )


def slot(
    *,
    slot_id: str,
    label: str,
    kind: ReferenceKind,
    description: str | None = None,
    required: bool = False,
    minimum_count: int = 0,
    maximum_count: int | None = None,
    accepted_roles: Iterable[str] = (),
    allows_multiple: bool = False,
) -> WorkflowPresentationReferenceSlot:
    return WorkflowPresentationReferenceSlot(
        slot_id=slot_id,
        label=label,
        kind=kind,
        description=description,
        required=required,
        minimum_count=minimum_count,
        maximum_count=maximum_count,
        accepted_roles=list(accepted_roles),
        allows_multiple=allows_multiple,
    )


def slots_from_requirements(
    requirements: Iterable[WorkflowReferenceRequirement],
) -> list[WorkflowPresentationReferenceSlot]:
    slots: list[WorkflowPresentationReferenceSlot] = []
    for index, requirement in enumerate(requirements, start=1):
        label = requirement.description.removesuffix(".")
        slots.append(
            slot(
                slot_id=f"{requirement.kind}-{index}",
                label=label,
                kind=requirement.kind,
                description=requirement.description,
                required=requirement.minimum_count > 0,
                minimum_count=requirement.minimum_count,
                maximum_count=requirement.maximum_count,
                accepted_roles=requirement.accepted_roles,
                allows_multiple=(
                    requirement.maximum_count is None or requirement.maximum_count > 1
                ),
            )
        )
    return slots
