from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .jobs import JobOutputPolicy, JobSubmitResult, WorkflowContextMetadata
from .models import CapabilityDescriptor


class WorkflowReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_handle: str | None = None
    kind: Literal["image", "video", "audio", "lora"]
    role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_handle(self) -> "WorkflowReference":
        if self.input_handle is not None and not self.input_handle.strip():
            raise ValueError("Workflow reference input_handle must be non-empty")
        return self


class WorkflowPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality: Literal["auto", "fast", "balanced", "high"] = "auto"


class WorkflowIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    prompt: str
    task: str | None = None
    negative_prompt: str | None = None
    references: list[WorkflowReference] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    output: JobOutputPolicy = Field(default_factory=JobOutputPolicy)
    preferences: WorkflowPreferences = Field(default_factory=WorkflowPreferences)
    context: WorkflowContextMetadata | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)


class WorkflowStageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str
    stage_type: str
    summary: str | None = None
    task: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    optional: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowReferenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["image", "video", "audio", "lora"]
    minimum_count: int = 0
    maximum_count: int | None = None
    accepted_roles: list[str] = Field(default_factory=list)
    description: str


class WorkflowPlanReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool = True
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reference_requirements: list[WorkflowReferenceRequirement] = Field(
        default_factory=list
    )
    allowed_output_formats: list[str] = Field(default_factory=list)


class WorkflowPresentationControlOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    label: str
    default: bool = False


class WorkflowPresentationControls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_presets: list[WorkflowPresentationControlOption] = Field(
        default_factory=list
    )
    aspect_presets: list[WorkflowPresentationControlOption] = Field(
        default_factory=list
    )
    duration_presets: list[WorkflowPresentationControlOption] = Field(
        default_factory=list
    )
    variation_counts: list[int] = Field(default_factory=list)


class WorkflowPresentationSubworkflow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    label: str
    mode: Literal["image", "video", "audio"]
    default: bool = False


class WorkflowPresentationReferenceSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str
    label: str
    kind: Literal["image", "video", "audio", "lora"]
    description: str | None = None
    required: bool = False
    minimum_count: int = 0
    maximum_count: int | None = None
    accepted_roles: list[str] = Field(default_factory=list)
    allows_multiple: bool = False


class WorkflowPlanPresentation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_mode: Literal["image", "video", "audio"]
    selected_task: str
    subworkflows: list[WorkflowPresentationSubworkflow] = Field(default_factory=list)
    reference_slots: list[WorkflowPresentationReferenceSlot] = Field(
        default_factory=list
    )
    controls: WorkflowPresentationControls = Field(
        default_factory=WorkflowPresentationControls
    )


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    family: str
    scheduler_class: str | None = None
    selected_task: str
    selected_profile: str | None = None
    pipeline_variant: str | None = None
    resolved_prompt: str
    references: list[WorkflowReference] = Field(default_factory=list)
    stages: list[WorkflowStageSpec] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowPlanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: CapabilityDescriptor
    plan: WorkflowPlan
    readiness: WorkflowPlanReadiness = Field(default_factory=WorkflowPlanReadiness)
    presentation: WorkflowPlanPresentation = Field(
        default_factory=lambda: WorkflowPlanPresentation(
            primary_mode="image",
            selected_task="image.generate",
        )
    )


class WorkflowRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: WorkflowIntent


class WorkflowRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: WorkflowPlan
    submit: JobSubmitResult
