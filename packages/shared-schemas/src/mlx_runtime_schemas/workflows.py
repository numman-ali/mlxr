from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .jobs import JobOutputPolicy, JobSubmitResult
from .models import CapabilityDescriptor


class WorkflowReference(BaseModel):
    input_handle: str
    kind: str
    role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowPreferences(BaseModel):
    natural_audio: bool = False
    no_music: bool = False
    enhance_prompt: bool = False
    quality: str = "auto"
    duration_seconds: float | None = None
    orientation: str | None = None


class WorkflowIntent(BaseModel):
    model_id: str
    prompt: str
    video_prompt: str | None = None
    audio_prompt: str | None = None
    references: list[WorkflowReference] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    output: JobOutputPolicy = Field(default_factory=JobOutputPolicy)
    preferences: WorkflowPreferences = Field(default_factory=WorkflowPreferences)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prompt(self) -> "WorkflowIntent":
        if not self.prompt.strip():
            raise ValueError("Workflow intent requires a non-empty prompt")
        return self


class WorkflowStageSpec(BaseModel):
    stage_id: str
    stage_type: str
    summary: str | None = None
    task: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    optional: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowPlan(BaseModel):
    model_id: str
    family: str
    scheduler_class: str | None = None
    selected_task: str
    selected_profile: str | None = None
    pipeline_variant: str | None = None
    resolved_prompt: str
    resolved_video_prompt: str | None = None
    resolved_audio_prompt: str | None = None
    references: list[WorkflowReference] = Field(default_factory=list)
    stages: list[WorkflowStageSpec] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowPlanResult(BaseModel):
    capability: CapabilityDescriptor
    plan: WorkflowPlan


class WorkflowRunRequest(BaseModel):
    intent: WorkflowIntent
    plan: WorkflowPlan | None = None


class WorkflowRunResult(BaseModel):
    plan: WorkflowPlan
    submit: JobSubmitResult
