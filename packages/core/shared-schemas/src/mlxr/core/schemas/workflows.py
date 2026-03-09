from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .jobs import JobOutputPolicy, JobSubmitResult
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
    model_config = ConfigDict(extra="forbid")

    stage_id: str
    stage_type: str
    summary: str | None = None
    task: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    optional: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class WorkflowRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: WorkflowIntent


class WorkflowRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: WorkflowPlan
    submit: JobSubmitResult
