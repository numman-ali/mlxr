from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RuntimeLimits(BaseModel):
    recommended_memory_gb: int | None = None
    min_memory_gb: int | None = None


class CapabilityDescriptor(BaseModel):
    model_id: str
    family: str
    tasks: list[str]
    modalities_in: list[str] = Field(default_factory=list)
    modalities_out: list[str] = Field(default_factory=list)
    conditioning: dict[str, bool] = Field(default_factory=dict)
    profiles: list[str] = Field(default_factory=list)
    streaming: dict[str, bool] = Field(default_factory=dict)
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRecord(BaseModel):
    model_id: str
    family: str
    source_ref: str | None = None
    artifact_ref: str | None = None
    loaded: bool = False
    capabilities: CapabilityDescriptor | None = None
