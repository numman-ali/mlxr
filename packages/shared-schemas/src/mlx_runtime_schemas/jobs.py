from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobState(StrEnum):
    ACCEPTED = "accepted"
    PREPARING = "preparing"
    LOADING_MODEL = "loading_model"
    RUNNING = "running"
    STREAMING_OUTPUT = "streaming_output"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RuntimeEventKind(StrEnum):
    JOB_ACCEPTED = "job.accepted"
    JOB_PHASE_CHANGED = "job.phase_changed"
    JOB_PROGRESS = "job.progress"
    JOB_OUTPUT_DELTA = "job.output.delta"
    JOB_SEGMENT = "job.segment"
    JOB_AUDIO_CHUNK = "job.audio_chunk"
    JOB_ARTIFACT_READY = "job.artifact.ready"
    JOB_METRICS = "job.metrics"
    JOB_WARNING = "job.warning"
    JOB_FAILED = "job.failed"
    JOB_COMPLETED = "job.completed"
    JOB_CANCELLED = "job.cancelled"


class JobOutputPolicy(BaseModel):
    artifact_format: str | None = None
    output_dir: str | None = None
    stream: bool = False


class JobRequest(BaseModel):
    model_id: str
    task: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    output: JobOutputPolicy = Field(default_factory=JobOutputPolicy)
    extensions: dict[str, Any] = Field(default_factory=dict)


class RuntimeEvent(BaseModel):
    job_id: str
    kind: RuntimeEventKind
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    phase: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class JobRecord(BaseModel):
    job_id: str
    request: JobRequest
    state: JobState
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
