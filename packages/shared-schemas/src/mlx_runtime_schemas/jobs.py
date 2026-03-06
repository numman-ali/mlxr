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


class OutputDestinationMode(StrEnum):
    RUNTIME_MANAGED = "runtime_managed"
    TRUSTED_LOCAL_EXPORT = "trusted_local_export"


class JobOutputPolicy(BaseModel):
    artifact_format: str | None = None
    destination_mode: OutputDestinationMode = OutputDestinationMode.RUNTIME_MANAGED
    export_ref: str | None = None
    stream: bool = False


class InputImportRequest(BaseModel):
    content_base64: str
    media_type: str | None = None
    role: str | None = None
    filename: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InputHandle(BaseModel):
    handle_id: str
    media_type: str | None = None
    role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InputHandleRecord(InputHandle):
    filename: str | None = None
    size_bytes: int | None = None
    storage_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArtifactHandle(BaseModel):
    artifact_id: str
    artifact_format: str
    role: str = "result"
    exportable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutputArtifactRecord(ArtifactHandle):
    job_id: str
    filename: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    storage_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArtifactExportRequest(BaseModel):
    destination_path: str
    overwrite: bool = False


class ArtifactExportResult(BaseModel):
    artifact_id: str
    destination_path: str
    size_bytes: int | None = None


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
    artifacts: list[OutputArtifactRecord] = Field(default_factory=list)


class JobSubmitResult(BaseModel):
    job_id: str
    record: JobRecord
