from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mlx_runtime_schemas import CapabilityDescriptor, JobRecord, JobRequest, RuntimeEvent


@dataclass(slots=True)
class SourceRef:
    uri: str
    revision: str | None = None
    trust_remote_code: bool = False
    family_hint: str | None = None


@dataclass(slots=True)
class ResolvedSource:
    ref: SourceRef
    local_path: Path
    allow_patterns: tuple[str, ...] = ()
    source_hash: str | None = None
    detected_layout: str | None = None


@dataclass(slots=True)
class SourceInspection:
    family: str
    variant: str | None = None
    tasks: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversionProfile:
    precision: str = "bf16"
    compile_profile: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeProfile:
    execution_mode: str = "media"
    device: str = "gpu"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelArtifact:
    artifact_path: Path
    format_version: str
    weight_format: str
    capability: CapabilityDescriptor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoadedHandle:
    model_id: str
    family: str
    capability: CapabilityDescriptor
    metadata: dict[str, Any] = field(default_factory=dict)


class FamilyAdapter(Protocol):
    family_id: str

    def resolve_source(self, ref: SourceRef) -> ResolvedSource: ...

    def inspect_source(self, source: ResolvedSource) -> SourceInspection: ...

    def convert(self, source: ResolvedSource, profile: ConversionProfile) -> ModelArtifact: ...

    def load(self, artifact: ModelArtifact, runtime: RuntimeProfile) -> LoadedHandle: ...

    def capabilities(self, handle: LoadedHandle | None = None) -> CapabilityDescriptor: ...

    def execute(self, handle: LoadedHandle, job: JobRecord) -> list[RuntimeEvent]: ...

    def unload(self, handle: LoadedHandle) -> None: ...
