from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mlxr.core.schemas import (
    ArtifactHandle,
    AuthRequirements,
    CapabilityDescriptor,
    PortableArtifactRecord,
    ProvenanceRecord,
    ResolvedSource,
    RuntimeEvent,
    SourceRef,
)


@dataclass(slots=True)
class ProviderInspection:
    resolved: ResolvedSource
    bytes_total: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FetchPolicy:
    allow_patterns: tuple[str, ...] = ()
    eager: bool = False
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceMaterialization:
    resolved: ResolvedSource
    provenance: ProvenanceRecord
    materialization_mode: str
    local_path: Path | None = None
    local_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FamilyInspection:
    family: str
    variant: str | None = None
    tasks: tuple[str, ...] = ()
    scheduler_class: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversionPlan:
    model_id: str
    precision: str = "bf16"
    target_format: str = "mlx_portable_bundle"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionProfile:
    task: str
    profile: str
    device: str = "gpu"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortableArtifact:
    record: PortableArtifactRecord
    storage_path: Path | None = None
    payload_items: tuple["ArtifactPayloadItem", ...] = ()


@dataclass(slots=True)
class ArtifactPayloadItem:
    source_path: Path
    relative_path: Path


@dataclass(slots=True)
class ConversionSource:
    role: str
    source_id: str
    source: SourceRef
    materialization: SourceMaterialization


@dataclass(slots=True)
class LoadedModelHandle:
    model_id: str
    family: str
    artifact_digest: str
    capability: CapabilityDescriptor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionStage:
    stage_id: str
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageResult:
    events: list[RuntimeEvent] = field(default_factory=list)
    artifacts: list[ArtifactHandle] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class SourceProviderAdapter(Protocol):
    provider_id: str

    def resolve(self, source_ref: SourceRef) -> ResolvedSource: ...

    def inspect(self, resolved: ResolvedSource) -> ProviderInspection: ...

    def auth_requirements(self, source_ref: SourceRef) -> AuthRequirements: ...

    def fetch(
        self, resolved: ResolvedSource, policy: FetchPolicy
    ) -> SourceMaterialization: ...

    def provenance(self, resolved: ResolvedSource) -> ProvenanceRecord: ...


class ModelFamilyAdapter(Protocol):
    family_id: str

    def inspect_source(self, source: ResolvedSource) -> FamilyInspection: ...

    def fetch_policy_for_conversion(
        self, role: str, source: ResolvedSource
    ) -> FetchPolicy: ...

    def convert(
        self, sources: dict[str, ConversionSource], plan: ConversionPlan
    ) -> PortableArtifact: ...

    def normalize_capability(
        self, artifact: PortableArtifact
    ) -> CapabilityDescriptor: ...

    def load(
        self, artifact: PortableArtifact, profile: ExecutionProfile
    ) -> LoadedModelHandle: ...

    def capabilities(self, artifact: PortableArtifact) -> CapabilityDescriptor: ...

    def run_stage(
        self, loaded: LoadedModelHandle, stage: ExecutionStage
    ) -> StageResult: ...

    def unload(self, loaded: LoadedModelHandle) -> None: ...
