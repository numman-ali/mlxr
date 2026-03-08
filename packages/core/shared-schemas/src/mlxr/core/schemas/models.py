from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .sources import ProvenanceRecord, SourceRef


class HardwareTier(BaseModel):
    tier: str
    memory_gb: int | None = None
    notes: str | None = None
    disabled_profiles: list[str] = Field(default_factory=list)


class PolicyDescriptor(BaseModel):
    license: str | None = None
    access_state: str = "unknown"
    remote_code_required: bool = False
    remote_code_approved: bool = False
    redistribution_state: str | None = None


class ExtensionSchemaDescriptor(BaseModel):
    namespace: str
    version: str


class CapabilityDescriptor(BaseModel):
    model_id: str
    artifact_digest: str
    family: str
    family_variant: str | None = None
    tasks: list[str] = Field(default_factory=list)
    modalities_in: list[str] = Field(default_factory=list)
    modalities_out: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    conditioning: dict[str, Any] = Field(default_factory=dict)
    profiles_by_task: dict[str, list[str]] = Field(default_factory=dict)
    streaming: dict[str, bool] = Field(default_factory=dict)
    artifacts_out: list[str] = Field(default_factory=list)
    scheduler_class: str
    hardware_tiers: list[HardwareTier] = Field(default_factory=list)
    dependencies: dict[str, Any] = Field(default_factory=dict)
    policy: PolicyDescriptor = Field(default_factory=PolicyDescriptor)
    extensions_schema: ExtensionSchemaDescriptor | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PortableArtifactComponentRecord(BaseModel):
    role: str
    kind: str
    relative_path: str
    storage_key: str = ""
    source_id: str
    resolved_ref: str | None = None
    size_bytes: int | None = None
    component_digest: str | None = None
    provenance: ProvenanceRecord
    metadata: dict[str, Any] = Field(default_factory=dict)


class PortableArtifactRecord(BaseModel):
    model_id: str
    artifact_digest: str
    family: str
    family_variant: str | None = None
    format_version: str
    weight_format: str
    storage_key: str
    capability: CapabilityDescriptor
    provenance: ProvenanceRecord
    components: list[PortableArtifactComponentRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRecord(BaseModel):
    model_id: str
    family: str
    source: SourceRef | None = None
    artifact: PortableArtifactRecord | None = None
    loaded: bool = False
    capability: CapabilityDescriptor | None = None


class ArtifactConversionRequest(BaseModel):
    source_id: str | None = None
    source_bindings: dict[str, str] | None = None
    family: str | None = None
    model_id: str
    precision: str = "bf16"
    target_format: str = "mlx_portable_bundle"
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source_selection(self) -> "ArtifactConversionRequest":
        has_source_id = isinstance(self.source_id, str) and bool(self.source_id.strip())
        has_source_bindings = bool(self.source_bindings)
        if has_source_id == has_source_bindings:
            raise ValueError(
                "Artifact conversion requires exactly one of source_id or source_bindings"
            )
        if has_source_bindings and self.family is None:
            raise ValueError(
                "Artifact conversion with source_bindings requires request.family"
            )
        if self.source_bindings is not None:
            for role, source_id in self.source_bindings.items():
                if not role.strip():
                    raise ValueError("Artifact conversion role names must be non-empty")
                if not source_id.strip():
                    raise ValueError(
                        "Artifact conversion source_bindings values must be non-empty"
                    )
        return self


class ArtifactConversionTimingsMs(BaseModel):
    fetch_ms_by_role: dict[str, float] = Field(default_factory=dict)
    fetch_total_ms: float
    family_convert_ms: float
    persist_ms: float
    total_ms: float


class ArtifactConversionResult(BaseModel):
    artifact: PortableArtifactRecord
    model: ModelRecord
    timings_ms: ArtifactConversionTimingsMs | None = None
