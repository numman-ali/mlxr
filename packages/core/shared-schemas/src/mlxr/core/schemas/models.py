from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .sources import AuthRequirements, ProvenanceRecord, SourceRef


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


RecommendationTier = Literal["recommended", "advanced"]
SupportLevel = Literal["promoted", "supported"]
ModelInstallStatus = Literal["installed", "already_installed"]
ModelInstallOperationPhase = Literal[
    "queued",
    "resolving",
    "auth_required",
    "downloading",
    "converting",
    "registering",
    "completed",
    "failed",
    "cancelled",
]
ModelRemoveStatus = Literal["removed"]


class SupportedModelDescriptor(BaseModel):
    model_id: str
    display_name: str
    family: str
    family_variant: str | None = None
    recommendation_tier: RecommendationTier = "recommended"
    support_level: SupportLevel = "supported"
    tasks: list[str] = Field(default_factory=list)
    provider: str
    source_summary: str
    license: str | None = None
    access_state: str = "unknown"
    installed: bool = False
    installable: bool = True
    notes: str | None = None


class SupportedModelSourcePreview(BaseModel):
    role: str
    provider: str
    locator: dict[str, Any] = Field(default_factory=dict)
    resolved_ref: str | None = None
    access_state: str = "unknown"
    license: str | None = None
    bytes_total: int | None = None
    auth_requirements: AuthRequirements = Field(default_factory=AuthRequirements)
    remote_code_required: bool = False
    remote_code_approved: bool = False


class SupportedModelPreview(BaseModel):
    supported_model: SupportedModelDescriptor
    sources: list[SupportedModelSourcePreview] = Field(default_factory=list)
    total_source_bytes: int | None = None
    auth_required: bool = False
    auth_message: str | None = None


class ModelInstallRequest(BaseModel):
    model_id: str


class ModelInstallResult(BaseModel):
    status: ModelInstallStatus
    model: ModelRecord
    supported_model: SupportedModelDescriptor


class ModelInstallOperationRecord(BaseModel):
    operation_id: str
    model_id: str
    phase: ModelInstallOperationPhase
    supported_model: SupportedModelDescriptor
    preview: SupportedModelPreview | None = None
    result: ModelInstallResult | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstalledModelDetails(BaseModel):
    model: ModelRecord
    supported_model: SupportedModelDescriptor | None = None
    managed_storage_key: str | None = None
    managed_size_bytes: int | None = None
    referenced_source_ids: list[str] = Field(default_factory=list)


class ModelRemoveResult(BaseModel):
    status: ModelRemoveStatus
    model_id: str
    artifact_digest: str | None = None
    removed_source_ids: list[str] = Field(default_factory=list)
    removed_storage_key: str | None = None


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
