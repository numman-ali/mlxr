from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class SourceMaterializationRequest(BaseModel):
    mode: str = "provider-cache-ref"


class SourceAuth(BaseModel):
    token_ref: str | None = None


class SourcePolicy(BaseModel):
    allow_remote_code: bool = False


class SourceRef(BaseModel):
    provider: str
    locator: dict[str, Any] = Field(default_factory=dict)
    materialization: SourceMaterializationRequest = Field(
        default_factory=SourceMaterializationRequest
    )
    auth: SourceAuth = Field(default_factory=SourceAuth)
    policy: SourcePolicy = Field(default_factory=SourcePolicy)
    family_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuthRequirements(BaseModel):
    required: bool = False
    supported: list[str] = Field(default_factory=list)
    message: str | None = None


class SourceFileRecord(BaseModel):
    path: str
    size_bytes: int | None = None
    digest: str | None = None


class ResolvedSource(BaseModel):
    provider: str
    locator: dict[str, Any] = Field(default_factory=dict)
    pinned_ref: str | None = None
    access_state: str = "unknown"
    license: str | None = None
    remote_code_required: bool = False
    auth_requirements: AuthRequirements = Field(default_factory=AuthRequirements)
    files: list[SourceFileRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProvenanceRecord(BaseModel):
    provider: str
    locator: dict[str, Any] = Field(default_factory=dict)
    resolved_ref: str | None = None
    license: str | None = None
    access_state: str = "unknown"
    remote_code_required: bool = False
    remote_code_approved: bool = False
    blob_digests: dict[str, str] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderInspectionResult(BaseModel):
    bytes_total: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FamilyInspectionResult(BaseModel):
    family: str
    variant: str | None = None
    tasks: list[str] = Field(default_factory=list)
    scheduler_class: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceInspectionResult(BaseModel):
    resolved_source: ResolvedSource
    provider_inspection: ProviderInspectionResult
    provenance: ProvenanceRecord
    family_inspection: FamilyInspectionResult | None = None


class SourceRegistrationRecord(BaseModel):
    source_id: str
    source: SourceRef
    resolved_source: ResolvedSource
    provenance: ProvenanceRecord
    family_hint: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
