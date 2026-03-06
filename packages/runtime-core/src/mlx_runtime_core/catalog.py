from __future__ import annotations

from datetime import datetime, timezone

from mlx_runtime_schemas import (
    ArtifactConversionRequest,
    ArtifactConversionResult,
    CapabilityDescriptor,
    FamilyInspectionResult,
    ModelRecord,
    PortableArtifactRecord,
    ProviderInspectionResult,
    SourceInspectionResult,
    SourceRef,
    SourceRegistrationRecord,
)

from .contracts import (
    ConversionPlan,
    ModelFamilyAdapter,
    PortableArtifact,
    SourceProviderAdapter,
)
from .manifests import (
    ArtifactManifestStore,
    ModelManifestStore,
    SourceManifestStore,
    source_id_for_ref,
)
from .registry import RuntimeRegistry
from .runtime_home import RuntimeHome


class RuntimeCatalogError(Exception):
    pass


class CatalogNotFoundError(RuntimeCatalogError):
    pass


class CatalogConflictError(RuntimeCatalogError):
    pass


class CatalogValidationError(RuntimeCatalogError):
    pass


class RuntimeCatalog:
    def __init__(self, registry: RuntimeRegistry, runtime_home: RuntimeHome) -> None:
        self.registry = registry
        self.runtime_home = runtime_home
        self.runtime_home.ensure_layout()
        self.sources = SourceManifestStore(runtime_home)
        self.artifacts = ArtifactManifestStore(runtime_home)
        self.models = ModelManifestStore(runtime_home)

    def inspect_source(self, source_ref: SourceRef) -> SourceInspectionResult:
        provider = self._provider(source_ref.provider)
        resolved = provider.resolve(source_ref)
        inspection = provider.inspect(resolved)
        provenance = provider.provenance(resolved)
        family_inspection = None
        if source_ref.family_hint:
            family = self._family(source_ref.family_hint)
            family_result = family.inspect_source(resolved)
            family_inspection = FamilyInspectionResult(
                family=family_result.family,
                variant=family_result.variant,
                tasks=list(family_result.tasks),
                scheduler_class=family_result.scheduler_class,
                metadata=family_result.metadata,
            )
        return SourceInspectionResult(
            resolved_source=resolved,
            provider_inspection=ProviderInspectionResult(
                bytes_total=inspection.bytes_total,
                metadata=inspection.metadata,
            ),
            provenance=provenance,
            family_inspection=family_inspection,
        )

    def register_source(self, source_ref: SourceRef) -> SourceRegistrationRecord:
        inspection = self.inspect_source(source_ref)
        source_id = source_id_for_ref(source_ref)
        existing = self.sources.get(source_id)
        now = datetime.now(timezone.utc)
        record = SourceRegistrationRecord(
            source_id=source_id,
            source=source_ref,
            resolved_source=inspection.resolved_source,
            provenance=inspection.provenance,
            family_hint=source_ref.family_hint,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            metadata={
                "provider_inspection": inspection.provider_inspection.model_dump(
                    mode="json"
                ),
                "family_inspection": (
                    inspection.family_inspection.model_dump(mode="json")
                    if inspection.family_inspection is not None
                    else None
                ),
            },
        )
        return self.sources.save(record)

    def list_sources(self) -> list[SourceRegistrationRecord]:
        return self.sources.list()

    def get_source(self, source_id: str) -> SourceRegistrationRecord:
        record = self.sources.get(source_id)
        if record is None:
            raise CatalogNotFoundError(f"Unknown source '{source_id}'")
        return record

    def convert_artifact(
        self, request: ArtifactConversionRequest
    ) -> ArtifactConversionResult:
        source_record = self.get_source(request.source_id)
        family_id = request.family or source_record.family_hint
        if not family_id:
            raise CatalogValidationError(
                "Artifact conversion requires request.family or a registered source family_hint"
            )

        provider = self._provider(source_record.source.provider)
        family = self._family(family_id)
        fetch_policy = family.fetch_policy_for_conversion(source_record.resolved_source)
        materialization = provider.fetch(source_record.resolved_source, fetch_policy)
        artifact = family.convert(
            materialization,
            ConversionPlan(
                model_id=request.model_id,
                precision=request.precision,
                target_format=request.target_format,
                options=request.options,
            ),
        )
        persisted_artifact = self._persist_artifact(artifact)

        existing_model = self.models.get(request.model_id)
        if existing_model is not None and existing_model.artifact is not None:
            if (
                existing_model.artifact.artifact_digest
                != persisted_artifact.artifact_digest
            ):
                raise CatalogConflictError(
                    f"Model '{request.model_id}' is already registered to artifact "
                    f"'{existing_model.artifact.artifact_digest}'"
                )

        model_record = ModelRecord(
            model_id=request.model_id,
            family=family_id,
            source=source_record.source,
            artifact=persisted_artifact,
            loaded=existing_model.loaded if existing_model is not None else False,
            capability=persisted_artifact.capability,
        )
        persisted_model = self.models.save(model_record)
        return ArtifactConversionResult(
            artifact=persisted_artifact, model=persisted_model
        )

    def list_artifacts(self) -> list[PortableArtifactRecord]:
        return self.artifacts.list()

    def get_artifact(self, artifact_digest: str) -> PortableArtifactRecord:
        record = self.artifacts.get(artifact_digest)
        if record is None:
            raise CatalogNotFoundError(f"Unknown artifact '{artifact_digest}'")
        return record

    def list_models(self) -> list[ModelRecord]:
        return self.models.list()

    def get_model(self, model_id: str) -> ModelRecord:
        record = self.models.get(model_id)
        if record is None:
            raise CatalogNotFoundError(f"Unknown model '{model_id}'")
        return record

    def list_capabilities(self) -> list[CapabilityDescriptor]:
        return [
            record.capability
            for record in self.models.list()
            if record.capability is not None
        ]

    def _persist_artifact(self, artifact: PortableArtifact) -> PortableArtifactRecord:
        record = artifact.record.model_copy(
            update={
                "storage_key": self.runtime_home.artifact_storage_key(
                    artifact.record.family,
                    artifact.record.model_id,
                    artifact.record.artifact_digest,
                )
            }
        )
        artifact.storage_path = self.runtime_home.artifact_manifest_path(
            record.family,
            record.model_id,
            record.artifact_digest,
        ).parent
        artifact.record = record
        return self.artifacts.save(record)

    def _provider(self, provider_id: str) -> SourceProviderAdapter:
        if not self.registry.has_provider(provider_id):
            raise CatalogNotFoundError(f"Unknown provider '{provider_id}'")
        return self.registry.get_provider(provider_id)

    def _family(self, family_id: str) -> ModelFamilyAdapter:
        if not self.registry.has_family(family_id):
            raise CatalogNotFoundError(f"Unknown family '{family_id}'")
        return self.registry.get_family(family_id)
