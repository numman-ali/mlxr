from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from mlxr.core.schemas import (
    ArtifactConversionRequest,
    ArtifactConversionResult,
    ArtifactConversionTimingsMs,
    CapabilityDescriptor,
    FamilyInspectionResult,
    InstalledModelDetails,
    ModelInstallResult,
    ModelRecord,
    ModelRemoveResult,
    PortableArtifactRecord,
    ProviderInspectionResult,
    SourceInspectionResult,
    SourceInspectionTimingsMs,
    SourceRef,
    SourceRegistrationRecord,
    SupportedModelDescriptor,
    SupportedModelPreview,
    SupportedModelSourcePreview,
)

from .contracts import (
    ArtifactPayloadItem,
    ConversionPlan,
    ConversionSource,
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
from .supported_models import SUPPORTED_MODEL_RECIPES, SupportedModelRecipe


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
        started_at = time.perf_counter()
        provider = self._provider(source_ref.provider)
        resolve_started_at = time.perf_counter()
        resolved = provider.resolve(source_ref)
        resolve_ms = _elapsed_ms(resolve_started_at)
        provider_inspect_started_at = time.perf_counter()
        inspection = provider.inspect(resolved)
        provider_inspect_ms = _elapsed_ms(provider_inspect_started_at)
        provenance_started_at = time.perf_counter()
        provenance = provider.provenance(resolved)
        provenance_ms = _elapsed_ms(provenance_started_at)
        family_inspection = None
        family_inspect_ms: float | None = None
        if source_ref.family_hint:
            family = self._family(source_ref.family_hint)
            family_inspect_started_at = time.perf_counter()
            family_result = family.inspect_source(resolved)
            family_inspect_ms = _elapsed_ms(family_inspect_started_at)
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
            timings_ms=SourceInspectionTimingsMs(
                resolve_ms=resolve_ms,
                provider_inspect_ms=provider_inspect_ms,
                family_inspect_ms=family_inspect_ms,
                provenance_ms=provenance_ms,
                total_ms=_elapsed_ms(started_at),
            ),
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
        self,
        request: ArtifactConversionRequest,
        *,
        phase_callback: Callable[[str], None] | None = None,
    ) -> ArtifactConversionResult:
        started_at = time.perf_counter()
        family_id, source_records = self._resolve_conversion_sources(request)
        family = self._family(family_id)
        conversion_sources: dict[str, ConversionSource] = {}
        fetch_ms_by_role: dict[str, float] = {}
        fetch_total_started_at = time.perf_counter()
        if phase_callback is not None:
            phase_callback("downloading")
        for role, source_record in source_records.items():
            provider = self._provider(source_record.source.provider)
            fetch_policy = family.fetch_policy_for_conversion(
                role, source_record.resolved_source
            )
            fetch_started_at = time.perf_counter()
            materialization = provider.fetch(
                source_record.resolved_source, fetch_policy
            )
            fetch_ms_by_role[role] = _elapsed_ms(fetch_started_at)
            conversion_sources[role] = ConversionSource(
                role=role,
                source_id=source_record.source_id,
                source=source_record.source,
                materialization=materialization,
            )
        fetch_total_ms = _elapsed_ms(fetch_total_started_at)
        if phase_callback is not None:
            phase_callback("converting")
        family_convert_started_at = time.perf_counter()
        artifact = family.convert(
            conversion_sources,
            ConversionPlan(
                model_id=request.model_id,
                precision=request.precision,
                target_format=request.target_format,
                options=request.options,
            ),
        )
        family_convert_ms = _elapsed_ms(family_convert_started_at)
        if phase_callback is not None:
            phase_callback("registering")
        persist_started_at = time.perf_counter()
        persisted_artifact = self._persist_artifact(artifact)
        persist_ms = _elapsed_ms(persist_started_at)

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

        primary_source_record = self._primary_source_record(source_records)
        model_record = ModelRecord(
            model_id=request.model_id,
            family=family_id,
            source=primary_source_record.source,
            artifact=persisted_artifact,
            loaded=existing_model.loaded if existing_model is not None else False,
            capability=persisted_artifact.capability,
        )
        persisted_model = self.models.save(model_record)
        return ArtifactConversionResult(
            artifact=persisted_artifact,
            model=persisted_model,
            timings_ms=ArtifactConversionTimingsMs(
                fetch_ms_by_role=fetch_ms_by_role,
                fetch_total_ms=fetch_total_ms,
                family_convert_ms=family_convert_ms,
                persist_ms=persist_ms,
                total_ms=_elapsed_ms(started_at),
            ),
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

    def list_supported_models(self) -> list[SupportedModelDescriptor]:
        installed_model_ids = {record.model_id for record in self.models.list()}
        return [
            recipe.to_descriptor(installed=recipe.model_id in installed_model_ids)
            for recipe in _available_supported_model_recipes(self.registry)
        ]

    def preview_supported_model(self, model_id: str) -> SupportedModelPreview:
        recipe = _supported_model_recipe(self.registry, model_id)
        supported_model = recipe.to_descriptor(
            installed=self.models.get(model_id) is not None
        )
        previews: list[SupportedModelSourcePreview] = []
        total_source_bytes = 0
        total_known = True
        auth_required = False
        auth_messages: list[str] = []

        for role, source_ref in _recipe_source_refs(recipe):
            provider = self._provider(source_ref.provider)
            try:
                inspection = self.inspect_source(source_ref)
            except PermissionError:
                auth = provider.auth_requirements(source_ref)
                previews.append(
                    SupportedModelSourcePreview(
                        role=role,
                        provider=source_ref.provider,
                        locator=source_ref.locator,
                        resolved_ref=source_ref.locator.get("revision"),
                        access_state=recipe.access_state,
                        license=recipe.license,
                        auth_requirements=auth,
                        remote_code_required=False,
                        remote_code_approved=source_ref.policy.allow_remote_code,
                    )
                )
                auth_required = auth_required or auth.required
                if auth.message:
                    auth_messages.append(auth.message)
                total_known = False
                continue

            bytes_total = inspection.provider_inspection.bytes_total
            if bytes_total is None:
                total_known = False
            else:
                total_source_bytes += bytes_total
            auth = inspection.resolved_source.auth_requirements
            previews.append(
                SupportedModelSourcePreview(
                    role=role,
                    provider=inspection.resolved_source.provider,
                    locator=inspection.resolved_source.locator,
                    resolved_ref=inspection.provenance.resolved_ref,
                    access_state=inspection.resolved_source.access_state,
                    license=inspection.resolved_source.license,
                    bytes_total=bytes_total,
                    auth_requirements=auth,
                    remote_code_required=inspection.resolved_source.remote_code_required,
                    remote_code_approved=bool(
                        inspection.resolved_source.metadata.get(
                            "remote_code_approved", False
                        )
                    ),
                )
            )
            auth_required = auth_required or auth.required
            if auth.message:
                auth_messages.append(auth.message)

        return SupportedModelPreview(
            supported_model=supported_model,
            sources=previews,
            total_source_bytes=total_source_bytes if total_known else None,
            auth_required=auth_required,
            auth_message=next((message for message in auth_messages if message), None),
        )

    def install_supported_model(
        self,
        model_id: str,
        *,
        phase_callback: Callable[[str], None] | None = None,
    ) -> ModelInstallResult:
        recipe = _supported_model_recipe(self.registry, model_id)
        existing_model = self.models.get(model_id)
        if existing_model is not None and existing_model.artifact is not None:
            return ModelInstallResult(
                status="already_installed",
                model=existing_model,
                supported_model=recipe.to_descriptor(installed=True),
            )

        if phase_callback is not None:
            phase_callback("resolving")
        registered_source_ids: dict[str, str] = {}
        if recipe.source_ref is not None:
            registered_source = self.register_source(recipe.source_ref)
            registered_source_ids["bundle"] = registered_source.source_id
        elif recipe.source_bindings is not None:
            for role, source_ref in recipe.source_bindings.items():
                registered_source = self.register_source(source_ref)
                registered_source_ids[role] = registered_source.source_id
        else:
            raise CatalogValidationError(
                f"Supported model '{model_id}' has no install recipe"
            )

        conversion_result = self.convert_artifact(
            recipe.to_conversion_request(registered_source_ids=registered_source_ids),
            phase_callback=phase_callback,
        )
        return ModelInstallResult(
            status="installed",
            model=conversion_result.model,
            supported_model=recipe.to_descriptor(installed=True),
        )

    def get_model_details(self, model_id: str) -> InstalledModelDetails:
        model = self.get_model(model_id)
        artifact = model.artifact
        if artifact is None:
            raise CatalogValidationError(
                f"Model '{model_id}' has no portable artifact to inspect"
            )
        managed_size_bytes = self._artifact_size_bytes(artifact)
        supported_model = None
        try:
            supported_model = _supported_model_recipe(
                self.registry, model_id
            ).to_descriptor(installed=True)
        except CatalogNotFoundError:
            supported_model = None
        return InstalledModelDetails(
            model=model,
            supported_model=supported_model,
            managed_storage_key=artifact.storage_key,
            managed_size_bytes=managed_size_bytes,
            referenced_source_ids=sorted(self._model_source_ids(model)),
        )

    def remove_model(self, model_id: str) -> ModelRemoveResult:
        model = self.get_model(model_id)
        if model.loaded:
            raise CatalogConflictError(
                f"Model '{model_id}' is currently loaded and cannot be removed"
            )

        referenced_source_ids = self._model_source_ids(model)
        remaining_source_ids = {
            source_id
            for record in self.models.list()
            if record.model_id != model_id
            for source_id in self._model_source_ids(record)
        }
        removed_source_ids = sorted(referenced_source_ids - remaining_source_ids)

        artifact_digest = None
        removed_storage_key = None
        if model.artifact is not None:
            artifact_digest = model.artifact.artifact_digest
            removed_storage_key = model.artifact.storage_key
            self.artifacts.delete(model.artifact.artifact_digest)

        self.models.delete(model_id)
        for source_id in removed_source_ids:
            self.sources.delete(source_id)

        return ModelRemoveResult(
            status="removed",
            model_id=model_id,
            artifact_digest=artifact_digest,
            removed_source_ids=removed_source_ids,
            removed_storage_key=removed_storage_key,
        )

    def list_capabilities(self) -> list[CapabilityDescriptor]:
        return [
            record.capability
            for record in self.models.list()
            if record.capability is not None
        ]

    def _persist_artifact(self, artifact: PortableArtifact) -> PortableArtifactRecord:
        artifact_root = self.runtime_home.artifact_dir(
            artifact.record.family,
            artifact.record.model_id,
            artifact.record.artifact_digest,
        )
        self._materialize_artifact_payload(artifact_root, artifact.payload_items)
        record = artifact.record.model_copy(
            update={
                "storage_key": self.runtime_home.artifact_storage_key(
                    artifact.record.family,
                    artifact.record.model_id,
                    artifact.record.artifact_digest,
                ),
                "components": [
                    component.model_copy(
                        update={
                            "storage_key": self._component_storage_key(
                                artifact.record.family,
                                artifact.record.model_id,
                                artifact.record.artifact_digest,
                                component.relative_path,
                            )
                        }
                    )
                    for component in artifact.record.components
                ],
            }
        )
        artifact.storage_path = artifact_root
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

    def _resolve_conversion_sources(
        self, request: ArtifactConversionRequest
    ) -> tuple[str, dict[str, SourceRegistrationRecord]]:
        if request.source_id is not None:
            source_record = self.get_source(request.source_id)
            family_id = request.family or source_record.family_hint
            if not family_id:
                raise CatalogValidationError(
                    "Artifact conversion requires request.family or a registered source family_hint"
                )
            return family_id, {"bundle": source_record}

        if request.source_bindings is None or request.family is None:
            raise CatalogValidationError(
                "Artifact conversion with source_bindings requires request.family"
            )

        source_records: dict[str, SourceRegistrationRecord] = {}
        for role, source_id in request.source_bindings.items():
            source_record = self.get_source(source_id)
            if (
                source_record.family_hint is not None
                and source_record.family_hint != request.family
            ):
                raise CatalogValidationError(
                    f"Source '{source_id}' is registered for family '{source_record.family_hint}', not '{request.family}'"
                )
            source_records[role] = source_record
        return request.family, source_records

    def _primary_source_record(
        self, source_records: dict[str, SourceRegistrationRecord]
    ) -> SourceRegistrationRecord:
        if "checkpoint" in source_records:
            return source_records["checkpoint"]
        if "bundle" in source_records:
            return source_records["bundle"]
        first_role = sorted(source_records)[0]
        return source_records[first_role]

    def _component_storage_key(
        self,
        family: str,
        model_id: str,
        artifact_digest: str,
        relative_path: str,
    ) -> str:
        base_key = self.runtime_home.artifact_storage_key(
            family, model_id, artifact_digest
        )
        return f"{base_key}/{relative_path}"

    def _materialize_artifact_payload(
        self, artifact_root: Path, payload_items: tuple[ArtifactPayloadItem, ...]
    ) -> None:
        seen_paths: set[str] = set()
        for item in payload_items:
            relative_path = self._validated_relative_payload_path(item.relative_path)
            relative_key = relative_path.as_posix()
            if relative_key in seen_paths:
                raise CatalogValidationError(
                    f"Artifact payload path '{relative_key}' was staged more than once"
                )
            seen_paths.add(relative_key)
            destination_path = artifact_root / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source_path, destination_path)

    def _validated_relative_payload_path(self, relative_path: Path) -> Path:
        if relative_path.is_absolute() or any(
            part == ".." for part in relative_path.parts
        ):
            raise CatalogValidationError(
                f"Artifact payload path '{relative_path}' must be relative and stay within the artifact root"
            )
        return relative_path

    def _artifact_size_bytes(self, artifact: PortableArtifactRecord) -> int | None:
        artifact_root = self.runtime_home.artifact_dir(
            artifact.family, artifact.model_id, artifact.artifact_digest
        )
        if not artifact_root.exists():
            return None
        return sum(
            path.stat().st_size for path in artifact_root.rglob("*") if path.is_file()
        )

    def _model_source_ids(self, model: ModelRecord) -> set[str]:
        artifact = model.artifact
        if artifact is None:
            return set()
        return {
            component.source_id
            for component in artifact.components
            if component.source_id.strip()
        }


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def _recipe_source_refs(
    recipe: SupportedModelRecipe,
) -> tuple[tuple[str, SourceRef], ...]:
    if recipe.source_ref is not None:
        return (("bundle", recipe.source_ref),)
    if recipe.source_bindings is not None:
        return tuple(sorted(recipe.source_bindings.items()))
    raise CatalogValidationError(
        f"Supported model '{recipe.model_id}' has no source configuration"
    )


def _available_supported_model_recipes(
    registry: RuntimeRegistry,
) -> tuple[SupportedModelRecipe, ...]:
    available: list[SupportedModelRecipe] = []
    for recipe in SUPPORTED_MODEL_RECIPES:
        if not registry.has_family(recipe.family):
            continue
        if recipe.source_ref is not None:
            if not registry.has_provider(recipe.source_ref.provider):
                continue
            available.append(recipe)
            continue
        if recipe.source_bindings is not None and all(
            registry.has_provider(source_ref.provider)
            for source_ref in recipe.source_bindings.values()
        ):
            available.append(recipe)
    return tuple(available)


def _supported_model_recipe(
    registry: RuntimeRegistry, model_id: str
) -> SupportedModelRecipe:
    for recipe in _available_supported_model_recipes(registry):
        if recipe.model_id == model_id:
            return recipe
    raise CatalogNotFoundError(f"Unknown supported model '{model_id}'")
