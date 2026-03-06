from __future__ import annotations

from fastapi import FastAPI, HTTPException

from mlx_runtime_core import (
    CatalogConflictError,
    CatalogNotFoundError,
    CatalogValidationError,
)
from mlx_runtime_schemas import (
    ArtifactConversionRequest,
    ArtifactConversionResult,
    CapabilityDescriptor,
    ModelRecord,
    PortableArtifactRecord,
    SourceInspectionResult,
    SourceRef,
    SourceRegistrationRecord,
)

from .state import RuntimeState


def create_app(state: RuntimeState | None = None) -> FastAPI:
    runtime = state or RuntimeState()
    app = FastAPI(
        title="MLXR Control Plane",
        description="Provider and family-aware local runtime scaffold for MLXR",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "transport_default": "uds"}

    @app.get("/v1/providers")
    def list_providers() -> dict[str, list[str]]:
        return {"providers": sorted(runtime.registry.providers())}

    @app.get("/v1/families")
    def list_families() -> dict[str, list[str]]:
        return {"families": sorted(runtime.registry.families())}

    @app.post("/v1/sources/inspect", response_model=SourceInspectionResult)
    def inspect_source(source_ref: SourceRef) -> SourceInspectionResult:
        try:
            return runtime.catalog.inspect_source(source_ref)
        except CatalogNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (CatalogValidationError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/sources/register", response_model=SourceRegistrationRecord)
    def register_source(source_ref: SourceRef) -> SourceRegistrationRecord:
        try:
            return runtime.catalog.register_source(source_ref)
        except CatalogNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (CatalogValidationError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/sources", response_model=list[SourceRegistrationRecord])
    def list_sources() -> list[SourceRegistrationRecord]:
        return runtime.catalog.list_sources()

    @app.get("/v1/sources/{source_id}", response_model=SourceRegistrationRecord)
    def get_source(source_id: str) -> SourceRegistrationRecord:
        try:
            return runtime.catalog.get_source(source_id)
        except CatalogNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/artifacts/convert", response_model=ArtifactConversionResult)
    def convert_artifact(
        request: ArtifactConversionRequest,
    ) -> ArtifactConversionResult:
        try:
            return runtime.catalog.convert_artifact(request)
        except CatalogNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CatalogConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (CatalogValidationError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/artifacts", response_model=list[PortableArtifactRecord])
    def list_artifacts() -> list[PortableArtifactRecord]:
        return runtime.catalog.list_artifacts()

    @app.get("/v1/artifacts/{artifact_digest}", response_model=PortableArtifactRecord)
    def get_artifact(artifact_digest: str) -> PortableArtifactRecord:
        try:
            return runtime.catalog.get_artifact(artifact_digest)
        except CatalogNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/models", response_model=list[ModelRecord])
    def list_models() -> list[ModelRecord]:
        return runtime.catalog.list_models()

    @app.get("/v1/models/{model_id}", response_model=ModelRecord)
    def get_model(model_id: str) -> ModelRecord:
        try:
            return runtime.catalog.get_model(model_id)
        except CatalogNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/capabilities", response_model=list[CapabilityDescriptor])
    def list_capabilities() -> list[CapabilityDescriptor]:
        return runtime.catalog.list_capabilities()

    return app


app = create_app()
