from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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

from .logging import get_control_plane_logger
from .state import RuntimeState


def create_app(state: RuntimeState | None = None) -> FastAPI:
    runtime = state or RuntimeState()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger = get_control_plane_logger()
        logger.info(
            "Control-plane app startup transport_default=uds runtime_home=%s",
            runtime.runtime_home.root,
        )
        try:
            yield
        finally:
            logger.info(
                "Control-plane app shutdown runtime_home=%s", runtime.runtime_home.root
            )

    app = FastAPI(
        title="MLXR Control Plane",
        description="Provider and family-aware local runtime scaffold for MLXR",
        version="0.1.0",
        lifespan=lifespan,
    )
    logger = get_control_plane_logger()

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
            result = runtime.catalog.inspect_source(source_ref)
            logger.info(
                "Source inspected provider=%s family_hint=%s",
                source_ref.provider,
                source_ref.family_hint,
            )
            return result
        except CatalogNotFoundError as exc:
            logger.warning(
                "Source inspect failed not_found provider=%s error=%s",
                source_ref.provider,
                exc,
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (CatalogValidationError, FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Source inspect failed validation provider=%s error=%s",
                source_ref.provider,
                exc,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/sources/register", response_model=SourceRegistrationRecord)
    def register_source(source_ref: SourceRef) -> SourceRegistrationRecord:
        try:
            result = runtime.catalog.register_source(source_ref)
            logger.info(
                "Source registered source_id=%s provider=%s family_hint=%s",
                result.source_id,
                result.source.provider,
                result.family_hint,
            )
            return result
        except CatalogNotFoundError as exc:
            logger.warning(
                "Source register failed not_found provider=%s error=%s",
                source_ref.provider,
                exc,
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (CatalogValidationError, FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Source register failed validation provider=%s error=%s",
                source_ref.provider,
                exc,
            )
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
            result = runtime.catalog.convert_artifact(request)
            logger.info(
                "Artifact converted source_id=%s model_id=%s artifact_digest=%s family=%s",
                request.source_id,
                result.model.model_id,
                result.artifact.artifact_digest,
                result.model.family,
            )
            logger.info(
                "Model registered model_id=%s artifact_digest=%s",
                result.model.model_id,
                result.artifact.artifact_digest,
            )
            return result
        except CatalogNotFoundError as exc:
            logger.warning(
                "Artifact convert failed not_found source_id=%s error=%s",
                request.source_id,
                exc,
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CatalogConflictError as exc:
            logger.warning(
                "Artifact convert failed conflict model_id=%s error=%s",
                request.model_id,
                exc,
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (CatalogValidationError, FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Artifact convert failed validation model_id=%s error=%s",
                request.model_id,
                exc,
            )
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
