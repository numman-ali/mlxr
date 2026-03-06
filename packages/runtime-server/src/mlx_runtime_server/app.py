from __future__ import annotations

import base64
import binascii
import json
import queue
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from mlx_runtime_core import (
    CatalogConflictError,
    CatalogNotFoundError,
    CatalogValidationError,
)
from mlx_runtime_schemas import (
    ArtifactConversionRequest,
    ArtifactConversionResult,
    ArtifactExportRequest,
    ArtifactExportResult,
    CapabilityDescriptor,
    InputHandleRecord,
    InputImportRequest,
    JobRecord,
    JobRequest,
    JobSubmitResult,
    ModelRecord,
    OutputArtifactRecord,
    PortableArtifactRecord,
    RuntimeEvent,
    RuntimeEventKind,
    SourceInspectionResult,
    SourceRef,
    SourceRegistrationRecord,
)

from .jobs import (
    TERMINAL_STATES,
    JobConflictError,
    JobManagerError,
    JobNotFoundError,
    JobValidationError,
)
from .logging import get_control_plane_logger
from .security import enforce_mutating_request_policy
from .state import RuntimeState


def create_app(state: RuntimeState | None = None) -> FastAPI:
    runtime = state or RuntimeState()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runtime.settings.validate_startup()
        logger = get_control_plane_logger()
        logger.info(
            "Control-plane app startup transport_default=%s runtime_home=%s",
            "http" if runtime.settings.http_enabled else "uds",
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

    def guard_mutation(request: Request) -> None:
        try:
            enforce_mutating_request_policy(request, runtime.settings)
        except HTTPException as exc:
            logger.warning(
                "HTTP mutation rejected method=%s path=%s reason=%s",
                request.method,
                request.url.path,
                exc.detail,
            )
            raise

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "transport_default": "http" if runtime.settings.http_enabled else "uds",
        }

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
        except PermissionError as exc:
            logger.warning(
                "Source inspect failed permission provider=%s error=%s",
                source_ref.provider,
                exc,
            )
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (CatalogValidationError, FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Source inspect failed validation provider=%s error=%s",
                source_ref.provider,
                exc,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/sources/register", response_model=SourceRegistrationRecord)
    def register_source(
        source_ref: SourceRef, request: Request
    ) -> SourceRegistrationRecord:
        guard_mutation(request)
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
        except PermissionError as exc:
            logger.warning(
                "Source register failed permission provider=%s error=%s",
                source_ref.provider,
                exc,
            )
            raise HTTPException(status_code=403, detail=str(exc)) from exc
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
        conversion_request: ArtifactConversionRequest, request: Request
    ) -> ArtifactConversionResult:
        guard_mutation(request)
        source_selector = json.dumps(
            {
                "source_id": conversion_request.source_id,
                "source_bindings": conversion_request.source_bindings,
            },
            sort_keys=True,
        )
        try:
            result = runtime.catalog.convert_artifact(conversion_request)
            logger.info(
                "Artifact converted source_selector=%s model_id=%s artifact_digest=%s family=%s",
                source_selector,
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
                "Artifact convert failed not_found source_selector=%s error=%s",
                source_selector,
                exc,
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CatalogConflictError as exc:
            logger.warning(
                "Artifact convert failed conflict model_id=%s error=%s",
                conversion_request.model_id,
                exc,
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionError as exc:
            logger.warning(
                "Artifact convert failed permission model_id=%s error=%s",
                conversion_request.model_id,
                exc,
            )
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (CatalogValidationError, FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Artifact convert failed validation model_id=%s error=%s",
                conversion_request.model_id,
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

    @app.post("/v1/inputs/import", response_model=InputHandleRecord)
    def import_input(
        import_request: InputImportRequest, request: Request
    ) -> InputHandleRecord:
        guard_mutation(request)
        try:
            payload = base64.b64decode(import_request.content_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(
                status_code=400, detail="Invalid base64 input payload"
            ) from exc

        handle_id = f"inp_{uuid.uuid4().hex}"
        filename = import_request.filename or f"{handle_id}.bin"
        record = InputHandleRecord(
            handle_id=handle_id,
            media_type=import_request.media_type,
            role=import_request.role,
            filename=filename,
            size_bytes=len(payload),
            storage_key=runtime.runtime_home.input_storage_key(handle_id, filename),
            created_at=datetime.now(timezone.utc),
            metadata=import_request.metadata,
        )
        runtime.input_store.save(record, payload)
        logger.info(
            "Input imported handle_id=%s media_type=%s size_bytes=%s",
            record.handle_id,
            record.media_type,
            record.size_bytes,
        )
        return record

    @app.get("/v1/inputs", response_model=list[InputHandleRecord])
    def list_inputs() -> list[InputHandleRecord]:
        return runtime.input_store.list_records()

    @app.get("/v1/inputs/{handle_id}", response_model=InputHandleRecord)
    def get_input(handle_id: str) -> InputHandleRecord:
        record = runtime.input_store.get(handle_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown input handle '{handle_id}'"
            )
        return record

    @app.post("/v1/jobs", response_model=JobSubmitResult)
    def submit_job(job_request: JobRequest, request: Request) -> JobSubmitResult:
        guard_mutation(request)
        try:
            record = runtime.job_manager.submit(job_request)
            logger.info(
                "Job submitted job_id=%s model_id=%s task=%s",
                record.job_id,
                record.request.model_id,
                record.request.task,
            )
            return JobSubmitResult(job_id=record.job_id, record=record)
        except CatalogNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (JobValidationError, CatalogValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/jobs", response_model=list[JobRecord])
    def list_jobs() -> list[JobRecord]:
        return runtime.job_manager.list_jobs()

    @app.get("/v1/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        try:
            return runtime.job_manager.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobRecord)
    def cancel_job(job_id: str, request: Request) -> JobRecord:
        guard_mutation(request)
        try:
            record = runtime.job_manager.cancel(job_id)
            logger.info("Job cancellation requested job_id=%s", job_id)
            return record
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except JobManagerError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/events")
    def stream_job_events(job_id: str) -> StreamingResponse:
        try:
            runtime.job_manager.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        def generate() -> Iterator[str]:
            backlog, subscriber = runtime.job_manager.open_event_stream(job_id)
            seen = {event.model_dump_json() for event in backlog}
            for event in backlog:
                yield _sse_payload(event)

            try:
                while True:
                    try:
                        event = subscriber.get(timeout=0.5)
                    except queue.Empty:
                        latest = runtime.job_manager.get(job_id)
                        if latest.state in TERMINAL_STATES:
                            return
                        continue

                    event_key = event.model_dump_json()
                    if event_key in seen:
                        continue
                    seen.add(event_key)
                    yield _sse_payload(event)
                    if event.kind in {
                        RuntimeEventKind.JOB_COMPLETED,
                        RuntimeEventKind.JOB_FAILED,
                        RuntimeEventKind.JOB_CANCELLED,
                    }:
                        return
            finally:
                runtime.job_manager.unsubscribe(job_id, subscriber)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/v1/outputs/{artifact_id}", response_model=OutputArtifactRecord)
    def get_output(artifact_id: str) -> OutputArtifactRecord:
        try:
            return runtime.job_manager.get_output(artifact_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/outputs/{artifact_id}/download")
    def download_output(artifact_id: str) -> FileResponse:
        try:
            record = runtime.job_manager.get_output(artifact_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path=runtime.output_store.payload_path(record),
            media_type=record.media_type,
            filename=record.filename,
        )

    @app.post("/v1/outputs/{artifact_id}/export", response_model=ArtifactExportResult)
    def export_output(
        artifact_id: str,
        export_request: ArtifactExportRequest,
        request: Request,
    ) -> ArtifactExportResult:
        guard_mutation(request)
        try:
            result = runtime.job_manager.export_output(
                artifact_id=artifact_id,
                destination_path=export_request.destination_path,
                overwrite=export_request.overwrite,
            )
            logger.info(
                "Output exported artifact_id=%s destination=%s",
                artifact_id,
                result.destination_path,
            )
            return result
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except JobValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _sse_payload(event: RuntimeEvent) -> str:
    payload = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
    return f"event: {event.kind.value}\ndata: {payload}\n\n"


app = create_app()
