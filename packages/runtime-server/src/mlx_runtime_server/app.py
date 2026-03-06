from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from mlx_runtime_schemas import JobRecord, JobRequest, JobState, ModelRecord, RuntimeEvent, RuntimeEventKind

from .state import RuntimeState


def create_app(state: RuntimeState | None = None) -> FastAPI:
    runtime = state or RuntimeState()
    app = FastAPI(
        title="MLX Runtime",
        description="Job-oriented local daemon scaffold for MLXR",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, list[dict[str, object]]]:
        caps = [
            record.capabilities.model_dump() for record in runtime.models.values() if record.capabilities is not None
        ]
        return {"capabilities": caps}

    @app.get("/v1/models")
    def list_models() -> dict[str, list[dict[str, object]]]:
        return {"models": [record.model_dump() for record in runtime.models.values()]}

    @app.get("/v1/models/{model_id}")
    def get_model(model_id: str) -> dict[str, object]:
        if model_id not in runtime.models:
            raise HTTPException(status_code=404, detail=f"Unknown model '{model_id}'")
        return runtime.models[model_id].model_dump()

    @app.post("/v1/models/register")
    def register_model(record: ModelRecord) -> dict[str, object]:
        runtime.models[record.model_id] = record
        return record.model_dump()

    @app.post("/v1/jobs")
    def submit_job(request: JobRequest) -> dict[str, object]:
        if request.model_id not in runtime.models:
            raise HTTPException(status_code=404, detail=f"Unknown model '{request.model_id}'")

        now = datetime.now(timezone.utc)
        job = JobRecord(
            job_id=str(uuid4()),
            request=request,
            state=JobState.ACCEPTED,
            created_at=now,
            updated_at=now,
        )
        runtime.jobs[job.job_id] = job
        runtime.events[job.job_id].append(
            RuntimeEvent(
                job_id=job.job_id,
                kind=RuntimeEventKind.JOB_ACCEPTED,
                phase="accepted",
                data={"task": request.task, "model_id": request.model_id},
            )
        )
        return job.model_dump()

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, object]:
        if job_id not in runtime.jobs:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
        return runtime.jobs[job_id].model_dump()

    @app.get("/v1/jobs/{job_id}/events")
    def get_job_events(job_id: str) -> dict[str, list[dict[str, object]]]:
        if job_id not in runtime.jobs:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
        return {"events": [event.model_dump() for event in runtime.events[job_id]]}

    return app


app = create_app()
