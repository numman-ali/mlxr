from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Protocol

from mlx_runtime_core import (
    ExecutionProfile,
    ExecutionStage,
    PortableArtifact,
    RuntimeHome,
)
from mlx_runtime_schemas import JobRequest, ModelRecord, RuntimeEvent, RuntimeEventKind

from .registry import default_runtime_registry

WorkerMessage = dict[str, object]


class MessageQueue(Protocol):
    def put(self, item: WorkerMessage) -> None: ...

    def get_nowait(self) -> WorkerMessage: ...


def run_job_worker(
    *,
    job_id: str,
    request_data: dict[str, object],
    model_data: dict[str, object],
    runtime_home_root: str,
    event_queue: MessageQueue,
    command_queue: MessageQueue,
) -> None:
    request = JobRequest.model_validate(request_data)
    model = ModelRecord.model_validate(model_data)
    runtime_home = RuntimeHome(root=Path(runtime_home_root))
    try:
        registry = default_runtime_registry()
        family = registry.get_family(model.family)
        if model.artifact is None:
            raise ValueError(f"Model '{model.model_id}' has no portable artifact")

        emit_phase_event(
            event_queue,
            job_id=job_id,
            phase="loading_model",
            stage_id="load_model",
            kind=RuntimeEventKind.JOB_PHASE_CHANGED,
        )
        load_started = time.perf_counter()
        artifact = PortableArtifact(
            record=model.artifact,
            storage_path=runtime_home.artifact_dir(
                model.artifact.family,
                model.artifact.model_id,
                model.artifact.artifact_digest,
            ),
        )
        profile = ExecutionProfile(
            task=request.task,
            profile=_profile_for_request(model, request),
        )
        loaded = family.load(artifact, profile)
        emit_metrics_event(
            event_queue,
            job_id=job_id,
            stage_id="load_model",
            duration_ms=_duration_ms(load_started),
        )

        stage_ids = (
            "prompt_encode",
            "condition_inputs",
            "generate",
            "encode_output",
        )
        total_stages = len(stage_ids)
        created_artifacts = 0
        artifact_id = f"out_{job_id}"
        artifact_format = request.output.artifact_format
        if artifact_format is None:
            raise ValueError("Job output requires artifact_format before execution")
        for index, stage_id in enumerate(stage_ids, start=1):
            if cancellation_requested(command_queue):
                emit_simple_event(
                    event_queue,
                    job_id=job_id,
                    kind=RuntimeEventKind.JOB_CANCELLED,
                    phase="cancelled",
                    data={"stage_id": stage_id},
                )
                family.unload(loaded)
                event_queue.put({"type": "worker_exit"})
                return

            phase = "streaming_output" if stage_id == "encode_output" else "running"
            if stage_id in {"prompt_encode", "condition_inputs"}:
                phase = "preparing"
            emit_phase_event(
                event_queue,
                job_id=job_id,
                phase=phase,
                stage_id=stage_id,
                kind=RuntimeEventKind.JOB_PHASE_CHANGED,
            )

            stage_started = time.perf_counter()
            result = family.run_stage(
                loaded,
                ExecutionStage(
                    stage_id=stage_id,
                    inputs=request.inputs,
                    params={
                        "job_id": job_id,
                        "artifact_format": artifact_format,
                        "artifact_id": artifact_id,
                        "output_dir": str(
                            runtime_home.output_artifact_dir(job_id, artifact_id)
                        ),
                        "storage_key": runtime_home.output_artifact_storage_key(
                            job_id,
                            artifact_id,
                            f"{artifact_id}.{artifact_format}",
                        ),
                        "num_frames": request.params.get("num_frames"),
                        "simulate_delay_seconds": request.extensions.get(
                            "simulate_delay_seconds", 0.05
                        ),
                    },
                ),
            )
            emit_metrics_event(
                event_queue,
                job_id=job_id,
                stage_id=stage_id,
                duration_ms=_duration_ms(stage_started),
                metrics=result.metrics,
            )
            emit_simple_event(
                event_queue,
                job_id=job_id,
                kind=RuntimeEventKind.JOB_PROGRESS,
                phase=phase,
                data={
                    "stage_id": stage_id,
                    "completed_stages": index,
                    "total_stages": total_stages,
                },
            )
            if result.artifacts:
                created_artifacts += len(result.artifacts)
                event_queue.put(
                    {
                        "type": "artifacts",
                        "job_id": job_id,
                        "artifacts": [
                            artifact.model_dump(mode="json")
                            for artifact in result.artifacts
                        ],
                    }
                )

        emit_phase_event(
            event_queue,
            job_id=job_id,
            phase="finalizing",
            stage_id="finalize",
            kind=RuntimeEventKind.JOB_PHASE_CHANGED,
        )
        family.unload(loaded)
        emit_simple_event(
            event_queue,
            job_id=job_id,
            kind=RuntimeEventKind.JOB_COMPLETED,
            phase="completed",
            data={"artifact_count": created_artifacts},
        )
    except Exception as exc:
        event_queue.put(
            {
                "type": "event",
                "event": RuntimeEvent(
                    job_id=job_id,
                    kind=RuntimeEventKind.JOB_FAILED,
                    phase="failed",
                    data={"error": str(exc)},
                ).model_dump(mode="json"),
            }
        )
    finally:
        event_queue.put({"type": "worker_exit"})


def cancellation_requested(command_queue: MessageQueue) -> bool:
    try:
        command = command_queue.get_nowait()
    except queue.Empty:
        return False
    return command.get("command") == "cancel"


def emit_phase_event(
    event_queue: MessageQueue,
    *,
    job_id: str,
    phase: str,
    stage_id: str,
    kind: RuntimeEventKind,
) -> None:
    emit_simple_event(
        event_queue,
        job_id=job_id,
        kind=kind,
        phase=phase,
        data={"stage_id": stage_id},
    )


def emit_metrics_event(
    event_queue: MessageQueue,
    *,
    job_id: str,
    stage_id: str,
    duration_ms: float,
    metrics: dict[str, object] | None = None,
) -> None:
    payload = {"stage_id": stage_id, "duration_ms": round(duration_ms, 3)}
    if metrics:
        payload["metrics"] = metrics
    emit_simple_event(
        event_queue,
        job_id=job_id,
        kind=RuntimeEventKind.JOB_METRICS,
        phase="running",
        data=payload,
    )


def emit_simple_event(
    event_queue: MessageQueue,
    *,
    job_id: str,
    kind: RuntimeEventKind,
    phase: str,
    data: dict[str, object],
) -> None:
    event_queue.put(
        {
            "type": "event",
            "event": RuntimeEvent(
                job_id=job_id,
                kind=kind,
                phase=phase,
                data=data,
            ).model_dump(mode="json"),
        }
    )


def _profile_for_request(model: ModelRecord, request: JobRequest) -> str:
    capability = model.capability or (
        model.artifact.capability if model.artifact else None
    )
    if capability is None:
        return "default"
    profiles = capability.profiles_by_task.get(request.task)
    if profiles:
        return profiles[0]
    return "default"


def _duration_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
