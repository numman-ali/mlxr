from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx
from mlxr.core.runtime import (
    ExecutionProfile,
    ExecutionStage,
    PortableArtifact,
    RuntimeHome,
)
from mlxr.core.schemas import JobRequest, ModelRecord, RuntimeEvent, RuntimeEventKind

from .registry import default_runtime_registry
from .store import InputStore

WorkerMessage = dict[str, object]


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    telemetry_available: bool
    active_bytes: int | None = None
    peak_bytes: int | None = None
    cache_bytes: int | None = None


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
        input_store = InputStore(runtime_home)
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
        load_memory_before = _begin_memory_measurement()
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
            phase="loading_model",
            memory=_memory_payload(load_memory_before, _memory_snapshot()),
        )

        resolved_inputs = _resolve_inputs(request.inputs, input_store)
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
            stage_memory_before = _begin_memory_measurement()
            result = family.run_stage(
                loaded,
                ExecutionStage(
                    stage_id=stage_id,
                    inputs=request.inputs,
                    params={
                        "task": request.task,
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
                        "resolved_inputs": resolved_inputs,
                        "width": request.params.get("width"),
                        "height": request.params.get("height"),
                        "num_frames": request.params.get("num_frames"),
                        "fps": request.params.get("fps"),
                        "seed": request.params.get("seed"),
                        "family_extensions": _family_extensions(
                            model.family, request.extensions
                        ),
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
                phase=phase,
                memory=_memory_payload(stage_memory_before, _memory_snapshot()),
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
        unload_started = time.perf_counter()
        unload_memory_before = _begin_memory_measurement()
        family.unload(loaded)
        emit_metrics_event(
            event_queue,
            job_id=job_id,
            stage_id="unload_model",
            duration_ms=_duration_ms(unload_started),
            phase="finalizing",
            memory=_memory_payload(unload_memory_before, _memory_snapshot()),
            metrics={"status": "unloaded"},
        )
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


def _family_extensions(family: str, extensions: dict[str, object]) -> dict[str, object]:
    raw_family_extensions = extensions.get(family)
    if raw_family_extensions is None:
        return {}
    if not isinstance(raw_family_extensions, dict):
        raise ValueError(f"Job extensions.{family} must be an object when provided")
    return dict(raw_family_extensions)


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
    phase: str,
    memory: dict[str, object],
    metrics: dict[str, object] | None = None,
) -> None:
    payload = {"stage_id": stage_id, "duration_ms": round(duration_ms, 3)}
    payload["memory"] = memory
    if metrics:
        payload["metrics"] = metrics
    emit_simple_event(
        event_queue,
        job_id=job_id,
        kind=RuntimeEventKind.JOB_METRICS,
        phase=phase,
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


def _resolve_inputs(
    inputs: dict[str, object], input_store: InputStore
) -> dict[str, object]:
    resolved_images: list[dict[str, object]] = []
    resolved_audio: dict[str, object] | None = None
    raw_images = inputs.get("images")
    if raw_images is not None:
        if not isinstance(raw_images, list):
            raise ValueError("LTX image conditioning inputs must be a list")
        for item in raw_images:
            if not isinstance(item, dict):
                raise ValueError("LTX image conditioning entries must be objects")
            handle_id = item.get("input_handle")
            if not isinstance(handle_id, str) or not handle_id:
                raise ValueError(
                    "LTX image conditioning requires non-empty input_handle values"
                )
            record = input_store.get(handle_id)
            if record is None:
                raise ValueError(f"Unknown input handle '{handle_id}'")
            payload_path = input_store.payload_path(record)
            resolved_images.append(
                {
                    "input_handle": handle_id,
                    "payload_path": str(payload_path),
                    "media_type": record.media_type,
                    "filename": record.filename,
                    "frame_index": item.get("frame_index", 0),
                    "strength": item.get("strength", 1.0),
                }
            )

    raw_audio = inputs.get("audio")
    if raw_audio is not None:
        if not isinstance(raw_audio, dict):
            raise ValueError("LTX audio conditioning input must be an object")
        handle_id = raw_audio.get("input_handle")
        if not isinstance(handle_id, str) or not handle_id:
            raise ValueError("LTX audio conditioning requires a non-empty input_handle")
        record = input_store.get(handle_id)
        if record is None:
            raise ValueError(f"Unknown input handle '{handle_id}'")
        payload_path = input_store.payload_path(record)
        resolved_audio = {
            "input_handle": handle_id,
            "payload_path": str(payload_path),
            "media_type": record.media_type,
            "filename": record.filename,
            "start_time_seconds": raw_audio.get("start_time_seconds", 0.0),
            "max_duration_seconds": raw_audio.get("max_duration_seconds"),
        }

    return {"images": resolved_images, "audio": resolved_audio}


def _begin_memory_measurement() -> MemorySnapshot:
    try:
        mx.reset_peak_memory()
    except Exception:
        return MemorySnapshot(telemetry_available=False)
    return _memory_snapshot()


def _memory_snapshot() -> MemorySnapshot:
    try:
        return MemorySnapshot(
            telemetry_available=True,
            active_bytes=int(mx.get_active_memory()),
            peak_bytes=int(mx.get_peak_memory()),
            cache_bytes=int(mx.get_cache_memory()),
        )
    except Exception:
        return MemorySnapshot(telemetry_available=False)


def _memory_payload(before: MemorySnapshot, after: MemorySnapshot) -> dict[str, object]:
    active_delta_bytes: int | None = None
    if before.active_bytes is not None and after.active_bytes is not None:
        active_delta_bytes = after.active_bytes - before.active_bytes
    return {
        "telemetry_available": before.telemetry_available and after.telemetry_available,
        "active_bytes": after.active_bytes,
        "peak_bytes": after.peak_bytes,
        "cache_bytes": after.cache_bytes,
        "active_delta_bytes": active_delta_bytes,
    }
