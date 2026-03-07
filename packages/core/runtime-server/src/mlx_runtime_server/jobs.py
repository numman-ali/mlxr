from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from queue import Queue
from typing import Protocol

from mlx_runtime_core import RuntimeCatalog
from mlx_runtime_schemas import (
    ArtifactExportResult,
    ArtifactHandle,
    CapabilityDescriptor,
    JobRecord,
    JobRequest,
    JobState,
    OutputArtifactRecord,
    OutputDestinationMode,
    RuntimeEvent,
    RuntimeEventKind,
)

from .settings import ServerSettings
from .store import InputStore, JobStore, OutputStore
from .worker import WorkerMessage, run_job_worker


class ManagedMessageQueue(Protocol):
    def put(self, item: WorkerMessage) -> None: ...

    def get(self) -> WorkerMessage: ...

    def close(self) -> None: ...


class ManagedProcess(Protocol):
    def start(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...


class ProcessContext(Protocol):
    def Queue(self) -> ManagedMessageQueue: ...

    def Process(self, target: object, kwargs: dict[str, object]) -> ManagedProcess: ...


TERMINAL_STATES = {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
HEAVY_JOB_SCHEDULER_CLASSES = {"media_video_dit"}


class JobManagerError(Exception):
    pass


class JobNotFoundError(JobManagerError):
    pass


class JobConflictError(JobManagerError):
    pass


class JobValidationError(JobManagerError):
    pass


@dataclass
class RunningJob:
    command_queue: ManagedMessageQueue
    event_queue: ManagedMessageQueue
    process: ManagedProcess
    listener: threading.Thread


class JobManager:
    def __init__(
        self,
        *,
        catalog: RuntimeCatalog,
        job_store: JobStore,
        input_store: InputStore,
        output_store: OutputStore,
        settings: ServerSettings,
    ) -> None:
        self.catalog = catalog
        self.job_store = job_store
        self.input_store = input_store
        self.output_store = output_store
        self.settings = settings
        self._running_jobs: dict[str, RunningJob] = {}
        self._subscribers: dict[str, list[Queue[RuntimeEvent]]] = {}
        self._lock = threading.RLock()

    def submit(self, request: JobRequest) -> JobRecord:
        prepared_request = self._prepared_request(request)
        model = self.catalog.get_model(prepared_request.model_id)
        capability = model.capability or (
            model.artifact.capability if model.artifact else None
        )
        if capability is None:
            raise JobValidationError(
                f"Model '{prepared_request.model_id}' has no capability descriptor"
            )
        self._validate_request(
            model_id=model.model_id, capability=capability, request=prepared_request
        )

        with self._lock:
            if (
                self._active_heavy_job_id() is not None
                and capability.scheduler_class in HEAVY_JOB_SCHEDULER_CLASSES
            ):
                raise JobConflictError("A heavy job is already running")

            job_id = f"job_{uuid.uuid4().hex}"
            now = datetime.now(timezone.utc)
            record = JobRecord(
                job_id=job_id,
                request=prepared_request,
                state=JobState.ACCEPTED,
                created_at=now,
                updated_at=now,
            )
            self.job_store.save(record)
            self._append_event(
                RuntimeEvent(
                    job_id=job_id,
                    kind=RuntimeEventKind.JOB_ACCEPTED,
                    phase=JobState.ACCEPTED.value,
                    data={
                        "model_id": prepared_request.model_id,
                        "task": prepared_request.task,
                    },
                )
            )

            context = _job_process_context()
            event_queue = context.Queue()
            command_queue = context.Queue()
            process = context.Process(
                target=run_job_worker,
                kwargs={
                    "job_id": job_id,
                    "request_data": prepared_request.model_dump(mode="json"),
                    "model_data": model.model_dump(mode="json"),
                    "runtime_home_root": str(self.job_store.runtime_home.root),
                    "event_queue": event_queue,
                    "command_queue": command_queue,
                },
            )
            listener = threading.Thread(
                target=self._listen_to_worker,
                args=(job_id, event_queue, command_queue, process),
                daemon=True,
            )
            process.start()
            listener.start()
            self._running_jobs[job_id] = RunningJob(
                command_queue=command_queue,
                event_queue=event_queue,
                process=process,
                listener=listener,
            )
            return record

    def get(self, job_id: str) -> JobRecord:
        record = self.job_store.get(job_id)
        if record is None:
            raise JobNotFoundError(f"Unknown job '{job_id}'")
        return record

    def list_jobs(self) -> list[JobRecord]:
        return self.job_store.list_records()

    def cancel(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record.state in TERMINAL_STATES:
            return record
        running_job = self._running_jobs.get(job_id)
        if running_job is None:
            raise JobConflictError(f"Job '{job_id}' is not running")
        running_job.command_queue.put({"command": "cancel"})
        return record

    def list_events(self, job_id: str) -> list[RuntimeEvent]:
        self.get(job_id)
        return self.job_store.list_events(job_id)

    def subscribe(self, job_id: str) -> Queue[RuntimeEvent]:
        self.get(job_id)
        subscriber: Queue[RuntimeEvent] = Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(subscriber)
        return subscriber

    def open_event_stream(
        self, job_id: str
    ) -> tuple[list[RuntimeEvent], Queue[RuntimeEvent]]:
        self.get(job_id)
        subscriber: Queue[RuntimeEvent] = Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(subscriber)
            backlog = self.job_store.list_events(job_id)
        return backlog, subscriber

    def unsubscribe(self, job_id: str, subscriber: Queue[RuntimeEvent]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(job_id)
            if subscribers is None:
                return
            self._subscribers[job_id] = [
                existing for existing in subscribers if existing is not subscriber
            ]

    def get_output(self, artifact_id: str) -> OutputArtifactRecord:
        record = self.output_store.get(artifact_id)
        if record is None:
            raise JobNotFoundError(f"Unknown output artifact '{artifact_id}'")
        return record

    def export_output(
        self, artifact_id: str, destination_path: str, overwrite: bool
    ) -> ArtifactExportResult:
        if self.settings.http_enabled:
            raise JobValidationError(
                "Trusted local export is only available when loopback HTTP is disabled"
            )
        record = self.get_output(artifact_id)
        source_path = self.output_store.payload_path(record)
        destination_file = Path(destination_path).expanduser()
        if destination_file.exists() and not overwrite:
            raise JobValidationError(
                f"Destination '{destination_file}' already exists; set overwrite=true to replace it"
            )
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_file)
        return ArtifactExportResult(
            artifact_id=artifact_id,
            destination_path=str(destination_file),
            size_bytes=record.size_bytes,
        )

    def _listen_to_worker(
        self,
        job_id: str,
        event_queue: ManagedMessageQueue,
        command_queue: ManagedMessageQueue,
        process: ManagedProcess,
    ) -> None:
        try:
            while True:
                message = event_queue.get()
                if not isinstance(message, dict):
                    continue
                message_type = message.get("type")
                if message_type == "event":
                    event = RuntimeEvent.model_validate(message["event"])
                    self._append_event(event)
                    self._update_job_from_event(event)
                elif message_type == "artifacts":
                    raw_artifacts = message.get("artifacts", [])
                    if isinstance(raw_artifacts, list):
                        self._store_output_artifacts(
                            job_id,
                            [
                                ArtifactHandle.model_validate(item)
                                for item in raw_artifacts
                            ],
                        )
                elif message_type == "worker_exit":
                    break
        finally:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            record = self.job_store.get(job_id)
            if record is not None and record.state not in TERMINAL_STATES:
                failed_record = record.model_copy(
                    update={
                        "state": JobState.FAILED,
                        "updated_at": datetime.now(timezone.utc),
                        "error": "Worker exited before reaching a terminal state",
                    }
                )
                self.job_store.save(failed_record)
            command_queue.close()
            event_queue.close()
            with self._lock:
                self._running_jobs.pop(job_id, None)
                self._subscribers.pop(job_id, None)

    def _append_event(self, event: RuntimeEvent) -> None:
        self.job_store.append_event(event.job_id, event)
        with self._lock:
            subscribers = list(self._subscribers.get(event.job_id, []))
        for subscriber in subscribers:
            subscriber.put_nowait(event)

    def _update_job_from_event(self, event: RuntimeEvent) -> None:
        record = self.job_store.get(event.job_id)
        if record is None:
            return
        next_record = record.model_copy(deep=True)
        next_record.updated_at = datetime.now(timezone.utc)
        if event.kind == RuntimeEventKind.JOB_PHASE_CHANGED and event.phase is not None:
            next_record.state = JobState(event.phase)
        elif event.kind == RuntimeEventKind.JOB_COMPLETED:
            next_record.state = JobState.COMPLETED
            next_record.error = None
        elif event.kind == RuntimeEventKind.JOB_FAILED:
            next_record.state = JobState.FAILED
            error = event.data.get("error")
            next_record.error = error if isinstance(error, str) else "Job failed"
        elif event.kind == RuntimeEventKind.JOB_CANCELLED:
            next_record.state = JobState.CANCELLED
            next_record.error = None
        self.job_store.save(next_record)

    def _store_output_artifacts(
        self, job_id: str, artifacts: list[ArtifactHandle]
    ) -> None:
        job = self.get(job_id)
        next_job = job.model_copy(deep=True)
        for artifact in artifacts:
            filename = artifact.metadata.get("filename")
            if not isinstance(filename, str) or not filename:
                filename = f"{artifact.artifact_id}.{artifact.artifact_format}"
            media_type = artifact.metadata.get("media_type")
            size_bytes = artifact.metadata.get("size_bytes")
            storage_key = artifact.metadata.get("storage_key")
            if not isinstance(storage_key, str) or not storage_key:
                raise JobValidationError(
                    f"Output artifact '{artifact.artifact_id}' is missing a storage key"
                )
            record = OutputArtifactRecord(
                artifact_id=artifact.artifact_id,
                artifact_format=artifact.artifact_format,
                role=artifact.role,
                exportable=artifact.exportable,
                metadata=artifact.metadata,
                job_id=job_id,
                filename=filename,
                media_type=media_type if isinstance(media_type, str) else None,
                size_bytes=size_bytes if isinstance(size_bytes, int) else None,
                storage_key=storage_key,
            )
            self.output_store.save(record)
            next_job.artifacts.append(record)
            self._append_event(
                RuntimeEvent(
                    job_id=job_id,
                    kind=RuntimeEventKind.JOB_ARTIFACT_READY,
                    phase=next_job.state.value,
                    data={
                        "artifact_id": record.artifact_id,
                        "artifact_format": record.artifact_format,
                        "storage_key": record.storage_key,
                    },
                )
            )
        next_job.updated_at = datetime.now(timezone.utc)
        self.job_store.save(next_job)

    def _prepared_request(self, request: JobRequest) -> JobRequest:
        if request.output.destination_mode != OutputDestinationMode.RUNTIME_MANAGED:
            raise JobValidationError(
                "Jobs must use runtime-managed outputs; export happens through explicit output routes"
            )
        model = self.catalog.get_model(request.model_id)
        capability = model.capability or (
            model.artifact.capability if model.artifact else None
        )
        if capability is None:
            return request
        artifact_format = request.output.artifact_format
        if artifact_format is None and capability.artifacts_out:
            output = request.output.model_copy(
                update={"artifact_format": capability.artifacts_out[0]}
            )
            return request.model_copy(update={"output": output})
        return request

    def _validate_request(
        self,
        *,
        model_id: str,
        capability: CapabilityDescriptor,
        request: JobRequest,
    ) -> None:
        if request.task not in capability.tasks:
            raise JobValidationError(
                f"Model '{model_id}' does not support task '{request.task}'"
            )
        artifact_format = request.output.artifact_format
        if artifact_format is None:
            raise JobValidationError("Job output requires an artifact format")
        if artifact_format not in capability.artifacts_out:
            raise JobValidationError(
                f"Artifact format '{artifact_format}' is not supported for '{model_id}'"
            )
        width = request.params.get("width")
        if width is not None and (not isinstance(width, int) or width % 32 != 0):
            raise JobValidationError("width must be an integer multiple of 32")
        height = request.params.get("height")
        if height is not None and (not isinstance(height, int) or height % 32 != 0):
            raise JobValidationError("height must be an integer multiple of 32")
        num_frames = request.params.get("num_frames")
        if num_frames is not None and (
            not isinstance(num_frames, int)
            or num_frames < 1
            or (num_frames - 1) % 8 != 0
        ):
            raise JobValidationError("num_frames must satisfy the current 8n+1 rule")
        images = request.inputs.get("images")
        audio = request.inputs.get("audio")
        if request.task == "video.condition.image" and (
            not isinstance(images, list) or not images
        ):
            raise JobValidationError(
                "video.condition.image requires at least one image input"
            )
        if request.task == "video.condition.audio" and not isinstance(audio, dict):
            raise JobValidationError(
                "video.condition.audio requires one audio input object"
            )
        if images is not None:
            if not isinstance(images, list):
                raise JobValidationError("images must be a list when provided")
            for image in images:
                if not isinstance(image, dict):
                    raise JobValidationError("images entries must be objects")
                handle_id = image.get("input_handle")
                if not isinstance(handle_id, str) or not handle_id:
                    raise JobValidationError(
                        "images entries require a non-empty input_handle"
                    )
                frame_index = image.get("frame_index", 0)
                if not isinstance(frame_index, int) or frame_index < 0:
                    raise JobValidationError(
                        "images frame_index must be a non-negative integer"
                    )
                if isinstance(num_frames, int) and frame_index >= num_frames:
                    raise JobValidationError(
                        "images frame_index must be within num_frames"
                    )
                strength = image.get("strength", 1.0)
                if not isinstance(strength, (int, float)):
                    raise JobValidationError("images strength must be numeric")
                strength_value = float(strength)
                if not 0.0 <= strength_value <= 1.0:
                    raise JobValidationError(
                        "images strength must be between 0.0 and 1.0"
                    )
        if audio is not None:
            if not isinstance(audio, dict):
                raise JobValidationError("audio must be an object when provided")
            handle_id = audio.get("input_handle")
            if not isinstance(handle_id, str) or not handle_id:
                raise JobValidationError(
                    "audio input requires a non-empty input_handle"
                )
            start_time_seconds = audio.get("start_time_seconds", 0.0)
            if not isinstance(start_time_seconds, (int, float)):
                raise JobValidationError("audio start_time_seconds must be numeric")
            if float(start_time_seconds) < 0.0:
                raise JobValidationError("audio start_time_seconds must be >= 0.0")
            max_duration_seconds = audio.get("max_duration_seconds")
            if max_duration_seconds is not None:
                if not isinstance(max_duration_seconds, (int, float)):
                    raise JobValidationError(
                        "audio max_duration_seconds must be numeric when provided"
                    )
                if float(max_duration_seconds) <= 0.0:
                    raise JobValidationError(
                        "audio max_duration_seconds must be > 0.0 when provided"
                    )
        for handle_id in _collect_input_handles(request.inputs):
            if self.input_store.get(handle_id) is None:
                raise JobValidationError(f"Unknown input handle '{handle_id}'")

    def _active_heavy_job_id(self) -> str | None:
        for job_id in self._running_jobs:
            record = self.job_store.get(job_id)
            if record is not None and record.state not in TERMINAL_STATES:
                return job_id
        return None


def _collect_input_handles(value: object) -> list[str]:
    handles: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "input_handle" and isinstance(nested, str):
                handles.append(nested)
            else:
                handles.extend(_collect_input_handles(nested))
    elif isinstance(value, list):
        for nested in value:
            handles.extend(_collect_input_handles(nested))
    return handles


def _job_process_context() -> ProcessContext:
    return get_context("spawn")
