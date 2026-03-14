from __future__ import annotations

import queue
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from queue import Queue
from typing import Callable, Protocol

from mlxr.core.runtime import RuntimeCatalog
from mlxr.core.schemas import (
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

    def get(self, timeout: float | None = None) -> WorkerMessage: ...

    def get_nowait(self) -> WorkerMessage: ...

    def close(self) -> None: ...


class ManagedProcess(Protocol):
    def start(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    @property
    def exitcode(self) -> int | None: ...


class SpawnQueueLike(Protocol):
    def put(self, item: WorkerMessage) -> None: ...

    def get(self, block: bool = True, timeout: float | None = None) -> object: ...

    def get_nowait(self) -> object: ...

    def close(self) -> None: ...


class ProcessContext(Protocol):
    def Queue(self, maxsize: int = 0) -> ManagedMessageQueue: ...

    def Process(
        self, target: Callable[..., object], kwargs: dict[str, object]
    ) -> ManagedProcess: ...


TERMINAL_STATES = {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
HEAVY_JOB_SCHEDULER_CLASSES = {"media_video_dit"}


class SpawnManagedQueue:
    def __init__(self, queue_obj: SpawnQueueLike) -> None:
        self._queue = queue_obj

    def put(self, item: WorkerMessage) -> None:
        self._queue.put(item)

    def get(self, timeout: float | None = None) -> WorkerMessage:
        item = self._queue.get(timeout=timeout)
        if not isinstance(item, dict):
            raise TypeError("Worker queue emitted a non-dictionary message")
        return item

    def get_nowait(self) -> WorkerMessage:
        item = self._queue.get_nowait()
        if not isinstance(item, dict):
            raise TypeError("Worker queue emitted a non-dictionary message")
        return item

    def close(self) -> None:
        self._queue.close()


class SpawnProcessContext:
    def __init__(self) -> None:
        self._context = get_context("spawn")

    def Queue(self, maxsize: int = 0) -> SpawnManagedQueue:
        return SpawnManagedQueue(self._context.Queue(maxsize=maxsize))

    def Process(
        self, target: Callable[..., object], kwargs: dict[str, object]
    ) -> ManagedProcess:
        return self._context.Process(target=target, kwargs=kwargs)


class ThreadMessageQueue:
    def __init__(self, maxsize: int = 0) -> None:
        self._queue: queue.Queue[WorkerMessage] = queue.Queue(maxsize=maxsize)

    def put(self, item: WorkerMessage) -> None:
        self._queue.put(item)

    def get(self, timeout: float | None = None) -> WorkerMessage:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> WorkerMessage:
        return self._queue.get_nowait()

    def close(self) -> None:
        return None


class ThreadManagedProcess:
    def __init__(
        self,
        *,
        target: Callable[..., object],
        kwargs: dict[str, object],
    ) -> None:
        self._target = target
        self._kwargs = kwargs
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self._target(**self._kwargs)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def terminate(self) -> None:
        return None

    @property
    def exitcode(self) -> int | None:
        if self._thread.is_alive():
            return None
        return 0


class ThreadProcessContext:
    def Queue(self, maxsize: int = 0) -> ThreadMessageQueue:
        return ThreadMessageQueue(maxsize=maxsize)

    def Process(
        self,
        target: Callable[..., object],
        kwargs: dict[str, object],
    ) -> ThreadManagedProcess:
        return ThreadManagedProcess(target=target, kwargs=kwargs)


def _process_exited(process: ManagedProcess) -> bool:
    exitcode = process.exitcode
    if exitcode is not None:
        return True
    return not process.is_alive()


def _integer_constraint(
    constraints: dict[str, object], key: str
) -> dict[str, object] | None:
    raw = constraints.get(key)
    return raw if isinstance(raw, dict) else None


def _validate_dimension_constraint(
    *,
    name: str,
    value: object,
    constraints: dict[str, object],
) -> None:
    if value is None:
        return
    if not isinstance(value, int):
        raise JobValidationError(f"{name} must be an integer")
    constraint = _integer_constraint(constraints, name)
    if constraint is None:
        return
    multiple_of = constraint.get("multiple_of")
    if isinstance(multiple_of, int) and multiple_of > 0 and value % multiple_of != 0:
        raise JobValidationError(f"{name} must be an integer multiple of {multiple_of}")
    minimum = constraint.get("minimum")
    if isinstance(minimum, int) and value < minimum:
        raise JobValidationError(f"{name} must be >= {minimum}")


def _validate_num_frames_constraint(
    value: object, constraints: dict[str, object]
) -> None:
    if value is None:
        return
    if not isinstance(value, int):
        raise JobValidationError("num_frames must be an integer")
    constraint = _integer_constraint(constraints, "num_frames")
    if constraint is None:
        return
    minimum = constraint.get("minimum")
    if isinstance(minimum, int) and value < minimum:
        raise JobValidationError(f"num_frames must be >= {minimum}")
    formula = constraint.get("formula")
    if formula == "8n+1" and (value < 1 or (value - 1) % 8 != 0):
        raise JobValidationError("num_frames must satisfy the current 8n+1 rule")


def _validate_numeric_constraint(
    *,
    name: str,
    value: object,
    constraints: dict[str, object],
    integer: bool,
) -> None:
    if value is None:
        return
    if integer:
        if not isinstance(value, int) or isinstance(value, bool):
            raise JobValidationError(f"{name} must be an integer")
        numeric_value = float(value)
    else:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise JobValidationError(f"{name} must be numeric")
        numeric_value = float(value)

    raw_constraint = constraints.get(name)
    if not isinstance(raw_constraint, dict):
        return

    fixed = raw_constraint.get("fixed")
    if fixed is not None:
        if integer:
            if not isinstance(fixed, int) or isinstance(fixed, bool):
                raise JobValidationError(f"{name} fixed constraint must be an integer")
            if int(numeric_value) != fixed:
                raise JobValidationError(f"{name} must be exactly {fixed}")
        else:
            if not isinstance(fixed, (int, float)) or isinstance(fixed, bool):
                raise JobValidationError(f"{name} fixed constraint must be numeric")
            if numeric_value != float(fixed):
                raise JobValidationError(f"{name} must be exactly {fixed}")

    minimum = raw_constraint.get("minimum")
    if minimum is not None:
        if not isinstance(minimum, (int, float)) or isinstance(minimum, bool):
            raise JobValidationError(f"{name} minimum constraint must be numeric")
        if numeric_value < float(minimum):
            raise JobValidationError(f"{name} must be >= {minimum}")

    maximum = raw_constraint.get("maximum")
    if maximum is not None:
        if not isinstance(maximum, (int, float)) or isinstance(maximum, bool):
            raise JobValidationError(f"{name} maximum constraint must be numeric")
        if numeric_value > float(maximum):
            raise JobValidationError(f"{name} must be <= {maximum}")

    if integer:
        multiple_of = raw_constraint.get("multiple_of")
        if multiple_of is not None:
            if (
                not isinstance(multiple_of, int)
                or isinstance(multiple_of, bool)
                or multiple_of <= 0
            ):
                raise JobValidationError(
                    f"{name} multiple_of constraint must be a positive integer"
                )
            if int(numeric_value) % multiple_of != 0:
                raise JobValidationError(
                    f"{name} must be an integer multiple of {multiple_of}"
                )


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
        self._reconcile_orphaned_jobs()

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

            context = _job_process_context(self.settings.job_execution_mode)
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
                try:
                    message = event_queue.get(timeout=0.2)
                except queue.Empty:
                    if _process_exited(process):
                        break
                    continue
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
                error_message = "Worker exited before reaching a terminal state"
                if process.exitcode is not None:
                    error_message = f"{error_message} (exitcode={process.exitcode})"
                failed_record = record.model_copy(
                    update={
                        "state": JobState.FAILED,
                        "updated_at": datetime.now(timezone.utc),
                        "error": error_message,
                    }
                )
                self.job_store.save(failed_record)
            command_queue.close()
            event_queue.close()
            with self._lock:
                self._running_jobs.pop(job_id, None)
                self._subscribers.pop(job_id, None)

    def _reconcile_orphaned_jobs(self) -> None:
        now = datetime.now(timezone.utc)
        for record in self.job_store.list_records():
            if record.state in TERMINAL_STATES:
                continue
            failed_record = record.model_copy(
                update={
                    "state": JobState.FAILED,
                    "updated_at": now,
                    "error": (
                        "Control-plane restarted before job reached a terminal state"
                    ),
                }
            )
            self.job_store.save(failed_record)

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
            if next_record.state not in TERMINAL_STATES:
                next_record.error = None
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
        if request.task == "video.condition.video" and artifact_format == "wav":
            raise JobValidationError(
                "video.condition.video currently only supports mp4 output"
            )
        width = request.params.get("width")
        height = request.params.get("height")
        constraints = dict(capability.constraints)
        _validate_dimension_constraint(
            name="width", value=width, constraints=constraints
        )
        _validate_dimension_constraint(
            name="height", value=height, constraints=constraints
        )
        num_frames = request.params.get("num_frames")
        _validate_num_frames_constraint(num_frames, constraints)
        _validate_numeric_constraint(
            name="num_inference_steps",
            value=request.params.get("num_inference_steps"),
            constraints=constraints,
            integer=True,
        )
        _validate_numeric_constraint(
            name="guidance_scale",
            value=request.params.get("guidance_scale"),
            constraints=constraints,
            integer=False,
        )
        images = request.inputs.get("images")
        videos = request.inputs.get("videos")
        audio = request.inputs.get("audio")
        loras = request.inputs.get("loras")
        if request.task == "image.edit" and (
            not isinstance(images, list) or not images
        ):
            raise JobValidationError("image.edit requires at least one image input")
        if request.task == "video.condition.image" and (
            not isinstance(images, list) or not images
        ):
            raise JobValidationError(
                "video.condition.image requires at least one image input"
            )
        if request.task == "video.interpolate" and (
            not isinstance(images, list) or len(images) < 2
        ):
            raise JobValidationError(
                "video.interpolate requires at least two image inputs"
            )
        if request.task == "video.condition.video" and (
            not isinstance(videos, list) or len(videos) != 1
        ):
            raise JobValidationError(
                "video.condition.video requires exactly one reference video input"
            )
        if request.task == "video.retake" and (
            not isinstance(videos, list) or len(videos) != 1
        ):
            raise JobValidationError(
                "video.retake requires exactly one source video input"
            )
        if request.task == "video.condition.audio" and not isinstance(audio, dict):
            raise JobValidationError(
                "video.condition.audio requires one audio input object"
            )
        if images is not None:
            if not isinstance(images, list):
                raise JobValidationError("images must be a list when provided")
            image_frame_indices: list[int] = []
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
                image_frame_indices.append(frame_index)
                strength = image.get("strength", 1.0)
                if not isinstance(strength, (int, float)):
                    raise JobValidationError("images strength must be numeric")
                strength_value = float(strength)
                if not 0.0 <= strength_value <= 1.0:
                    raise JobValidationError(
                        "images strength must be between 0.0 and 1.0"
                    )
            if (
                request.task == "video.interpolate"
                and len(set(image_frame_indices)) < 2
            ):
                raise JobValidationError(
                    "video.interpolate requires image inputs at at least two distinct frame indices"
                )
        if videos is not None:
            if not isinstance(videos, list):
                raise JobValidationError("videos must be a list when provided")
            for video in videos:
                if not isinstance(video, dict):
                    raise JobValidationError("videos entries must be objects")
                handle_id = video.get("input_handle")
                if not isinstance(handle_id, str) or not handle_id:
                    raise JobValidationError(
                        "videos entries require a non-empty input_handle"
                    )
                strength = video.get("strength", 1.0)
                if not isinstance(strength, (int, float)):
                    raise JobValidationError("videos strength must be numeric")
                strength_value = float(strength)
                if not 0.0 <= strength_value <= 1.0:
                    raise JobValidationError(
                        "videos strength must be between 0.0 and 1.0"
                    )
        if request.task == "video.retake":
            start_seconds = request.params.get("window_start_seconds")
            end_seconds = request.params.get("window_end_seconds")
            if not isinstance(start_seconds, (int, float)):
                raise JobValidationError(
                    "video.retake requires numeric params.window_start_seconds"
                )
            if not isinstance(end_seconds, (int, float)):
                raise JobValidationError(
                    "video.retake requires numeric params.window_end_seconds"
                )
            if float(start_seconds) < 0.0:
                raise JobValidationError(
                    "video.retake params.window_start_seconds must be >= 0.0"
                )
            if float(end_seconds) <= float(start_seconds):
                raise JobValidationError(
                    "video.retake params.window_end_seconds must be greater than params.window_start_seconds"
                )
            regenerate_video = request.params.get("regenerate_video", True)
            regenerate_audio = request.params.get("regenerate_audio", True)
            if not isinstance(regenerate_video, bool):
                raise JobValidationError(
                    "video.retake params.regenerate_video must be boolean when provided"
                )
            if not isinstance(regenerate_audio, bool):
                raise JobValidationError(
                    "video.retake params.regenerate_audio must be boolean when provided"
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
        if loras is not None:
            if not isinstance(loras, list):
                raise JobValidationError("loras must be a list when provided")
            for lora in loras:
                if not isinstance(lora, dict):
                    raise JobValidationError("loras entries must be objects")
                handle_id = lora.get("input_handle")
                if not isinstance(handle_id, str) or not handle_id:
                    raise JobValidationError(
                        "loras entries require a non-empty input_handle"
                    )
                strength = lora.get("strength", 1.0)
                if not isinstance(strength, (int, float)):
                    raise JobValidationError("loras strength must be numeric")
                if float(strength) <= 0.0:
                    raise JobValidationError("loras strength must be greater than 0.0")
        if request.task == "video.condition.video" and (
            not isinstance(loras, list) or len(loras) != 1
        ):
            raise JobValidationError(
                "video.condition.video requires exactly one LoRA input"
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


def _job_process_context(mode: str = "spawn") -> ProcessContext:
    normalized = mode.strip().lower()
    if normalized == "spawn":
        return SpawnProcessContext()
    if normalized == "thread":
        return ThreadProcessContext()
    raise RuntimeError("Unsupported job execution mode; expected 'spawn' or 'thread'")
