from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from mlxr.core.runtime import RuntimeHome
from mlxr.core.runtime.manifests import (
    canonical_json_data,
    read_json_model,
    write_json_atomic,
)
from mlxr.core.schemas import (
    InputHandleRecord,
    JobRecord,
    ModelInstallOperationRecord,
    OutputArtifactRecord,
    RuntimeEvent,
)


class InputStore:
    def __init__(self, runtime_home: RuntimeHome) -> None:
        self.runtime_home = runtime_home

    def save(self, record: InputHandleRecord, payload: bytes) -> InputHandleRecord:
        payload_path = self.payload_path(record)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(payload)
        return self.persist(record)

    def save_stream(
        self, record: InputHandleRecord, chunks: Iterable[bytes]
    ) -> InputHandleRecord:
        payload_path = self.payload_path(record)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        size_bytes = 0
        with payload_path.open("wb") as handle:
            for chunk in chunks:
                if not chunk:
                    continue
                handle.write(chunk)
                size_bytes += len(chunk)
        return self.persist(record.model_copy(update={"size_bytes": size_bytes}))

    def persist(self, record: InputHandleRecord) -> InputHandleRecord:
        write_json_atomic(
            self.runtime_home.input_handle_manifest_path(record.handle_id), record
        )
        return record

    def get(self, handle_id: str) -> InputHandleRecord | None:
        path = self.runtime_home.input_handle_manifest_path(handle_id)
        if not path.exists():
            return None
        return read_json_model(path, InputHandleRecord)

    def list_records(self) -> list[InputHandleRecord]:
        records = [
            read_json_model(path, InputHandleRecord)
            for path in self.runtime_home.inputs_dir.rglob("manifest.json")
        ]
        return sorted(records, key=lambda record: record.handle_id)

    def payload_path(self, record: InputHandleRecord) -> Path:
        filename = record.filename or f"{record.handle_id}.bin"
        return self.runtime_home.input_payload_path(record.handle_id, filename)


class JobStore:
    def __init__(self, runtime_home: RuntimeHome) -> None:
        self.runtime_home = runtime_home

    def save(self, record: JobRecord) -> JobRecord:
        write_json_atomic(self.runtime_home.job_record_path(record.job_id), record)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        path = self.runtime_home.job_record_path(job_id)
        if not path.exists():
            return None
        return read_json_model(path, JobRecord)

    def list_records(self) -> list[JobRecord]:
        records = [
            read_json_model(path, JobRecord)
            for path in self.runtime_home.jobs_dir.rglob("record.json")
        ]
        return sorted(records, key=lambda record: record.created_at)

    def append_event(self, job_id: str, event: RuntimeEvent) -> RuntimeEvent:
        path = self.runtime_home.job_events_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{canonical_json_data(event)}\n")
        return event

    def list_events(self, job_id: str) -> list[RuntimeEvent]:
        path = self.runtime_home.job_events_path(job_id)
        if not path.exists():
            return []
        events: list[RuntimeEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(RuntimeEvent.model_validate_json(line))
        return events


class ModelInstallStore:
    def __init__(self, runtime_home: RuntimeHome) -> None:
        self.runtime_home = runtime_home

    def save(self, record: ModelInstallOperationRecord) -> ModelInstallOperationRecord:
        write_json_atomic(
            self.runtime_home.model_install_record_path(record.operation_id), record
        )
        return record

    def get(self, operation_id: str) -> ModelInstallOperationRecord | None:
        path = self.runtime_home.model_install_record_path(operation_id)
        if not path.exists():
            return None
        return read_json_model(path, ModelInstallOperationRecord)

    def list_records(self) -> list[ModelInstallOperationRecord]:
        records = [
            read_json_model(path, ModelInstallOperationRecord)
            for path in self.runtime_home.model_installs_dir.glob("*.json")
        ]
        return sorted(records, key=lambda record: record.created_at)


class OutputStore:
    def __init__(self, runtime_home: RuntimeHome) -> None:
        self.runtime_home = runtime_home

    def save(self, record: OutputArtifactRecord) -> OutputArtifactRecord:
        write_json_atomic(
            self.runtime_home.output_artifact_manifest_path(
                record.job_id, record.artifact_id
            ),
            record,
        )
        return record

    def get(self, artifact_id: str) -> OutputArtifactRecord | None:
        for path in self.runtime_home.jobs_dir.rglob("manifest.json"):
            record = read_json_model(path, OutputArtifactRecord)
            if record.artifact_id == artifact_id:
                return record
        return None

    def payload_path(self, record: OutputArtifactRecord) -> Path:
        filename = record.filename or f"{record.artifact_id}.{record.artifact_format}"
        return self.runtime_home.output_artifact_path(
            record.job_id, record.artifact_id, filename
        )
