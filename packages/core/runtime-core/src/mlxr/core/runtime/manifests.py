from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from mlxr.core.schemas import (
    ModelRecord,
    PortableArtifactRecord,
    SourceRef,
    SourceRegistrationRecord,
)
from pydantic import BaseModel

from .runtime_home import RuntimeHome

ModelT = TypeVar("ModelT", bound=BaseModel)


def canonical_json_data(value: Any) -> str:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", exclude_none=True)
    else:
        payload = value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json_atomic(path: Path, value: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_data(value)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(f"{payload}\n", encoding="utf-8")
    tmp_path.replace(path)


def read_json_model(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


class SourceManifestStore:
    def __init__(self, runtime_home: RuntimeHome) -> None:
        self.runtime_home = runtime_home

    def save(self, record: SourceRegistrationRecord) -> SourceRegistrationRecord:
        path = self.runtime_home.source_manifest_path(
            record.source.provider, record.source_id
        )
        write_json_atomic(path, record)
        return record

    def get(self, source_id: str) -> SourceRegistrationRecord | None:
        for path in self.runtime_home.sources_ref_dir.rglob("manifest.json"):
            record = read_json_model(path, SourceRegistrationRecord)
            if record.source_id == source_id:
                return record
        return None

    def list(self) -> list[SourceRegistrationRecord]:
        records = [
            read_json_model(path, SourceRegistrationRecord)
            for path in self.runtime_home.sources_ref_dir.rglob("manifest.json")
        ]
        return sorted(records, key=lambda record: record.source_id)


class ArtifactManifestStore:
    def __init__(self, runtime_home: RuntimeHome) -> None:
        self.runtime_home = runtime_home

    def save(self, record: PortableArtifactRecord) -> PortableArtifactRecord:
        path = self.runtime_home.artifact_manifest_path(
            record.family, record.model_id, record.artifact_digest
        )
        write_json_atomic(path, record)
        return record

    def get(self, artifact_digest: str) -> PortableArtifactRecord | None:
        for path in self.runtime_home.artifacts_portable_dir.rglob("artifact.json"):
            record = read_json_model(path, PortableArtifactRecord)
            if record.artifact_digest == artifact_digest:
                return record
        return None

    def list(self) -> list[PortableArtifactRecord]:
        records = [
            read_json_model(path, PortableArtifactRecord)
            for path in self.runtime_home.artifacts_portable_dir.rglob("artifact.json")
        ]
        return sorted(
            records, key=lambda record: (record.model_id, record.artifact_digest)
        )


class ModelManifestStore:
    def __init__(self, runtime_home: RuntimeHome) -> None:
        self.runtime_home = runtime_home

    def save(self, record: ModelRecord) -> ModelRecord:
        path = self.runtime_home.model_manifest_path(record.model_id)
        write_json_atomic(path, record)
        return record

    def get(self, model_id: str) -> ModelRecord | None:
        path = self.runtime_home.model_manifest_path(model_id)
        if not path.exists():
            return None
        return read_json_model(path, ModelRecord)

    def list(self) -> list[ModelRecord]:
        records = [
            read_json_model(path, ModelRecord)
            for path in self.runtime_home.models_dir.glob("*.json")
        ]
        return sorted(records, key=lambda record: record.model_id)


def source_id_for_ref(source_ref: SourceRef) -> str:
    import hashlib

    digest = hashlib.sha256(canonical_json_data(source_ref).encode("utf-8")).hexdigest()
    return f"src_{digest}"
