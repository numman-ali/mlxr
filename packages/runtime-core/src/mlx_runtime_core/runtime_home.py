from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def _safe_path_segment(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "item"


@dataclass(frozen=True, slots=True)
class RuntimeHome:
    root: Path

    @classmethod
    def from_env(cls) -> "RuntimeHome":
        raw_root = os.environ.get("MLX_RUNTIME_HOME", str(Path.home() / ".mlx-runtime"))
        return cls(root=Path(raw_root).expanduser())

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def temp_dir(self) -> Path:
        return self.root / "temp"

    @property
    def inputs_dir(self) -> Path:
        return self.temp_dir / "inputs"

    @property
    def sources_ref_dir(self) -> Path:
        return self.root / "sources-ref"

    @property
    def artifacts_portable_dir(self) -> Path:
        return self.root / "artifacts-portable"

    @property
    def build_cache_local_dir(self) -> Path:
        return self.root / "build-cache-local"

    @property
    def models_dir(self) -> Path:
        return self.config_dir / "models"

    def ensure_layout(self) -> None:
        for directory in (
            self.config_dir,
            self.logs_dir,
            self.jobs_dir,
            self.temp_dir,
            self.inputs_dir,
            self.sources_ref_dir,
            self.artifacts_portable_dir,
            self.build_cache_local_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def source_manifest_path(self, provider: str, source_id: str) -> Path:
        return (
            self.sources_ref_dir
            / _safe_path_segment(provider)
            / _safe_path_segment(source_id)
            / "manifest.json"
        )

    def artifact_manifest_path(
        self, family: str, model_id: str, artifact_digest: str
    ) -> Path:
        return (
            self.artifacts_portable_dir
            / _safe_path_segment(family)
            / _safe_path_segment(model_id)
            / _safe_path_segment(artifact_digest)
            / "artifact.json"
        )

    def artifact_storage_key(
        self, family: str, model_id: str, artifact_digest: str
    ) -> str:
        return "/".join(
            (
                "artifacts-portable",
                _safe_path_segment(family),
                _safe_path_segment(model_id),
                _safe_path_segment(artifact_digest),
            )
        )

    def model_manifest_path(self, model_id: str) -> Path:
        return self.models_dir / f"{_safe_path_segment(model_id)}.json"

    def input_handle_dir(self, handle_id: str) -> Path:
        return self.inputs_dir / _safe_path_segment(handle_id)

    def input_handle_manifest_path(self, handle_id: str) -> Path:
        return self.input_handle_dir(handle_id) / "manifest.json"

    def input_payload_path(self, handle_id: str, filename: str) -> Path:
        return self.input_handle_dir(handle_id) / _safe_path_segment(filename)

    def input_storage_key(self, handle_id: str, filename: str) -> str:
        return "/".join(
            (
                "temp",
                "inputs",
                _safe_path_segment(handle_id),
                _safe_path_segment(filename),
            )
        )

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / _safe_path_segment(job_id)

    def job_record_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "record.json"

    def job_events_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "events.jsonl"

    def output_artifact_dir(self, job_id: str, artifact_id: str) -> Path:
        return self.job_dir(job_id) / "outputs" / _safe_path_segment(artifact_id)

    def output_artifact_manifest_path(self, job_id: str, artifact_id: str) -> Path:
        return self.output_artifact_dir(job_id, artifact_id) / "manifest.json"

    def output_artifact_path(
        self, job_id: str, artifact_id: str, filename: str
    ) -> Path:
        return self.output_artifact_dir(job_id, artifact_id) / _safe_path_segment(
            filename
        )

    def output_artifact_storage_key(
        self, job_id: str, artifact_id: str, filename: str
    ) -> str:
        return "/".join(
            (
                "jobs",
                _safe_path_segment(job_id),
                "outputs",
                _safe_path_segment(artifact_id),
                _safe_path_segment(filename),
            )
        )
