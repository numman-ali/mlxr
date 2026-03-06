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
