from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


def _split_csv(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    values = [item.strip() for item in raw_value.split(",")]
    return tuple(item for item in values if item)


def _origin_from_referer(referer: str) -> str | None:
    parsed = urlparse(referer)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(frozen=True, slots=True)
class ServerSettings:
    http_enabled: bool = False
    http_host: str | None = None
    http_port: int = 46321
    http_bearer_token: str | None = None
    allowed_origins: tuple[str, ...] = ()
    uds_path: str | None = None
    job_execution_mode: str = "spawn"

    @classmethod
    def from_env(cls) -> "ServerSettings":
        http_host = os.environ.get("MLX_RUNTIME_HTTP_HOST")
        http_port = int(os.environ.get("MLX_RUNTIME_HTTP_PORT", "46321"))
        return cls(
            http_enabled=http_host is not None,
            http_host=http_host,
            http_port=http_port,
            http_bearer_token=os.environ.get("MLX_RUNTIME_HTTP_TOKEN"),
            allowed_origins=_split_csv(os.environ.get("MLX_RUNTIME_ALLOWED_ORIGINS")),
            uds_path=os.environ.get("MLX_RUNTIME_UDS_PATH"),
            job_execution_mode=os.environ.get(
                "MLX_RUNTIME_JOB_EXECUTION_MODE", "spawn"
            ),
        )

    def validate_startup(self) -> None:
        if self.http_enabled and not self.http_bearer_token:
            raise RuntimeError(
                "MLX_RUNTIME_HTTP_TOKEN is required when loopback HTTP is enabled"
            )
        normalized_mode = self.job_execution_mode.strip().lower()
        if normalized_mode not in {"spawn", "thread"}:
            raise RuntimeError(
                "MLX_RUNTIME_JOB_EXECUTION_MODE must be 'spawn' or 'thread'"
            )

    def origin_allowed(self, origin: str) -> bool:
        return origin in self.allowed_origins

    def referer_allowed(self, referer: str) -> bool:
        origin = _origin_from_referer(referer)
        return origin is not None and self.origin_allowed(origin)
