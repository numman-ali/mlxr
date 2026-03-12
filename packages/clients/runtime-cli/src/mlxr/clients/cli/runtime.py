from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from mlxr.core.runtime import RuntimeHome
from mlxr.core.schemas import (
    ArtifactExportResult,
    InputHandleRecord,
    JobRecord,
    ModelInstallResult,
    ModelRecord,
    SupportedModelDescriptor,
    WorkflowIntent,
    WorkflowPlanResult,
    WorkflowRunRequest,
    WorkflowRunResult,
)

from .constants import DEFAULT_STARTUP_TIMEOUT_SECONDS
from .daemon import (
    daemon_socket_path,
    ensure_runtime_daemon,
)
from .io_utils import _default_uds_path, _file_chunks, _media_type_for_path

_ScalarParamValue = str | int | float | bool | None
_RequestParamValue = _ScalarParamValue | Sequence[_ScalarParamValue]


@dataclass(frozen=True, slots=True)
class _RuntimeConnectionOptions:
    base_url: str | None
    uds_path: Path | None
    http_token: str | None


class RuntimeUnavailableError(RuntimeError):
    pass


class RuntimeApiError(RuntimeError):
    def __init__(self, *, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class RuntimeClient:
    def __init__(
        self,
        *,
        base_url: str | None,
        uds_path: Path | None,
        http_token: str | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if http_token:
            headers["Authorization"] = f"Bearer {http_token}"
        resolved_base_url = (
            base_url.rstrip("/") if base_url else None
        ) or "http://mlxr"
        transport: httpx.BaseTransport | None = None
        if base_url is None:
            resolved_uds_path = (uds_path or _default_uds_path()).expanduser().resolve()
            transport = httpx.HTTPTransport(
                uds=str(resolved_uds_path), retries=0, trust_env=False
            )
        self._client = httpx.Client(
            base_url=resolved_base_url,
            headers=headers,
            transport=transport,
            timeout=None,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, str]:
        response = self._request("GET", "/health")
        payload = response.json()
        return {key: str(value) for key, value in payload.items()}

    def import_file(self, path: Path, *, kind: str) -> InputHandleRecord:
        media_type = _media_type_for_path(path, kind)
        response = self._request(
            "POST",
            "/v1/inputs/import-file",
            params={
                "filename": path.name,
                "media_type": media_type,
                "role": kind,
            },
            content=_file_chunks(path),
        )
        return InputHandleRecord.model_validate(response.json())

    def plan(self, intent: WorkflowIntent) -> WorkflowPlanResult:
        response = self._request(
            "POST",
            "/v1/workflows/plan",
            json_payload=intent.model_dump(mode="json"),
        )
        return WorkflowPlanResult.model_validate(response.json())

    def run(self, intent: WorkflowIntent) -> WorkflowRunResult:
        response = self._request(
            "POST",
            "/v1/workflows/run",
            json_payload=WorkflowRunRequest(intent=intent).model_dump(mode="json"),
        )
        return WorkflowRunResult.model_validate(response.json())

    def get_job(self, job_id: str) -> JobRecord:
        response = self._request("GET", f"/v1/jobs/{job_id}")
        return JobRecord.model_validate(response.json())

    def export_output(
        self, artifact_id: str, *, destination_path: Path, overwrite: bool
    ) -> ArtifactExportResult:
        response = self._request(
            "POST",
            f"/v1/outputs/{artifact_id}/export",
            json_payload={
                "destination_path": str(destination_path),
                "overwrite": overwrite,
            },
        )
        return ArtifactExportResult.model_validate(response.json())

    def list_models(self) -> list[ModelRecord]:
        response = self._request("GET", "/v1/models")
        return [ModelRecord.model_validate(item) for item in response.json()]

    def list_supported_models(self) -> list[SupportedModelDescriptor]:
        response = self._request("GET", "/v1/models/supported")
        return [
            SupportedModelDescriptor.model_validate(item) for item in response.json()
        ]

    def install_model(self, model_id: str) -> ModelInstallResult:
        response = self._request(
            "POST",
            "/v1/models/install",
            json_payload={"model_id": model_id},
        )
        return ModelInstallResult.model_validate(response.json())

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: object | None = None,
        params: Mapping[str, _RequestParamValue] | None = None,
        content: Iterator[bytes] | bytes | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.request(
                method,
                path,
                json=json_payload,
                params=params,
                content=content,
            )
        except httpx.ConnectError as exc:
            raise RuntimeUnavailableError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(str(exc)) from exc
        if response.status_code >= 400:
            raise RuntimeApiError(
                status_code=response.status_code,
                detail=_response_error_detail(response),
            )
        return response


@contextmanager
def _runtime_client_for_args(
    args: argparse.Namespace,
    *,
    auto_start: bool,
) -> Iterator[RuntimeClient]:
    options = _runtime_connection_options_from_args(args)
    uds_path = options.uds_path
    if auto_start and options.base_url is None:
        runtime_home = RuntimeHome.from_env()
        socket_path = daemon_socket_path(runtime_home, uds_path)
        status = ensure_runtime_daemon(
            runtime_home=runtime_home,
            socket_path=socket_path,
            startup_timeout_seconds=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        )
        if status.status == "started":
            print(
                f"Started MLXR runtime at {status.socket_path}",
                file=sys.stderr,
            )
        uds_path = socket_path
    client = RuntimeClient(
        base_url=options.base_url,
        uds_path=uds_path,
        http_token=options.http_token,
    )
    try:
        yield client
    finally:
        client.close()


def _runtime_connection_options_from_args(
    args: argparse.Namespace,
) -> _RuntimeConnectionOptions:
    uds_path = args.uds_path if isinstance(args.uds_path, Path) else None
    return _RuntimeConnectionOptions(
        base_url=str(args.runtime_url) if args.runtime_url else None,
        uds_path=uds_path,
        http_token=str(args.http_token) if args.http_token is not None else None,
    )


def _response_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail
    return response.text.strip() or f"HTTP {response.status_code}"


def _runtime_unavailable_message(args: argparse.Namespace, error: Exception) -> str:
    if args.runtime_url:
        return f"Could not reach the explicit runtime at {args.runtime_url}: {error}"
    return (
        "Could not reach the local MLXR runtime. "
        "Run `mlxr serve` to start it, or rerun with `--runtime-url` for an explicit endpoint."
    )
