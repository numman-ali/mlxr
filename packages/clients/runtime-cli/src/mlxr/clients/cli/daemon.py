from __future__ import annotations

import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from mlxr.core.runtime import RuntimeHome

_DEFAULT_BASE_URL = "http://mlxr"
_STDIO_LOG_FILENAME = "control-plane-stdio.log"
_DAEMON_METADATA_FILENAME = "control-plane-daemon.json"
_MAX_UDS_PATH_BYTES = 100


class RuntimeDaemonError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DaemonMetadata:
    pid: int
    runtime_home: str
    socket_path: str
    stdio_log_path: str
    started_at: str
    package_version: str | None = None


@dataclass(frozen=True, slots=True)
class DaemonStatus:
    status: str
    runtime_home: str
    socket_path: str
    stdio_log_path: str
    metadata_path: str
    managed: bool
    healthy: bool
    pid: int | None = None
    message: str | None = None


def daemon_socket_path(
    runtime_home: RuntimeHome, explicit_uds_path: Path | None
) -> Path:
    return (
        explicit_uds_path.expanduser().resolve()
        if explicit_uds_path is not None
        else (runtime_home.temp_dir / "control-plane.sock").resolve()
    )


def daemon_stdio_log_path(runtime_home: RuntimeHome) -> Path:
    return (runtime_home.logs_dir / _STDIO_LOG_FILENAME).resolve()


def daemon_metadata_path(runtime_home: RuntimeHome) -> Path:
    return (runtime_home.temp_dir / _DAEMON_METADATA_FILENAME).resolve()


def daemon_health(*, socket_path: Path, timeout_seconds: float = 2.0) -> bool:
    if not socket_path.exists():
        return False
    transport = httpx.HTTPTransport(uds=str(socket_path), retries=0, trust_env=False)
    client = httpx.Client(
        transport=transport,
        base_url=_DEFAULT_BASE_URL,
        timeout=httpx.Timeout(timeout_seconds),
    )
    try:
        response = client.get("/health")
        return response.status_code == 200
    except httpx.HTTPError:
        return False
    finally:
        client.close()


def read_daemon_metadata(runtime_home: RuntimeHome) -> DaemonMetadata | None:
    path = daemon_metadata_path(runtime_home)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return DaemonMetadata(
        pid=int(payload["pid"]),
        runtime_home=str(payload["runtime_home"]),
        socket_path=str(payload["socket_path"]),
        stdio_log_path=str(payload["stdio_log_path"]),
        started_at=str(payload["started_at"]),
        package_version=(
            str(payload["package_version"])
            if payload.get("package_version") is not None
            else None
        ),
    )


def daemon_status(*, runtime_home: RuntimeHome, socket_path: Path) -> DaemonStatus:
    runtime_home.ensure_layout()
    metadata = read_daemon_metadata(runtime_home)
    healthy = daemon_health(socket_path=socket_path)
    pid = metadata.pid if metadata is not None else None
    pid_running = _pid_running(pid)
    managed = metadata is not None
    if healthy and managed:
        return DaemonStatus(
            status="running",
            runtime_home=str(runtime_home.root),
            socket_path=str(socket_path),
            stdio_log_path=str(daemon_stdio_log_path(runtime_home)),
            metadata_path=str(daemon_metadata_path(runtime_home)),
            managed=True,
            healthy=True,
            pid=pid,
            message="Managed local daemon is healthy.",
        )
    if healthy:
        return DaemonStatus(
            status="running",
            runtime_home=str(runtime_home.root),
            socket_path=str(socket_path),
            stdio_log_path=str(daemon_stdio_log_path(runtime_home)),
            metadata_path=str(daemon_metadata_path(runtime_home)),
            managed=False,
            healthy=True,
            pid=pid if pid_running else None,
            message="A healthy daemon is reachable, but it is not managed by mlxr.",
        )
    if managed and pid_running:
        return DaemonStatus(
            status="unhealthy",
            runtime_home=str(runtime_home.root),
            socket_path=str(socket_path),
            stdio_log_path=str(daemon_stdio_log_path(runtime_home)),
            metadata_path=str(daemon_metadata_path(runtime_home)),
            managed=True,
            healthy=False,
            pid=pid,
            message="Managed daemon process exists, but the health endpoint is unavailable.",
        )
    return DaemonStatus(
        status="stopped",
        runtime_home=str(runtime_home.root),
        socket_path=str(socket_path),
        stdio_log_path=str(daemon_stdio_log_path(runtime_home)),
        metadata_path=str(daemon_metadata_path(runtime_home)),
        managed=managed,
        healthy=False,
        pid=pid if pid_running else None,
        message="No healthy local daemon is running.",
    )


def ensure_runtime_daemon(
    *,
    runtime_home: RuntimeHome,
    socket_path: Path,
    startup_timeout_seconds: float = 30.0,
) -> DaemonStatus:
    runtime_home.ensure_layout()
    _validate_socket_path(socket_path)
    current = daemon_status(runtime_home=runtime_home, socket_path=socket_path)
    if current.healthy:
        return current
    _cleanup_stale_daemon(runtime_home=runtime_home, socket_path=socket_path)
    stdio_log_path = daemon_stdio_log_path(runtime_home)
    stdio_log_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    process_env = os.environ.copy()
    process_env["MLX_RUNTIME_HOME"] = str(runtime_home.root)
    process_env["MLX_RUNTIME_UDS_PATH"] = str(socket_path)
    process_env.pop("MLX_RUNTIME_HTTP_HOST", None)
    process_env.pop("MLX_RUNTIME_HTTP_PORT", None)
    process_env.pop("MLX_RUNTIME_HTTP_TOKEN", None)
    process_env.pop("MLX_RUNTIME_ALLOWED_ORIGINS", None)

    with stdio_log_path.open("ab") as stdio_handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "mlxr.core.server"],
            env=process_env,
            stdout=stdio_handle,
            stderr=subprocess.STDOUT,
        )

    transport = httpx.HTTPTransport(uds=str(socket_path), retries=0, trust_env=False)
    client = httpx.Client(
        transport=transport,
        base_url=_DEFAULT_BASE_URL,
        timeout=httpx.Timeout(30.0),
    )
    try:
        _wait_for_health(
            client,
            process=process,
            startup_timeout_seconds=startup_timeout_seconds,
        )
    except Exception as exc:
        _terminate_process(process)
        raise RuntimeDaemonError(
            f"Failed to start the local runtime. See {stdio_log_path}"
        ) from exc
    finally:
        client.close()

    metadata = DaemonMetadata(
        pid=process.pid,
        runtime_home=str(runtime_home.root),
        socket_path=str(socket_path),
        stdio_log_path=str(stdio_log_path),
        started_at=datetime.now(timezone.utc).isoformat(),
        package_version=_installed_package_version(),
    )
    daemon_metadata_path(runtime_home).write_text(
        f"{json.dumps(asdict(metadata), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    # The daemon is intentionally detached after the health check. Once we
    # switch to PID-based management, suppress Popen's "still running" warning
    # for this handoff object.
    process.returncode = 0
    return DaemonStatus(
        status="started",
        runtime_home=str(runtime_home.root),
        socket_path=str(socket_path),
        stdio_log_path=str(stdio_log_path),
        metadata_path=str(daemon_metadata_path(runtime_home)),
        managed=True,
        healthy=True,
        pid=process.pid,
        message="Started a reusable local daemon.",
    )


def stop_runtime_daemon(
    *,
    runtime_home: RuntimeHome,
    socket_path: Path,
) -> DaemonStatus:
    runtime_home.ensure_layout()
    metadata = read_daemon_metadata(runtime_home)
    metadata_path = daemon_metadata_path(runtime_home)
    stdio_log_path = daemon_stdio_log_path(runtime_home)
    if metadata is None:
        if daemon_health(socket_path=socket_path):
            return DaemonStatus(
                status="running",
                runtime_home=str(runtime_home.root),
                socket_path=str(socket_path),
                stdio_log_path=str(stdio_log_path),
                metadata_path=str(metadata_path),
                managed=False,
                healthy=True,
                message="A healthy daemon is running, but it is not managed by mlxr.",
            )
        if socket_path.exists():
            socket_path.unlink()
        return DaemonStatus(
            status="stopped",
            runtime_home=str(runtime_home.root),
            socket_path=str(socket_path),
            stdio_log_path=str(stdio_log_path),
            metadata_path=str(metadata_path),
            managed=False,
            healthy=False,
            message="No managed daemon is running.",
        )

    process = _process_from_pid(metadata.pid)
    if process is not None:
        _terminate_pid(process)
    if socket_path.exists():
        socket_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()
    return DaemonStatus(
        status="stopped",
        runtime_home=str(runtime_home.root),
        socket_path=str(socket_path),
        stdio_log_path=str(stdio_log_path),
        metadata_path=str(metadata_path),
        managed=True,
        healthy=False,
        pid=metadata.pid,
        message="Stopped the managed local daemon.",
    )


def _cleanup_stale_daemon(*, runtime_home: RuntimeHome, socket_path: Path) -> None:
    metadata = read_daemon_metadata(runtime_home)
    metadata_path = daemon_metadata_path(runtime_home)
    if metadata is not None and not _pid_running(metadata.pid):
        if metadata_path.exists():
            metadata_path.unlink()
    if socket_path.exists() and not daemon_health(socket_path=socket_path):
        socket_path.unlink()


def _validate_socket_path(socket_path: Path) -> None:
    encoded = os.fsencode(str(socket_path))
    if len(encoded) <= _MAX_UDS_PATH_BYTES:
        return
    raise RuntimeDaemonError(
        "Local daemon socket path is too long for a Unix domain socket. "
        "Set MLX_RUNTIME_UDS_PATH or pass --uds-path with a shorter path."
    )


def _wait_for_health(
    client: httpx.Client,
    *,
    process: subprocess.Popen[bytes],
    startup_timeout_seconds: float,
) -> None:
    deadline = time.time() + startup_timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeDaemonError("Runtime daemon exited before becoming healthy.")
        try:
            response = client.get("/health")
        except httpx.HTTPError:
            time.sleep(0.1)
            continue
        if response.status_code == 200:
            return
        time.sleep(0.1)
    raise RuntimeDaemonError("Timed out waiting for runtime daemon health check.")


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _pid_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_from_pid(pid: int) -> int | None:
    if not _pid_running(pid):
        return None
    return pid


def _terminate_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not _pid_running(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not _pid_running(pid):
            return
        time.sleep(0.05)


def _installed_package_version() -> str | None:
    for package_name in ("mlxr", "mlx-runtime-cli"):
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None
