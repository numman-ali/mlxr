from __future__ import annotations

from pathlib import Path

import uvicorn
from mlxr.core.runtime import RuntimeHome

from .app import create_app
from .settings import ServerSettings
from .state import RuntimeState


def _uds_socket_path(settings: ServerSettings, runtime_home: RuntimeHome) -> Path:
    if settings.uds_path is not None:
        return Path(settings.uds_path).expanduser()
    return runtime_home.temp_dir / "control-plane.sock"


def main() -> None:
    settings = ServerSettings.from_env()
    settings.validate_startup()
    state = RuntimeState(settings=settings)
    app = create_app(state)
    if settings.http_enabled and settings.http_host is not None:
        uvicorn.run(app, host=settings.http_host, port=settings.http_port)
        return

    runtime_home = RuntimeHome.from_env()
    runtime_home.ensure_layout()
    socket_path = _uds_socket_path(settings, runtime_home)
    socket_dir = socket_path.parent
    socket_dir.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    uvicorn.run(app, uds=str(socket_path))


if __name__ == "__main__":
    main()
