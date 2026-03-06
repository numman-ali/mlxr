from __future__ import annotations

import uvicorn
from mlx_runtime_core import RuntimeHome

from .app import create_app
from .settings import ServerSettings
from .state import RuntimeState


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
    socket_dir = runtime_home.temp_dir
    socket_dir.mkdir(parents=True, exist_ok=True)
    socket_path = socket_dir / "control-plane.sock"
    if socket_path.exists():
        socket_path.unlink()
    uvicorn.run(app, uds=str(socket_path))


if __name__ == "__main__":
    main()
